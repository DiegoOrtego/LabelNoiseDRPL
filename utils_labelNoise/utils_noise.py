
from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
import scipy.stats as stats
import math
from IPython import embed
import numpy as np
from matplotlib import pyplot as plt
from IPython import embed
from utils.AverageMeter import AverageMeter
from utils.criterion import *
import time
#from sklearn.mixture import GaussianMixture as GMM
import warnings
warnings.filterwarnings('ignore')
from sklearn import preprocessing as preprocessing
from tempfile import TemporaryFile


import os

global_step = 0

##################### Loss tracking and noise modeling #######################

def track_training_loss(args, model, device, train_loader, epoch):
    model.eval()

    with torch.no_grad():
        all_losses_t = torch.Tensor().to(device)
        all_probs = torch.Tensor().to(device)
        all_target = torch.Tensor().to(device)

        all_index = torch.LongTensor().to(device)
        counter = 1

        for batch_idx, (data, _, soft_labels, index, target) in enumerate(train_loader):

            data, target, soft_labels, index = data.to(device), target.to(device), soft_labels.to(device), index.to(device)

            prediction = model(data)
            prediction = F.log_softmax(prediction, dim=1)

            idx_loss = F.nll_loss(prediction, target, reduction = 'none')
            idx_loss.detach_()
            all_losses_t = torch.cat((all_losses_t, idx_loss))
            all_index = torch.cat((all_index, index))


            if counter % 15 == 0:
                print('Tracking iteration [{}/{} ({:.0f}%)]'.format(counter * len(data), len(train_loader.dataset),
                               100. * counter / len(train_loader)))
            counter = counter + 1

        all_losses_t = all_losses_t
        all_losses = torch.zeros(all_losses_t.size())
        all_losses[all_index.cpu()] = all_losses_t.data.cpu()
        loss_tr = all_losses.numpy()

        # outliers detection
        max_perc = np.percentile(loss_tr, 95)
        min_perc = np.percentile(loss_tr, 5)
        loss_tr = loss_tr[(loss_tr<=max_perc) & (loss_tr>=min_perc)]

        bmm_model_maxLoss = torch.FloatTensor([max_perc]).to(device)
        bmm_model_minLoss = torch.FloatTensor([min_perc]).to(device) + 10e-6

        loss_tr = (loss_tr - bmm_model_minLoss.data.cpu().numpy()) / (bmm_model_maxLoss.data.cpu().numpy() - bmm_model_minLoss.data.cpu().numpy() + 1e-6)

        loss_tr[loss_tr>=1] = 1-10e-4
        loss_tr[loss_tr <= 0] = 10e-4
        print('######## Estimating BMM ########')
        bmm_model = BetaMixture1D(max_iters=10)
        bmm_model.fit(loss_tr)

        bmm_model.create_lookup(1)
        print('######## BMM created ########')

        ###Save all BMM probs in last epoch (sorted)
        if (epoch == args.epoch or epoch == args.epoch_begin) and (args.method == "SoftRelabeling" or args.method == "None" or args.method == "Mixup" or args.method == "ssl" or \
                args.method == "ssl2") and args.save_BMM_probs == 'True':

            # save bmm probs ordered
            if args.ssl_oracle == "False":

                with torch.no_grad():

                    all_index = torch.LongTensor().to(device)

                    for batch_idx, (data, _, soft_labels, index, target) in enumerate(train_loader):

                        data, target, soft_labels, index = data.to(device), target.to(device), soft_labels.to(device), index.to(device)

                        prediction = model(data)
                        probs = F.softmax(prediction, dim=1)
                        prediction = F.log_softmax(prediction, dim=1)
                        batch_losses = F.nll_loss(prediction, target, reduction='none')
                        batch_losses.detach_()
                        all_index = torch.cat((all_index, index))

                        if batch_idx == 0:
                            loss_t = batch_losses.cpu().numpy()
                            probs_t = probs.detach_().cpu().numpy()
                        else:
                            loss_t = np.concatenate((loss_t, batch_losses.cpu().numpy()))
                            probs_t = np.concatenate((probs_t, probs.detach_().cpu().numpy()),axis=0)


                        if epoch == args.epoch:
                            batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
                            batch_losses[batch_losses >= 1] = 1 - 10e-4
                            batch_losses[batch_losses <= 0] = 10e-4

                            B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

                            if batch_idx==0:
                                B_t = B
                            else:
                                B_t = np.concatenate((B_t, B))

                loss_sorted = np.zeros(len(loss_t))
                loss_sorted[all_index.cpu().numpy()] = loss_t

                probs_sorted = np.zeros(probs_t.shape)
                probs_sorted[all_index.cpu().numpy()] = probs_t

                bmm_path = 'BMM_probs/'

                if args.method == "SoftRelabeling":
                    if epoch == args.epoch:
                        B_sorted = np.zeros(len(B_t))
                        B_sorted[all_index.cpu().numpy()] = B_t
                        np.save(bmm_path + 'BMM_probs_' + args.dataset + '_R_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', B_sorted)
                        np.save(bmm_path + 'LossEnd_' + args.dataset + '_R_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', loss_sorted)
                        np.save(bmm_path + 'ProbsEnd_' + args.dataset + '_R_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', probs_sorted)

                elif args.method == "ssl":

                    if epoch == args.epoch:
                        B_sorted = np.zeros(len(B_t))
                        B_sorted[all_index.cpu().numpy()] = B_t
                        np.save(bmm_path + 'BMM_probs_' + args.dataset + '_ssl_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', B_sorted)
                        np.save(bmm_path + 'LossEnd_' + args.dataset + '_ssl_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', loss_sorted)
                        np.save(bmm_path + 'ProbsEnd_' + args.dataset + '_ssl_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy',
                                probs_sorted)
                elif args.method == "ssl2":

                    if epoch == args.epoch:
                        B_sorted = np.zeros(len(B_t))
                        B_sorted[all_index.cpu().numpy()] = B_t
                        np.save(bmm_path + 'BMM_probs_' + args.dataset + '_ssl2_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', B_sorted)
                        np.save('BMM_probs/LossEnd_' + args.dataset + '_ssl2_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', loss_sorted)
                        np.save('BMM_probs/ProbsEnd_' + args.dataset + '_ssl2_' + args.noise_type + '_' + str(args.noise_ratio) + "_" + str(args.seed_initialization) + "_" + str(args.seed_dataset) + '.npy', probs_sorted)

    return all_losses.data.numpy(), bmm_model, bmm_model_maxLoss, bmm_model_minLoss

##############################################################################

############################### Cross-entropy loss training ###############################
def CE_loss(preds, labels, device, args, criterion):
    # introduce prior prob distribution p



    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)

    # ignore constant
    loss_all = criterion(preds, labels)
    loss = torch.mean(loss_all)
    return prob, loss, loss_all

def train_CrossEntropy(args, model, device, train_loader, optimizer, epoch, num_classes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    criterion = nn.CrossEntropyLoss()
    counter = 1
    for batch_idx, (images, labels, soft_labels, index, original_noisy_labels) in enumerate(train_loader):

        images, labels, soft_labels, index = images.to(device), labels.to(device), soft_labels.to(device), index.to(device)

        # compute output
        outputs = model(images)
        prob, loss, _ = CE_loss(outputs, labels, device, args, criterion)
        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 5])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                prec1,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum



############################### Mixup training ###############################

def mixup_data(x, y, alpha=1.0, device='cuda'):
    '''Returns mixed inputs, pairs of targets, and lambda'''
    if alpha > 0:
        lam = np.random.beta(alpha, alpha)
    else:
        lam = 1

    batch_size = x.size()[0]
    if device=='cuda':
        index = torch.randperm(batch_size).cuda()
    else:
        index = torch.randperm(batch_size)

    mixed_x = lam * x + (1 - lam) * x[index, :]
    y_a, y_b = y, y[index]
    return mixed_x, y_a, y_b, lam


def mixup_criterion(pred, y_a, y_b, lam, criterion):
    prob = F.softmax(pred, dim=1)

    return prob, lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

def train_mixup(args, model, device, train_loader, optimizer, epoch, num_classes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()


    model.train()


    end = time.time()

    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
    criterion = nn.CrossEntropyLoss()
    counter = 1

    for batch_idx, (images, labels, soft_labels, index, original_noisy_labels) in enumerate(train_loader):

        images, labels, soft_labels, index = images.to(device), labels.to(device), soft_labels.to(device), index.to(device)

        inputs, targets_a, targets_b, lam = mixup_data(images, labels, args.alpha, device)

        # compute output
        outputs = model(inputs)
        prob, loss = mixup_criterion(outputs, targets_a, targets_b, lam, criterion)
        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 5])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                prec1,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum


###################################################################################


############################### Relabeling Cross-entropy loss training ###############################
def joint_opt_loss(preds, soft_labels, device, args, num_classes, epoch):
    # introduce prior prob distribution p

    p = torch.ones(num_classes).to(device) / num_classes

    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)

    L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))
    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))


    loss = L_c + args.reg_term1 * L_p + args.reg_term2 * L_e

    return prob, loss

def train_RelabelingCrossEntropy(args, model, device, train_loader, optimizer, epoch, num_classes):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)

    counter = 1
    for batch_idx, (images, labels, soft_labels, index, original_noisy_labels) in enumerate(train_loader):

        images, labels, soft_labels, index = images.to(device), labels.to(device), soft_labels.to(device), index.to(device)

        # compute output
        outputs = model(images)
        prob, loss = joint_opt_loss(outputs, soft_labels, device, args, num_classes, epoch)
        results[index.cpu().detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 5])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step

        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                prec1,
                optimizer.param_groups[0]['lr']))
        counter = counter + 1
    # update soft labels
    train_loader.dataset.update_labels(results)

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum
###################################################################################


################################ Semi-supervised learning #####################################



def loss_mixup_reg_ep(preds, labels, targets_a, targets_b, device, lam, args):
    prob = F.softmax(preds, dim=1)
    prob_avg = torch.mean(prob, dim=0)
    p = torch.ones(args.num_classes).to(device) / args.num_classes

    # L_c = -torch.mean(torch.sum(soft_labels * F.log_softmax(preds, dim=1), dim=1))   # Soft labels
    mixup_loss_a = -torch.mean(torch.sum(targets_a * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss_b = -torch.mean(torch.sum(targets_b * F.log_softmax(preds, dim=1), dim=1))
    mixup_loss = lam * mixup_loss_a + (1 - lam) * mixup_loss_b

    L_p = -torch.sum(torch.log(prob_avg) * p)
    L_e = -torch.mean(torch.sum(prob * F.log_softmax(preds, dim=1), dim=1))

    loss = mixup_loss + args.reg_term1 * L_p + args.reg_term2 * L_e
    return prob, loss


def train_CrossEntropy_partialRelab(args, model, device, train_loader, optimizer, epoch, unlabeled_idx, labeled_idx, bmm_model, bmm_model_maxLoss, bmm_model_minLoss):
    batch_time = AverageMeter()
    train_loss = AverageMeter()
    top1 = AverageMeter()
    top5 = AverageMeter()

    # switch to train mode
    model.train()

    end = time.time()

    results = np.zeros((len(train_loader.dataset), args.num_classes), dtype=np.float32)

    print("SSL pseudo-labeling with mixup and label noise regularization")

    #############################################################################################

    counter = 1
    all_index = torch.LongTensor().to(device)
    for batch_idx, (images, labels, soft_labels, index2, original_noisy_labels) in enumerate(train_loader):
        if len(index2)<6:##Discard mini-batch if it's small
            continue

        images = images.to(device)
        labels = labels.to(device)
        soft_labels = soft_labels.to(device)
        index2 = index2.to(device)
        original_noisy_labels = original_noisy_labels.to(device)
        # z_exp_labels = z_exp_labels.to(device)

        optimizer.zero_grad()
        output_x1 = model(images)
        output_x1.detach_()
        optimizer.zero_grad()

        alpha = args.alpha
        images, targets_a, targets_b, lam = mixup_data(images, soft_labels, alpha, device)

        # compute output
        outputs = model(images)

        prob = F.softmax(output_x1, dim=1)
        prob_mixup, loss = loss_mixup_reg_ep(outputs, labels, targets_a, targets_b, device, lam, args)
        outputs = output_x1

        results[index2.detach().cpu().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        prec1, prec5 = accuracy_v2(outputs, labels, top=[1, 5])
        train_loss.update(loss.item(), images.size(0))
        top1.update(prec1.item(), images.size(0))
        top5.update(prec5.item(), images.size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ### BMM probabilities on the fly
        if epoch > args.full_relab_begin:
            prediction = F.log_softmax(output_x1, dim=1)
            batch_losses = F.nll_loss(prediction, original_noisy_labels, reduction='none')
            batch_losses.detach_()
            all_index = torch.cat((all_index, index2))
            batch_losses = (batch_losses - bmm_model_minLoss) / (bmm_model_maxLoss - bmm_model_minLoss + 1e-6)
            batch_losses[batch_losses >= 1] = 1 - 10e-4
            batch_losses[batch_losses <= 0] = 10e-4

            B = bmm_model.look_lookup(batch_losses, bmm_model_maxLoss, bmm_model_minLoss)

            if batch_idx == 0:
                B_t = B
            else:
                B_t = np.concatenate((B_t, B))

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if counter % 15 == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}, Accuracy: {:.0f}%, Learning rate: {:.6f}'.format(
                epoch, counter * len(images), len(train_loader.dataset),
                       100. * counter / len(train_loader), loss.item(),
                       prec1, optimizer.param_groups[0]['lr']))
        counter = counter + 1

    if epoch > args.full_relab_begin:
        B_sorted = np.zeros(len(B_t))
        B_sorted[all_index.cpu().numpy()] = B_t


    # update soft labels
    if epoch > args.full_relab_begin:
        unlabeled_idx = np.concatenate((labeled_idx, unlabeled_idx))
    train_loader.dataset.update_labels_randRelab(results, unlabeled_idx, args.random_relab)

    return train_loss.avg, top5.avg, top1.avg, batch_time.sum, unlabeled_idx, labeled_idx

###########################################################################################################################

def test_cleaning(args, model, device, test_loader):
    model.eval()
    loss_per_batch = []
    acc_val_per_batch =[]
    test_loss = 0
    correct = 0
    with torch.no_grad():
        if args.dataset=="ImageNet32" or args.dataset=="ImageNet64":
            for batch_idx, (data, target, _, _, _) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = F.log_softmax(output, dim=1)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                loss_per_batch.append(F.nll_loss(output, target).item())
                pred = output.max(1, keepdim=True)[1]  # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc_val_per_batch.append(100. * correct / ((batch_idx + 1) * args.test_batch_size))
        else:
            for batch_idx, (data, target) in enumerate(test_loader):
                data, target = data.to(device), target.to(device)
                output = model(data)
                output = F.log_softmax(output, dim=1)
                test_loss += F.nll_loss(output, target, reduction='sum').item()
                loss_per_batch.append(F.nll_loss(output, target).item())
                pred = output.max(1, keepdim=True)[1] # get the index of the max log-probability
                correct += pred.eq(target.view_as(pred)).sum().item()
                acc_val_per_batch.append(100. * correct / ((batch_idx+1)*args.test_batch_size))

    test_loss /= len(test_loader.dataset)
    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))

    loss_per_epoch = [np.average(loss_per_batch)]
    #acc_val_per_epoch = [np.average(acc_val_per_batch)]
    acc_val_per_epoch = [np.array(100. * correct / len(test_loader.dataset))]

    return (loss_per_epoch, acc_val_per_epoch)

################### CODE FOR THE BETA MODEL  ########################

def weighted_mean(x, w):
    return np.sum(w * x) / np.sum(w)

def fit_beta_weighted(x, w):
    x_bar = weighted_mean(x, w)
    s2 = weighted_mean((x - x_bar)**2, w)
    alpha = x_bar * ((x_bar * (1 - x_bar)) / s2 - 1)
    beta = alpha * (1 - x_bar) /x_bar
    return alpha, beta


class BetaMixture1D(object):
    def __init__(self, max_iters=10,
                 alphas_init=[1, 2],
                 betas_init=[2, 1],
                 weights_init=[0.5, 0.5]):
        self.alphas = np.array(alphas_init, dtype=np.float64)
        self.betas = np.array(betas_init, dtype=np.float64)
        self.weight = np.array(weights_init, dtype=np.float64)
        self.max_iters = max_iters
        self.lookup = np.zeros(100, dtype=np.float64)
        self.lookup_resolution = 100
        self.lookup_loss = np.zeros(100, dtype=np.float64)
        self.eps_nan = 1e-12

    def likelihood(self, x, y):
        return stats.beta.pdf(x, self.alphas[y], self.betas[y])

    def weighted_likelihood(self, x, y):
        return self.weight[y] * self.likelihood(x, y)

    def probability(self, x):
        return sum(self.weighted_likelihood(x, y) for y in range(2))

    def posterior(self, x, y):
        return self.weighted_likelihood(x, y) / (self.probability(x) + self.eps_nan)

    def responsibilities(self, x):
        r =  np.array([self.weighted_likelihood(x, i) for i in range(2)])
        # there are ~200 samples below that value
        r[r <= self.eps_nan] = self.eps_nan
        r /= r.sum(axis=0)
        return r

    def score_samples(self, x):
        return -np.log(self.probability(x))

    def fit(self, x):
        x = np.copy(x)

        # EM on beta distributions unsable with x == 0 or 1
        eps = 1e-4
        x[x >= 1 - eps] = 1 - eps
        x[x <= eps] = eps

        for i in range(self.max_iters):

            # E-step
            r = self.responsibilities(x)

            # M-step
            self.alphas[0], self.betas[0] = fit_beta_weighted(x, r[0])
            self.alphas[1], self.betas[1] = fit_beta_weighted(x, r[1])
            self.weight = r.sum(axis=1)
            self.weight /= self.weight.sum()

        return self

    def predict(self, x):
        return self.posterior(x, 1) > 0.5

    def create_lookup(self, y):
        x_l = np.linspace(0+self.eps_nan, 1-self.eps_nan, self.lookup_resolution)
        lookup_t = self.posterior(x_l, y)
        lookup_t[np.argmax(lookup_t):] = lookup_t.max()
        self.lookup = lookup_t
        self.lookup_loss = x_l # I do not use this one at the end

    def look_lookup(self, x, loss_max, loss_min):
        x_i = x.clone().cpu().numpy()
        x_i = np.array((self.lookup_resolution * x_i).astype(int))
        x_i[x_i < 0] = 0
        x_i[x_i == self.lookup_resolution] = self.lookup_resolution - 1
        return self.lookup[x_i]

    def plot(self):
        x = np.linspace(0, 1, 100)
        plt.plot(x, self.weighted_likelihood(x, 0), label='negative')
        plt.plot(x, self.weighted_likelihood(x, 1), label='positive')
        plt.plot(x, self.probability(x), lw=2, label='mixture')

    def __str__(self):
        return 'BetaMixture1D(w={}, a={}, b={})'.format(self.weight, self.alphas, self.betas)
