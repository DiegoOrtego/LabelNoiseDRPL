
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import torchvision.transforms as transforms
import argparse
import logging
import os
import time
import random
import sys

sys.path.append('../utils_labelNoise')

from dataset.cifar10_dataset import *

import torch.utils.data as data
from torch import optim
from torchvision import datasets, transforms, models

from utils_noise import *
from utils.criterion import accuracy_v2
from utils.AverageMeter import AverageMeter
import models as mod

logger = logging.getLogger()
logger.setLevel(logging.INFO)

def save_checkpoint(state, filename='checkpoint.pth.tar'):
    torch.save(state, filename)

def parse_args():
    parser = argparse.ArgumentParser(description='command for the first train')
    parser.add_argument('--lr', type=float, default=0.1, help='learning rate')
    parser.add_argument('--batch_size', type=int, default=128, help='#images in each mini-batch')
    parser.add_argument('--test_batch_size', type=int, default=100, help='#images in each mini-batch')
    parser.add_argument('--cuda_dev', type=int, default=0, help='GPU to select')
    parser.add_argument('--epoch', type=int, default=200, help='training epoches')
    parser.add_argument('--num_classes', type=int, default=100, help='Number of in-distribution classes')
    parser.add_argument('--wd', type=float, default=1e-4, help='weight decay')
    parser.add_argument('--momentum', default=0.9, type=float, help='momentum')
    parser.add_argument('--noise_type', default='random_in_noise', help='noise type of the dataset')
    parser.add_argument('--train_root', default='./data', help='root for train data')
    parser.add_argument('--epoch_begin', type=int, default=40, help='the epoch to begin update labels')
    parser.add_argument('--epoch_update', type=int, default=10, help='#epoch to average to update soft labels')
    parser.add_argument('--noise_ratio', type=float, default=0.7, help='percent of noise')
    parser.add_argument('--out', type=str, default='./data/model_data', help='Directory of the output')
    parser.add_argument('--reg_term1', type=float, default=0.8, help='Hyper param for loss')
    parser.add_argument('--reg_term2', type=float, default=0.4, help='Hyper param for loss')
    parser.add_argument('--alpha', type=int, default=1, help='Beta distribution parameter for mixup')
    parser.add_argument('--download', type=bool, default=False, help='download dataset')
    parser.add_argument('--network', type=str, default='PR18', help='the backbone of the network')
    parser.add_argument('--seed_initialization', type=int, default=1, help='random seed (default: 1)')
    parser.add_argument('--seed_dataset', type=int, default=42, help='random seed (default: 1)')
    parser.add_argument('--M', action='append', type=int, default=[], help="Milestones for the LR sheduler")
    parser.add_argument('--experiment_name', type=str, default = 'Proof',help='name of the experiment (for the output files)')
    parser.add_argument('--method', type=str, default='None', help='ssl or ssl2 (iteration 1 or two of semi-supervised)')
    parser.add_argument('--bmm_th', type=float, default=0.05, help='BMM threshold to get clean/noisy samples and do ssl')
    parser.add_argument('--ssl_warmup', type=str, default='False', help='Warm-up with detected clean labels before BMM thresholding')
    parser.add_argument('--random_relab', type=float, default=0.0, help='ratio of labeles to relabel randomly')
    parser.add_argument('--initial_epoch', type=int, default=1, help='First epoch (1 or higher if there is a warmup')
    parser.add_argument('--full_relab', type=str, default='None', help='Switch from SSL to full-relabeling. Options are: None, Normal, Curriculum')
    parser.add_argument('--full_relab_begin', type=int, default=9999, help='Epoch to start full relabeling')
    parser.add_argument('--initial_th', type=float, default=0.001, help='initial th for BMM that decides moving labeled to unlabeled')
    parser.add_argument('--save_BMM_probs', type=str, default='False', help='Save per-sample BMM noisy probs (sorted)')
    parser.add_argument('--dataset', type=str, default='ImageNet32', help='ImageNet32, ImageNet64, CIFAR-10, CIFAR-100')
    parser.add_argument('--ssl_oracle', type=str, default='False', help='Train SSL with an oracle clean/noisy detection')
    parser.add_argument('--only_clean', type=str, default='False', help='True if we want to train with detected clean data only for a particular type of noise')

    args = parser.parse_args()
    return args

def data_config(args, transform_train, transform_test, num_classes):

    if args.method=="ssl" or args.method=="ssl2":
        #trainset, testset, true_labels, noisy_labels, labeled_idx, unlabeled_idx, train_noisy_indexes, all_labels = get_dataset(args, transform_train, transform_test, num_classes)
        trainset, testset, true_labels, labeled_idx, unlabeled_idx, train_noisy_indexes, all_labels = get_dataset(args, transform_train, transform_test, num_classes)
        trackset, _, _, _, _, _, _ = get_dataset(args, transform_train, transform_test, num_classes)
    else:
        trainset, testset, clean_labels, noisy_labels, noisy_indexes, all_labels = get_dataset(args, transform_train, transform_test, num_classes)
        trackset, _, _, _, _, _ = get_dataset(args, transform_train, transform_test, num_classes)


    train_loader = torch.utils.data.DataLoader(trainset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(testset, batch_size=args.test_batch_size, shuffle=False, num_workers=8, pin_memory=True)


    track_loader = torch.utils.data.DataLoader(trackset, batch_size=args.batch_size, shuffle=True, num_workers=8, pin_memory=True)

    print('############# Data loaded #############')

    if args.method == "ssl" or args.method == "ssl2":
        return train_loader, test_loader, track_loader, true_labels, labeled_idx, unlabeled_idx, train_noisy_indexes, trainset, all_labels
    else:
        return train_loader, test_loader, track_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels



def main(args):
    os.environ['CUDA_VISIBLE_DEVICES'] = str(args.cuda_dev)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    torch.backends.cudnn.deterministic = True  # fix the GPU to deterministic mode
    torch.manual_seed(args.seed_initialization)  # CPU seed
    if device == "cuda":
        torch.cuda.manual_seed_all(args.seed_initialization)  # GPU seed

    random.seed(args.seed_initialization)  # python seed for image transformation

    if args.dataset == 'CIFAR-10':
        mean = [0.4914, 0.4822, 0.4465]
        std = [0.2023, 0.1994, 0.2010]
    elif args.dataset == 'CIFAR-100':
        mean = [0.5071, 0.4867, 0.4408]
        std = [0.2675, 0.2565, 0.2761]
    elif args.dataset == 'ImageNet32' or args.dataset == 'ImageNet64':
        mean = [0.485, 0.456, 0.406]
        std = [0.229, 0.224, 0.225]

    if args.dataset == 'ImageNet32':

        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif args.dataset == 'ImageNet64':
        transform_train = transforms.Compose([
            transforms.RandomCrop(64, padding=8),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])

    elif args.dataset == "CIFAR-10" or args.dataset == "CIFAR-100":
        transform_train = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean, std),
        ])


    transform_test = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(mean, std),
    ])

    # data loader
    num_classes = args.num_classes
    if args.method == "ssl" or args.method == "ssl2":
        train_loader, test_loader, track_loader, true_labels, labeled_idx, unlabeled_idx, noisy_indexes, trainset, all_labels = data_config(args, transform_train, transform_test,  num_classes)
    else:
        train_loader, test_loader, track_loader, clean_labels, noisy_labels, noisy_indexes, trainset, all_labels = data_config(args, transform_train, transform_test, num_classes)

    st = time.time()

    model = mod.PreActResNet18(num_classes=num_classes).to(device)

    print('Total params: {:.2f} M'.format(sum(p.numel() for p in model.parameters()) / 1000000.0))

    milestones = args.M

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.wd)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=milestones, gamma=0.1)


    loss_train_epoch = []
    loss_val_epoch = []

    acc_train_per_epoch = []
    acc_val_per_epoch = []

    loss_per_epoch_train = []

    exp_path = os.path.join('./', 'noise_models_' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset), str(args.noise_ratio))
    res_path = os.path.join('./', 'metrics' + args.network + '_{0}_SI{1}_SD{2}'.format(args.experiment_name, args.seed_initialization, args.seed_dataset), str(args.noise_ratio))

    if not os.path.isdir(res_path):
        os.makedirs(res_path)

    if not os.path.isdir(exp_path):
        os.makedirs(exp_path)

    if not os.path.isdir("BMM_probs"):
        os.makedirs("BMM_probs")
    if not os.path.isdir("checkpoints"):
        os.makedirs("checkpoints")
    if not os.path.isdir("RelabelingLabels"):
        os.makedirs("RelabelingLabels")


    if args.method == "ssl" or args.method == "ssl2":
        np.save(res_path + '/' + str(args.noise_ratio) + '_true_labels.npy', np.asarray(true_labels))
        np.save(res_path + '/' + str(args.noise_ratio) + '_unlab_idx.npy', unlabeled_idx)
        np.save(res_path + '/' + str(args.noise_ratio) + '_lab_idx.npy', labeled_idx)
    else:
        np.save(res_path + '/' + str(args.noise_ratio) + '_true_labels.npy', np.asarray(clean_labels))
        np.save(res_path + '/' + str(args.noise_ratio) + '_noisy_labels.npy', np.asarray(noisy_labels))

    np.save(res_path + '/' + str(args.noise_ratio) + '_diff_labels.npy', noisy_indexes)
    np.save(res_path + '/' + str(args.noise_ratio) + '_all_labels.npy', all_labels)

    bmm_model = 0
    bmm_model_maxLoss = 0
    bmm_model_minLoss = 0
    cont = 0

    load = False
    save = False


    if args.ssl_warmup == 'True':
        load = False
        save = True

    if load:

        if args.method == 'ssl':
            path = './checkpoints/warmUp_{0}_{1}_{2}_{3}_SI{4}_SD{5}.hdf5'.format(args.initial_epoch, args.noise_type, args.noise_ratio, args.network, args.seed_initialization, args.seed_dataset)
        elif args.method == 'ssl2':
            path = './checkpoints/warmUp_{0}_{1}_{2}_{3}_SI{4}_SD{5}_2.hdf5'.format(args.initial_epoch, args.noise_type, args.noise_ratio, args.network, args.seed_initialization, args.seed_dataset)


        checkpoint = torch.load(path)
        print("Load model in epoch " + str(args.initial_epoch))
        print("Path loaded: ", path)
        model.load_state_dict(checkpoint['state_dict'])

        print("Relabeling the unlabeled samples...")
        model.eval()
        initial_rand_relab = args.random_relab
        results = np.zeros((len(train_loader.dataset), num_classes), dtype=np.float32)
        for images, labels, soft_labels, index, z_exp_labels in train_loader:
            images = images.to(device)
            outputs = model(images)
            prob = F.softmax(outputs, dim=1)
            results[index.detach().numpy().tolist()] = prob.cpu().detach().numpy().tolist()

        train_loader.dataset.update_labels_randRelab(results, unlabeled_idx, initial_rand_relab)
        print("Start training...")


    for epoch in range(1, args.epoch + 1):

        scheduler.step()

        if args.method == "ssl" or args.method == "ssl2":
            print(args.experiment_name, "Detected clean samples: " + str(len(labeled_idx)))
            loss_train, top_5_train_ac, top1_train_ac, train_time, unlabeled_idx, labeled_idx = train_CrossEntropy_partialRelab(args, model, device, train_loader, optimizer, epoch, unlabeled_idx, labeled_idx, bmm_model, bmm_model_maxLoss, bmm_model_minLoss)
        elif args.method == "SoftRelabeling":
            loss_train, top_5_train_ac, top1_train_ac, train_time = train_RelabelingCrossEntropy(args, model, device, train_loader, optimizer, epoch, num_classes)

        ### Training tracking loss
        print('######## Tracking loss ########')
        epoch_losses_train, bmm_model, bmm_model_maxLoss, bmm_model_minLoss = track_training_loss(args, model, device, track_loader, epoch)

        loss_per_epoch_train.append(epoch_losses_train)
        loss_train_epoch += [loss_train]

        # Validation
        print('######## Test ########')
        loss_val, acc_val_per_epoch_i = test_cleaning(args, model, device, test_loader)

        loss_val_epoch += loss_val
        acc_train_per_epoch += [top1_train_ac]
        acc_val_per_epoch += acc_val_per_epoch_i

        print('Epoch time: {:.2f} seconds\n'.format(time.time()-st))
        st = time.time()

        if epoch == 1:#initial_epoch:
            best_acc_val = acc_val_per_epoch_i[-1]
            snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%d_bestAccVal_%.5f' % (
                epoch, loss_val_epoch[-1], acc_val_per_epoch_i[-1], args.noise_ratio, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
        else:
            if acc_val_per_epoch_i[-1] > best_acc_val:
                best_acc_val = acc_val_per_epoch_i[-1]

                if cont > 0:
                    try:
                        os.remove(os.path.join(exp_path, 'opt_' + snapBest + '.pth'))
                        os.remove(os.path.join(exp_path, snapBest + '.pth'))
                    except OSError:
                        pass
                snapBest = 'best_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestAccVal_%.5f' % (
                    epoch, loss_val_epoch[-1], acc_val_per_epoch_i[-1], args.noise_ratio, best_acc_val)
                torch.save(model.state_dict(), os.path.join(exp_path, snapBest + '.pth'))
                torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapBest + '.pth'))

        cont += 1

        if epoch == args.epoch:
            snapLast = 'last_epoch_%d_valLoss_%.5f_valAcc_%.5f_noise_%.2f_bestValLoss_%.5f' % (
                epoch, loss_val_epoch[-1], acc_val_per_epoch_i[-1], args.noise_ratio, best_acc_val)
            torch.save(model.state_dict(), os.path.join(exp_path, snapLast + '.pth'))
            torch.save(optimizer.state_dict(), os.path.join(exp_path, 'opt_' + snapLast + '.pth'))

        ### Saving model to load it again
        if args.ssl_warmup == 'True':
            cond = (epoch == args.epoch)
            name = 'warmUp'
            save = True
        else:
            cond = False

        if cond and save:
            print("Saving models...")
            if args.method == 'ssl':
                path = './checkpoints/{0}_{1}_{2}_{3}_{4}_SI{5}_SD{6}.hdf5'.format(name, epoch, args.noise_type, args.noise_ratio, args.network, args.seed_initialization, args.seed_dataset)
            elif args.method == 'ssl2':
                path = './checkpoints/{0}_{1}_{2}_{3}_{4}_SI{5}_SD{6}_2.hdf5'.format(name, epoch, args.noise_type, args.noise_ratio, args.network, args.seed_initialization, args.seed_dataset)

            save_checkpoint({'state_dict': model.state_dict()}, filename=path)


        ####### Saving metrics
        # Save losses:
        np.save(res_path + '/' + str(args.noise_ratio) + '_LOSS_epoch_train.npy', np.asarray(loss_train_epoch))
        np.save(res_path + '/' + str(args.noise_ratio) + '_LOSS_epoch_val.npy', np.asarray(loss_val_epoch))

        # save accuracies:
        np.save(res_path + '/' + str(args.noise_ratio) + '_accuracy_per_epoch_train.npy', np.asarray(acc_train_per_epoch))
        np.save(res_path + '/' + str(args.noise_ratio) + '_accuracy_per_epoch_val.npy', np.asarray(acc_val_per_epoch))

        # save individual losses per epoch
        np.save(res_path + '/' + str(args.noise_ratio) + '_losses_per_epoch_train.npy', np.asarray(loss_per_epoch_train))

    print('Best ac:%f' % best_acc_val)



if __name__ == "__main__":
    args = parse_args()
    logging.info(args)
    # train
    main(args)
