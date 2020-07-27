import os
import pickle
import torchvision as tv

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed

def get_dataset(args, transform_train, transform_test, num_classes):
    # prepare datasets

    #################################### Train set #############################################
    cifar_train = Cifar100Train(args, train=True, transform=transform_train, download=args.download, pslab_transform = transform_test)

    #################################### Noise corruption ######################################
    if args.noise_type == 'random_in_noise':
        cifar_train.random_in_noise()
        if args.only_clean == 'True':
            cifar_train.get_clean_detected()
    elif args.noise_type == 'real_in_noise':
        cifar_train.real_in_noise()
        if args.only_clean == 'True':
            cifar_train.get_clean_detected()
    else:
        print ('No noise')

    cifar_train.labelsNoisyOriginal = cifar_train.train_labels.copy()

    #################################### Test set #############################################
    testset = tv.datasets.CIFAR100(root='./data', train=False, download=False, transform=transform_test)

    ###########################################################################################

    if args.method == 'ssl' or args.method == 'ssl2':
        if args.ssl_warmup=='True':
            cifar_train.ssl_data_warmup()
        else:
            cifar_train.ssl_data()

        return cifar_train, testset, cifar_train.clean_labels, cifar_train.labeled_idx, cifar_train.unlabeled_idx, cifar_train.noisy_indexes, cifar_train.labelsNoisyOriginal

    return cifar_train, testset, cifar_train.clean_labels, cifar_train.noisy_labels, cifar_train.noisy_indexes,  cifar_train.labelsNoisyOriginal



class Cifar100Train(tv.datasets.CIFAR100):
    # including hard labels & soft labels
    def __init__(self, args, train=True, transform=None, target_transform=None, pslab_transform=None, download=False):
        super(Cifar100Train, self).__init__(args.train_root, train=train, transform=transform, target_transform=target_transform, download=download)
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        # For old pytorch version comment this
        self.train_labels = self.targets
        self.train_data = self.data

        self.args = args
        self.num_classes = self.args.num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.out_data = []
        self.out_labels = []
        self.soft_labels = []
        self.labelsNoisyOriginal = []
        self._num = []
        self._count = 1
        self.prediction = []
        self.confusion_matrix_in = np.array([])
        self.confusion_matrix_out = np.array([])
        self.labeled_idx = []
        self.unlabeled_idx = []

        self.gaus_noise = False
        self.pslab_transform = pslab_transform

        # From in ou split function:
        self.soft_labels = np.zeros((len(self.train_labels), self.num_classes), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.train_data), self.num_classes), dtype=np.float32)
        self._num = int(len(self.train_labels) * self.args.noise_ratio)


    ################# Random in-distribution noise #########################
    def random_in_noise(self):

        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.train_labels))
        clean_labels = np.copy(self.train_labels)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.train_labels[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # train_labels[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.train_labels[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.train_labels[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.train_labels[idxes[i]]] = 1

        self.train_labels = np.asarray(self.train_labels, dtype=np.long)
        self.noisy_labels = np.copy(self.train_labels)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
        self.confusion_matrix_in = (np.ones((self.args.num_classes, self.args.num_classes)) - np.identity(self.args.num_classes))\
                                    *(self.args.noise_ratio/(self.num_classes -1)) + \
                                    np.identity(self.args.num_classes)*(1 - self.args.noise_ratio)


    ##########################################################################


    ################# Real in-distribution noise #########################

    def real_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        ##### Create te confusion matrix #####

        self.confusion_matrix_in = np.identity(self.args.num_classes) * (1 - self.args.noise_ratio)

        idxes = np.random.permutation(len(self.train_labels))
        clean_labels = np.copy(self.train_labels)

        with open('data/cifar-100-python/train', 'rb') as f:
            entry = pickle.load(f, encoding='latin1')

        coarse_targets = np.asarray(entry['coarse_labels'])

        targets = np.array(self.train_labels)
        num_subclasses = self.args.num_classes // 20

        for i in range(20):
            # embed()
            subclass_targets = np.unique(targets[coarse_targets == i])
            clean = subclass_targets
            noisy = np.concatenate([clean[1:], clean[:1]])
            for j in range(num_subclasses):
                self.confusion_matrix_in[clean[j], noisy[j]] = self.args.noise_ratio



        for t in range(len(idxes)):
            self.soft_labels[idxes[t]][self.train_labels[idxes[t]]] = 0  ## Remove soft-label created during label mapping
            current_label = self.train_labels[idxes[t]]
            conf_vec = self.confusion_matrix_in[current_label,:]
            label_sym = np.random.choice(np.arange(0, self.num_classes), p=conf_vec.transpose())
            self.train_labels[idxes[t]] = label_sym
            self.soft_labels[idxes[t]][self.train_labels[idxes[t]]] = 1

            if label_sym == current_label:
                self.clean_indexes.append(idxes[t])
            else:
                self.noisy_indexes.append(idxes[t])

        self.train_labels = np.asarray(self.train_labels, dtype=np.long)
        self.clean_indexes = np.asarray(self.clean_indexes, dtype=np.long)
        self.noisy_indexes = np.asarray(self.noisy_indexes, dtype=np.long)
        self.noisy_labels = self.train_labels
        self.clean_labels = clean_labels


    ##########################################################################
    def update_labels_randRelab(self, result, unlabeled_idx, rand_ratio): # By Eric

        nb_noisy = len(unlabeled_idx) #Amount of unlabeled to relabel
        nb_rand = int(nb_noisy*rand_ratio) #Amount to change randomly
        idx_noisy_all = list(range(nb_noisy))
        idx_noisy_all = np.random.permutation(idx_noisy_all)

        idx_rand = idx_noisy_all[:nb_rand]
        idx_relab = idx_noisy_all[nb_rand:]

        if rand_ratio == 0.0:
            idx_relab = list(range(len(unlabeled_idx)))
            idx_rand = []


        relabel_indexes = list(unlabeled_idx[idx_relab])
        self.soft_labels[relabel_indexes] = result[relabel_indexes]

        self.train_labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis=1).astype(np.int64)

        for idx_num in unlabeled_idx[idx_rand]:
            new_soft = np.ones(self.args.num_classes)
            new_soft = new_soft*(1/self.args.num_classes)
            self.soft_labels[idx_num] = new_soft
            self.train_labels[idx_num] = self.soft_labels[idx_num].argmax(axis=0).astype(np.int64)


        print("Samples relabeled with the prediction: ", str(len(idx_relab)))
        print("Samples relabeled with uniform random relabeling: " + str(len(idx_rand)))

        #embed()

        self._count += 1

    ##########################################################################

    def update_labels(self, result):
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.

        idx = self._count % self.args.epoch_update
        self.prediction[idx-1,:] = result

        if self._count >= self.args.epoch_begin:
            print('Relabeling')
            self.soft_labels = self.prediction.mean(axis = 0)
            # check the paper for this, take the average output prob as s used both in soft and hard labels
            self.train_labels = self.soft_labels.argmax(axis = 1).astype(np.int64)
        self._count += 1
        # save params

        if self._count == self.args.epoch and self.args.method == "SoftRelabeling":
            dst = "RelabelingLabels/" + self.args.dataset + "_" + self.args.network + "_" + self.args.noise_type + "_" + str(self.args.noise_ratio) + ".npz"#FOlder to save new labels
            np.savez(dst, data=self.train_data, hard_labels=self.train_labels, soft_labels=self.soft_labels)


    def __getitem__(self, index):
        img, labels, soft_labels, noisy_labels = self.train_data[index], self.train_labels[index], self.soft_labels[index], self.labelsNoisyOriginal[index]
        # doing this so that it is consistent with all other datasets.
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            labels = self.target_transform(labels)

        return img, labels, soft_labels, index, noisy_labels


    def ssl_data_warmup(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        if self.args.ssl_oracle == "True":
            bmm_noisy_probs = np.zeros((len(self.train_labels)))
            bmm_noisy_probs[np.sort(self.noisy_indexes)] = 1  ## Emulating that noisy samples have a probability 1 of being noisy and clean samples 0 (from the initialization in the line before)
        else:
            bmm_path = 'BMM_probs/'

            if self.args.method == 'ssl':
                bmm_noisy_probs = np.load(
                    bmm_path + 'BMM_probs_' + self.args.dataset + '_R_' + self.args.noise_type + '_' + str(
                        self.args.noise_ratio) + "_" + str(self.args.seed_initialization) + "_" + str(
                        self.args.seed_dataset) + '.npy')

            elif self.args.method == 'ssl2':

                bmm_noisy_probs = np.load(
                    bmm_path + 'BMM_probs_' + self.args.dataset + '_ssl_' + self.args.noise_type + '_' + str(
                        self.args.noise_ratio) + "_" + str(self.args.seed_initialization) + "_" + str(
                        self.args.seed_dataset) + '.npy')

        self.labeled_idx = np.asarray(np.where(bmm_noisy_probs <= self.args.bmm_th)[0])
        self.unlabeled_idx = np.asarray([]) #Empty to avoid relabeling

        for i in range(len(self.unlabeled_idx)): #Set unlabeled idx randomly

            label_sym = np.random.randint(self.num_classes, dtype=np.int32)
            self.train_labels[self.unlabeled_idx[i]] = label_sym
            # Set to zero the soft label to remove the previous soft-label from the noise
            self.soft_labels[self.unlabeled_idx[i]] = np.zeros((self.num_classes), dtype=np.float32)
            self.soft_labels[self.unlabeled_idx[i]][self.train_labels[self.unlabeled_idx[i]]] = 1

        self.train_data = self.train_data[self.labeled_idx]
        self.train_labels = np.array(self.train_labels)[self.labeled_idx]
        self.soft_labels = self.soft_labels[self.labeled_idx]

        # self.train_data = self.train_data[self.labeled_idx]
        # self.train_labels = np.array(self.train_labels)[self.labeled_idx]

    def ssl_data(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        if self.args.ssl_oracle == "True":
            bmm_noisy_probs = np.zeros((len(self.train_labels)))
            bmm_noisy_probs[np.sort(self.noisy_indexes)] = 1  ## Emulating that noisy samples have a probability 1 of being noisy and clean samples 0 (from the initialization in the line before)
        else:
            bmm_path = 'BMM_probs/'

            if self.args.method == 'ssl':
                bmm_noisy_probs = np.load(
                    bmm_path + 'BMM_probs_' + self.args.dataset + '_R_' + self.args.noise_type + '_' + str(
                        self.args.noise_ratio) + "_" + str(self.args.seed_initialization) + "_" + str(
                        self.args.seed_dataset) + '.npy')

            elif self.args.method == 'ssl2':
                bmm_noisy_probs = np.load(
                    bmm_path + 'BMM_probs_' + self.args.dataset + '_ssl_' + self.args.noise_type + '_' + str(
                        self.args.noise_ratio) + "_" + str(self.args.seed_initialization) + "_" + str(
                        self.args.seed_dataset) + '.npy')




        self.labeled_idx = np.asarray(np.where(bmm_noisy_probs <= self.args.bmm_th)[0])
        self.unlabeled_idx = np.asarray(np.where(bmm_noisy_probs > self.args.bmm_th)[0])

        for i in range(len(self.unlabeled_idx)):#Set unlabeled idx randomly
            label_sym = np.random.randint(self.num_classes, dtype=np.int32)
            self.train_labels[self.unlabeled_idx[i]] = label_sym
            #Set to zero the soft label to remove the previous soft-label from the noise
            self.soft_labels[self.unlabeled_idx[i]] = np.zeros((self.num_classes), dtype=np.float32)
            self.soft_labels[self.unlabeled_idx[i]][self.train_labels[self.unlabeled_idx[i]]] = 1
