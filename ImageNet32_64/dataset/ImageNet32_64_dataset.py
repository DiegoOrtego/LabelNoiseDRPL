import os
import pickle

import numpy as np
from PIL import Image
from torch.utils.data import Dataset
import random
import time
from IPython import embed
import torch
import torch.nn.functional as F

_base_folder32 = 'imagenet-32-batches-py'
_base_folder64 = 'imagenet-64-batches-py'

_train_list = ['train_data_batch_1',
               'train_data_batch_2',
               'train_data_batch_3',
               'train_data_batch_4',
               'train_data_batch_5',
               'train_data_batch_6',
               'train_data_batch_7',
               'train_data_batch_8',
               'train_data_batch_9',
               'train_data_batch_10']
_val_list = ['val_data']

_label_file = 'map_clsloc.txt'


def get_dataset(args, transform_train, transform_test, num_classes):
    # prepare datasets

    #################################### Train set #############################################
    total_classes = 1000
    imageNet_train = ImageNet(args, train=True, transform=transform_train, num_classes=num_classes) # Load all 1000 classes in memory
    imageNet_train.in_out_split(total_classes) # Create in-distribution and out-of-distribution split

    #################################### Noise corruption ######################################
    if args.noise_type == 'random_in_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.random_in_noise()
        if args.only_clean == 'True':
            imageNet_train.get_clean_detected()

    elif args.noise_type == 'random_out_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.random_out_noise()
        if args.only_clean == 'True':
            imageNet_train.get_clean_detected()

    elif args.noise_type == 'real_in_noise':
        imageNet_train.label_mapping() # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_in_noise()
        if args.only_clean == 'True':
            imageNet_train.get_clean_detected()

    elif args.noise_type == 'real_in_noise2':
        imageNet_train.label_mapping() # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_in_noise2()


    elif args.noise_type == 'real_out_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_out_noise()
        if args.only_clean == 'True':
            imageNet_train.get_clean_detected()

    elif args.noise_type == 'clean':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.get_clean()


    else:
        print ('No noise')
    ############################################################################################

    imageNet_train.labelsNoisyOriginal = imageNet_train.labels.copy()

    #################################### Test set #############################################
    imageNet_test = ImageNet(args, train=False, transform=transform_test, num_classes=num_classes)  # Load all 1000 test classes in memory
    imageNet_test.in_out_split(total_classes)  # Create in-distribution and out-of-distribution split
    imageNet_test.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
    imageNet_test.labelsNoisyOriginal = imageNet_test.labels.copy()
    ###########################################################################################

    return imageNet_train, imageNet_test, imageNet_train.clean_labels, imageNet_train.noisy_labels, imageNet_train.noisy_indexes,  imageNet_train.labelsNoisyOriginal



def get_dataset_ssl(args, transform_train, transform_test, num_classes):
    # prepare datasets

    #################################### Train set #############################################
    total_classes = 1000
    imageNet_train = ImageNet(args, train=True, transform=transform_train, num_classes=num_classes) # Load all 1000 classes in memory
    imageNet_train.in_out_split(total_classes) # Create in-distribution and out-of-distribution split


    #################################### Noise corruption ######################################
    if args.noise_type == 'random_in_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.random_in_noise()

    elif args.noise_type == 'random_out_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.random_out_noise()


    elif args.noise_type == 'real_in_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_in_noise()

    elif args.noise_type == 'real_in_noise2':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_in_noise2()

    elif args.noise_type == 'real_out_noise':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.real_out_noise()


    elif args.noise_type == 'clean':
        imageNet_train.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
        imageNet_train.get_clean()

    else:
        print('No noise')
    ############################################################################################

    imageNet_train.labelsNoisyOriginal = imageNet_train.labels.copy()

    #################################### Test set #############################################
    imageNet_test = ImageNet(args, train=False, transform=transform_test, num_classes=num_classes)  # Load all 1000 test classes in memory
    imageNet_test.in_out_split(total_classes)  # Create in-distribution and out-of-distribution split
    imageNet_test.label_mapping()  # Rename labels from 0-1000 to 0-num_classes
    imageNet_test.labelsNoisyOriginal = imageNet_test.labels.copy()
    ###########################################################################################

    ### Get clean and noisy data
    if args.ssl_warmup=='True':
        imageNet_train.ssl_data_warmup()
    else:
        imageNet_train.ssl_data()


    return imageNet_train, imageNet_test, imageNet_train.clean_labels, \
           imageNet_train.labeled_idx, imageNet_train.unlabeled_idx, imageNet_train.noisy_indexes, imageNet_train.labelsNoisyOriginal


class ImageNet(Dataset):
    """`ImageNet32 and 64 based on <https://patrykchrabaszcz.github.io/Imagenet32/>`_ dataset.
    Warning: this will load the whole dataset into memory! Please ensure that
    4 GB of memory is available before loading.
    Refer to ``map_clsloc.txt`` for label information.
    The integer labels in this dataset are offset by -1 from ``map_clsloc.txt``
    to make indexing start from 0.
    Args:
        root (string): Root directory of dataset where directory
            ``imagenet-32-batches-py`` exists.
        train (bool, optional): If True, creates dataset from training set, otherwise
            creates from validation set.
        transform (callable, optional): A function/transform that  takes in an PIL image
            and returns a transformed version. E.g, ``transforms.RandomCrop``
        target_transform (callable, optional): A function/transform that takes in the
            target and transforms it.
        exclude (list, optional): List of class indices to omit from dataset.
        remap_labels (bool, optional): If True and exclude is not None, remaps
            remaining class labels so it is contiguous.
    """
    def __init__(self, args, train=True, transform=None,
                 target_transform=None, num_classes=500):
        self.root = os.path.expanduser(args.train_root)
        self.transform = transform
        self.target_transform = target_transform
        self.train = train  # Training set or validation set

        ####### NEW
        self.args = args
        self.num_classes = num_classes
        self.in_index = []
        self.out_index = []
        self.noisy_indexes = []
        self.clean_labels = []
        self.noisy_labels = []
        self.num_classes = num_classes
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
        ################
        # Now load the picked numpy arrays
        if args.dataset == "ImageNet32":
            size = 32
            _base_folder = _base_folder32
        elif args.dataset == "ImageNet64":
            _base_folder = _base_folder64
            size = 64

        if self.train:
            self.data = []
            self.labels = []

            for f in _train_list:
                file = os.path.join(self.root, self.args.dataset, _base_folder, f)

                with open(file, 'rb') as fo:
                    entry = pickle.load(fo, encoding='latin1')
                    self.data.append(entry['data'])
                    self.labels += entry['labels']
            self.data = np.concatenate(self.data)
            self.data = self.data.reshape((-1, 3, size, size))
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.labels = np.array(self.labels) - 1
        else:

            f = _val_list[0]

            file = os.path.join(self.root, self.args.dataset, _base_folder, f)
            with open(file, 'rb') as fo:
                entry = pickle.load(fo, encoding='latin1')
                self.data = entry['data']
                self.labels = entry['labels']

            self.data = self.data.reshape((-1, 3, size, size))
            self.data = self.data.transpose((0, 2, 3, 1))  # Convert to HWC
            self.labels = np.array(self.labels) - 1

        self.labels = self.labels.tolist()


    def __getitem__(self, index):
        """
        Args:
            index (int): Index
        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target, soft_labels, noisy_labels = self.data[index], self.labels[index], self.soft_labels[index], self.labelsNoisyOriginal[index]

        # Doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, soft_labels, index, noisy_labels


    def __len__(self):
        return len(self.data)


    def __repr__(self):
        fmt_str = 'Dataset ' + self.__class__.__name__ + '\n'
        fmt_str += '    Number of datapoints: {}\n'.format(self.__len__())
        tmp = 'train' if self.train is True else 'val'
        fmt_str += '    Split: {}\n'.format(tmp)
        fmt_str += '    Root Location: {}\n'.format(self.root)
        tmp = '    Transforms (if any): '
        fmt_str += '{0}{1}\n'.format(tmp, self.transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        tmp = '    Target Transforms (if any): '
        fmt_str += '{0}{1}'.format(tmp, self.target_transform.__repr__().replace('\n', '\n' + ' ' * len(tmp)))
        return fmt_str

    ######################## In-distribution / Out-of-distribution split #########################
    def in_out_split(self, total_classes):
        np.random.seed(self.args.seed_dataset)

        train = np.array(self.labels)
        in_indexes = []
        out_indexes = []

        class_permutation = np.random.permutation(total_classes)
        self.class_permutation = class_permutation


        cont = 1
        for id in class_permutation:
            indexes = np.where(train == id)[0]
            np.random.shuffle(indexes)
            if cont <= self.num_classes:
                in_indexes.extend(indexes)
            else:
                out_indexes.extend(indexes)

            cont = cont + 1

        np.random.shuffle(in_indexes)
        np.random.shuffle(out_indexes)

        self.out_data = self.data.copy()
        self.out_labels = self.labels.copy()

        self.data = self.data[in_indexes]
        self.labels = np.array(self.labels)[in_indexes]
        self.soft_labels = np.zeros((len(self.labels), self.num_classes), dtype=np.float32)
        self.prediction = np.zeros((self.args.epoch_update, len(self.data), self.num_classes), dtype=np.float32)
        self._num = int(len(self.labels) * self.args.noise_ratio)

        self.out_data = self.out_data[out_indexes]
        self.out_labels = np.array(self.out_labels)[out_indexes]

        self.in_index = in_indexes
        self.out_index = out_indexes

        confusion_matrix_all = np.load('ConfMatrix.npy')
        self.confusion_matrix_in = confusion_matrix_all[np.ix_(class_permutation[:self.num_classes], class_permutation[:self.num_classes])] + 10e-6
        self.confusion_matrix_in = self.confusion_matrix_in / np.sum(self.confusion_matrix_in, 1).transpose()[:, None]
        self.confusion_matrix_out = confusion_matrix_all[np.ix_(class_permutation[self.num_classes:], class_permutation[:self.num_classes])] + 10e-6
        self.confusion_matrix_out = self.confusion_matrix_out / np.sum(self.confusion_matrix_out, 0).transpose()[None, :]##Column normalized due to the way noise is introduced (i.e. labels are kept and observed labels contain the corruption, thus corruption must come from the columns)
    ################################################################################################


    ######################## Label mapping #########################
    def label_mapping(self):
        np.random.seed(self.args.seed_dataset)

        mappingIn = {y: x for x, y in enumerate(self.class_permutation[:self.args.num_classes])}
        self.mappingIn = mappingIn
        labels_orig = self.labels.copy()
        for k, v in sorted(mappingIn.items()):
            idx = labels_orig == k
            self.labels[idx] = v

        mappingOut = {y: x for x, y in enumerate(self.class_permutation[self.args.num_classes:])}
        self.mappingOut = mappingOut
        labels_orig2 = self.out_labels.copy()
        for k, v in sorted(mappingOut.items()):
            idx = labels_orig2 == k
            self.out_labels[idx] = v
    #################################################################

    def update_labels(self, result):
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.

        idx = self._count % self.args.epoch_update
        self.prediction[idx-1,:] = result

        if self._count >= self.args.epoch_begin:
            print('Relabeling')
            self.soft_labels = self.prediction.mean(axis = 0)
            # check the paper for this, take the average output prob as s used both in soft and hard labels
            self.labels = self.soft_labels.argmax(axis = 1).astype(np.int64)
        self._count += 1
        # save params

        if self._count == self.args.epoch and self.args.method == "SoftRelabeling":
            dst = "RelabelingLabels/" + self.args.dataset + "_" + self.args.network + "_" + self.args.noise_type + "_" + str(self.args.noise_ratio) + ".npz"#FOlder to save new labels
            np.savez(dst, data=self.data, hard_labels=self.labels, soft_labels=self.soft_labels)


    def update_labels_randRelab(self, result, unlabeled_idx, rand_ratio): # By Eric
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.

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

        self.labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis=1).astype(np.int64)

        for idx_num in unlabeled_idx[idx_rand]:
            new_soft = np.ones(self.args.num_classes)
            new_soft = new_soft*(1/self.args.num_classes)
            self.soft_labels[idx_num] = new_soft
            self.labels[idx_num] = self.soft_labels[idx_num].argmax(axis=0).astype(np.int64)


        print("Samples relabeled with the prediction: ", str(len(idx_relab)))
        print("Samples relabeled with uniform random relabeling: " + str(len(idx_rand)))

        self._count += 1

    def update_labels_randRelab(self, result, unlabeled_idx, rand_ratio): # By Eric
        # use the average output prob of the network of the past [epoch_update] epochs as s.
        # update from [begin] epoch.


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

        self.labels[relabel_indexes] = self.soft_labels[relabel_indexes].argmax(axis=1).astype(np.int64)

        for idx_num in unlabeled_idx[idx_rand]:
            new_soft = np.ones(self.args.num_classes)
            new_soft = new_soft*(1/self.args.num_classes)
            self.soft_labels[idx_num] = new_soft
            self.labels[idx_num] = self.soft_labels[idx_num].argmax(axis=0).astype(np.int64)


        print("Samples relabeled with the prediction: ", str(len(idx_relab)))
        print("Samples relabeled with uniform random relabeling: " + str(len(idx_rand)))


        self._count += 1


    def reload_labels(self):
        dst = "RelabelingLabels/" + self.args.dataset + "_" + self.args.network + "_" + self.args.noise_type + "_" + str(self.args.noise_ratio) + ".npz"#FOlder to save new labels
        param = np.load(dst)
        self.data = param['data']
        self.labels = param['hard_labels']
        self.soft_labels = param['soft_labels']

    ################# Prune to keep only clean #########################
    def get_clean(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.labels))
        for i in range(len(idxes)):
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1

        self.data = self.data[self._num:]
        self.labels = self.labels[self._num:]
        self.soft_labels = self.soft_labels[self._num:]
    ##########################################################################

    ################# Prune to keep only detected clean #########################
    def get_clean_detected(self):
        # to be more equal, every category can be processed separately
        ##Before calling this function, noise addition is called, where soft_labels are defined (one-hot encoding)
        bmm_path = 'BMM_probs/'

        bmm_noisy_probs = np.load(bmm_path + 'BMM_probs_' + self.args.dataset + '_R_' + self.args.noise_type + '_' + str(self.args.noise_ratio) + "_" + str(self.args.seed_initialization) + "_" + str(self.args.seed_dataset) + '.npy')

        self.labeled_idx = np.asarray(np.where(bmm_noisy_probs <= self.args.bmm_th)[0])

        ##Same as clean_data function, but now we keep only those clean samples indexes according to the BMM
        self.data = self.data[self.labeled_idx]
        self.labels = self.labels[self.labeled_idx]
        self.soft_labels = self.soft_labels[self.labeled_idx]
        self.labelsNoisyOriginal = self.labels #This is just to have this variable with data of the same length (but it does not make sense)

    ##########################################################################


    ################# Random in-distribution noise #########################
    def random_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)
        idxes = np.random.permutation(len(self.labels))
        clean_labels = np.copy(self.labels)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 0 ## Remove soft-label created during label mapping
                # train_labels[idxes[i]] -> another category
                label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                while(label_sym==self.labels[idxes[i]]):#To exclude the original label
                    label_sym = np.random.randint(self.num_classes, dtype=np.int32)
                self.labels[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1

        self.noisy_labels = np.copy(self.labels)
        self.noisy_indexes = noisy_indexes
        self.clean_labels = clean_labels
        self.clean_indexes = clean_indexes
    ##########################################################################

    ################# Random out-of-distribution noise #########################
    def random_out_noise(self):
        np.random.seed(self.args.seed_dataset)

        idxes = np.random.permutation(len(self.labels))
        idxes2 = np.random.permutation(len(self.out_labels))

        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        #clean_labels = np.copy(self.train_labels)
        for i in range(len(idxes)):
            if i < self._num:
                # Train data gets an image from outside the distribution
                self.data[idxes[i]] = self.out_data[idxes2[i]]
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1
        self.noisy_indexes = noisy_indexes
        self.clean_indexes = clean_indexes
        self.clean_labels = np.copy(self.labels)
    ##########################################################################

    ################# Real in-distribution noise #########################

    def real_in_noise(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        confusion_matrix_new = self.confusion_matrix_in[~np.eye(self.confusion_matrix_in.shape[0],dtype=bool)].reshape(self.confusion_matrix_in.shape[0],-1) # Remove diagonal elements
        confusion_matrix_new = confusion_matrix_new / np.sum(confusion_matrix_new,1).transpose()[:,None] # Renormalize each row (for each row: probability for each class to be confunded with the original one)

        idxes = np.random.permutation(len(self.labels))
        clean_labels = np.copy(self.labels)
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        #embed()
        for i in range(len(idxes)):
            if i < self._num:
                self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 0  ## Remove soft-label created during label mapping
                # train_labels[idxes[i]] -> another category
                current_label = self.labels[idxes[i]]
                conf_vec = confusion_matrix_new[current_label,:]
                label_sym = np.random.choice(np.arange(0, self.num_classes-1), p=conf_vec.transpose())
                if label_sym >= current_label:
                    label_sym = label_sym + 1

                self.labels[idxes[i]] = label_sym
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1

        self.noisy_labels = self.labels
        self.noisy_indexes = noisy_indexes
        self.clean_indexes = clean_indexes
        self.clean_labels = clean_labels
    ##########################################################################

    def real_in_noise2(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        confusion_matrix_new2 = self.confusion_matrix_in.copy()
        np.fill_diagonal(confusion_matrix_new2, 0)
        confusion_matrix_new2 = confusion_matrix_new2.transpose() ##Transpose is done here. Later, we will get the row, i.e. the original column.
        confusion_matrix_new2 = confusion_matrix_new2 / np.sum(confusion_matrix_new2, 1).transpose()[:, None]
        confusion_matrix_new = confusion_matrix_new2[~np.eye(confusion_matrix_new2.shape[0], dtype=bool)].reshape(confusion_matrix_new2.shape[0], -1)  # Remove diagonal elements

        clean_labels = np.copy(self.labels)
        idx_classes_noisy = np.empty((self.num_classes,), dtype=object)
        idx_classes_clean = np.empty((self.num_classes,), dtype=object)
        idx_classes_noisy_replicated = np.empty((self.num_classes,), dtype=object)
        noisy_indexes = []
        clean_indexes = []
        ##Create array with indexes that define noisy and clean samples for each class. We will use the noisy indexes to introduce noise.
        for i in range(self.num_classes):
            ## Find all samples in a class
            idx_class = np.where(self.labels == i)[0]
            ## Randomize samples order
            idx_class = np.random.permutation(idx_class)
            ## Number of noisy samples per class dependent on noise ratio
            num_noisy_samples_class = round(self.args.noise_ratio * len(idx_class))
            ## The indexes of the first num_noisy_samples_class are stored to be used as noisy
            idx_classes_noisy[i] = idx_class[:num_noisy_samples_class]
            idx_classes_clean[i] = idx_class[num_noisy_samples_class:]
            ## Keep noisy indexes for tracking purposes
            noisy_indexes = np.concatenate((noisy_indexes, idx_classes_noisy[i]))
            clean_indexes = np.concatenate((clean_indexes, idx_classes_clean[i]))
            ## Replicate the noisy data and labels without corruption to assure selection of the correct image
            if i == 0:
                data_noisy = self.data[idx_classes_noisy[i]].copy()
                labels_set_noisy = self.labels[idx_classes_noisy[i]].copy()

            else:
                data_noisy = np.concatenate((data_noisy, self.data[idx_classes_noisy[i]].copy()))
                labels_set_noisy = np.concatenate((labels_set_noisy, self.labels[idx_classes_noisy[i]].copy()))


            if i==0:
                idx_classes_noisy_replicated[i] = np.array(range(len(data_noisy)))
            else:
                idx_classes_noisy_replicated[i] = np.array(range(idx_classes_noisy_replicated[i - 1][-1] + 1, idx_classes_noisy_replicated[i - 1][-1] + 1 + num_noisy_samples_class))

        ## Define soft labels. They have not been defined before
        idx = np.array(range(len(self.labels)))
        for i in range(len(self.labels)):
            self.soft_labels[idx[i]][self.labels[idx[i]]] = 1

        ## Introduce noise
        for i in range(self.num_classes):
            idx_class_noisy = idx_classes_noisy[i]

            for j in range(len(idx_class_noisy)):
                ### Randomly choose a sample from another class given by the confusion matrix
                current_label = self.labels[idx_class_noisy[j]] ######
                conf_vec = confusion_matrix_new[current_label, :]
                label_sym = np.random.choice(np.arange(0, self.num_classes - 1), p=conf_vec.transpose())
                if label_sym >= current_label:
                    label_sym = label_sym + 1
                ###### Get a sample from class label_sym to replace the current image
                random_idx = np.random.randint(len(idx_classes_noisy_replicated[label_sym]), dtype=np.int32)
                ## We are getting the index of the noisy image randomly from a set of noisy images
                self.data[idx_class_noisy[j]] = data_noisy[idx_classes_noisy_replicated[label_sym][random_idx]]
                ## As we are changing the image, now the clean label has to change for that image
                assert labels_set_noisy[idx_classes_noisy_replicated[label_sym][random_idx]] == label_sym
                clean_labels[idx_class_noisy[j]] = labels_set_noisy[idx_classes_noisy_replicated[label_sym][random_idx]]


        self.noisy_labels = self.labels
        self.noisy_indexes = noisy_indexes.astype(int)
        self.clean_indexes = clean_indexes.astype(int)
        self.clean_labels = clean_labels

    ################# Real out-of-distribution noise #########################
    def real_out_noise(self):
        np.random.seed(self.args.seed_dataset)
        num_classes = 1000 - self.num_classes  ##The parameter num_classes has the #in-distribution classes

        confusion_matrix_new = self.confusion_matrix_out# / np.sum(self.confusion_matrix_out, 1).transpose()[:,None]  # Renormalize each row (for each row: probability for each class to be confunded with the original one)

        idxes = np.random.permutation(len(self.labels))
        noisy_indexes = idxes[0:self._num]
        clean_indexes = idxes[self._num:]
        per_class_samples = []
        num_out_classes = 1000-self.num_classes
        for c in range(num_out_classes):
            class_idx = np.where(self.out_labels == c)[0]  # Index of all samples out with label c
            np.random.shuffle(class_idx)
            per_class_samples.append(class_idx)

        sample_selection_array = np.zeros([num_out_classes], dtype=int)
        for i in range(len(idxes)):
            if i < self._num:
                current_label = self.labels[idxes[i]]
                conf_vec = confusion_matrix_new[:, current_label]
                label_sym = np.random.choice(np.arange(0, num_classes), p=conf_vec)
                idxes_real_out_sample_vec = per_class_samples[label_sym]

                self.data[idxes[i]] = self.out_data[idxes_real_out_sample_vec[sample_selection_array[label_sym]]]
                sample_selection_array[label_sym] = sample_selection_array[label_sym] + 1
            self.soft_labels[idxes[i]][self.labels[idxes[i]]] = 1
        self.noisy_indexes = noisy_indexes
        self.clean_indexes = clean_indexes
        self.clean_labels = np.copy(self.labels)
    ##########################################################################

    def ssl_data(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        if self.args.ssl_oracle == "True":
            bmm_noisy_probs = np.zeros((len(self.labels)))
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
            self.labels[self.unlabeled_idx[i]] = label_sym
            #Set to zero the soft label to remove the previous soft-label from the noise
            self.soft_labels[self.unlabeled_idx[i]] = np.zeros((self.num_classes), dtype=np.float32)
            self.soft_labels[self.unlabeled_idx[i]][self.labels[self.unlabeled_idx[i]]] = 1

    def ssl_data_warmup(self):
        # to be more equal, every category can be processed separately
        np.random.seed(self.args.seed_dataset)

        if self.args.ssl_oracle == "True":
            bmm_noisy_probs = np.zeros((len(self.labels)))
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
            self.labels[self.unlabeled_idx[i]] = label_sym
            # Set to zero the soft label to remove the previous soft-label from the noise
            self.soft_labels[self.unlabeled_idx[i]] = np.zeros((self.num_classes), dtype=np.float32)
            self.soft_labels[self.unlabeled_idx[i]][self.labels[self.unlabeled_idx[i]]] = 1

        self.data = self.data[self.labeled_idx]
        self.labels = np.array(self.labels)[self.labeled_idx]
        self.soft_labels = self.soft_labels[self.labeled_idx]

