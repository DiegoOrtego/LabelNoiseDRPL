import os
import numpy as np
import matplotlib.pyplot as plt
from IPython import embed



#res_path = './metricsPreResNet18_R_real_in_ssl_wup'
#res_path = './metricsPreResNet18_R_real_in_ssl'
#res_path = './metricsPreResNet18_R_real_in_ssl_mix1'
#res_path = './metricsPreResNet18_R_real_in01_ssl'
#res_path = './metricsPreResNet18_R_real_in0'


#res_path = './metricsPR18_DynBootSoft_real_in_SI1_SD42'
#res_path = './metricsWR28_DynBootSoft_real_in_SI1_SD42'
#res_path = './metricsWR28_DynBootSoft_random_in_SI1_SD42'

#res_path = './metricsPR18_Mixup_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase2_Wup_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase3_SSL_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase3_SSL_random_in_SI1_SD42'
#res_path = './metricsPR18_Relab_random_in_stage1_SI1_SD42'
#res_path = 'OLDexp/metricsPR18_Relab_random_in_stage1_SI1_SD42'

#res_path = './metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'

#res_path = './metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'
#res_path = './metricsPR18_ForwardOracle_real_in_SI1_SD42'
#res_path = './metricsPR18_ForwardOracle_random_in_SI1_SD42'

#res_path = './metricsPR18_Phase1_Relabeling_real_inPrueba_SI1_SD42'
#res_path = './metricsPR18_Relab_real_in_stage1_SI1_SD42'

#res_path = './metricsWR28_Phase1_Relabeling_random_in_SI1_SD42'
#res_path = 'WR28_2Experiments/metricsWR28_Phase1_Relabeling_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase3_SSL_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase3_SSL_random_in_SI1_SD42'
#res_path = './metricsWR28_Phase3_SSL_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase3_SSL_MT_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase5_SSL2_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase7_SSL3_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase9_SSL4_real_in_SI1_SD42'
#res_path = './metricsWR28_Phase5_SSL2_random_in_SI1_SD42'

#noise_models_WR28_Phase3_SSL_real_in_SI1_SD42

#res_path = './metricsPR18_DynBootSoft64_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase3_SSL64_real_in_SI1_SD42'
#res_path = 'WR28_2Experiments/metricsWR28_DynBootSoft_real_in_SI1_SD42'
#res_path = './metricsWR28_DynBootSoft64_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase1_Relabeling_real_in2_SI1_SD42'
#res_path = './metricsPR18_Phase3_SSL_real_in2_SI1_SD42'
#res_path = './metricsWR28_4_Phase3_SSL_real_in_SI1_SD42'
#res_path = './metricsWR28_4_DynBootSoft_real_in_SI1_SD42'
#res_path = './metricsPR18_DynBootSoft_real_in2_SI1_SD42'

#res_path = './metricsPR18_DynBootSoft_real_in_SI1_SD42'
#res_path = './metricsPR18_DynBootSoftSSL_real_in105_SI1_SD42'
#res_path = './metricsPR18_DynBootSoftSSL_real_in95_SI1_SD42'
#res_path = 'PR18_Experiments/metricsPR18_DynBootSoftSSL_real_in_SI1_SD42'
#res_path = './metricsPR18_DynBootSoftSSLreg_real_in_SI1_SD42'
#res_path = './metricsPR18_Phase3_SSL_real_in_SI1_SD42'
#res_path = 'PR18_Experiments/metricsPR18_Phase3_SSL_real_in_SI1_SD42'
#res_path = 'metricsPR18_Phase3_SSL_real_in_SI1_SD42'
#res_path = 'metricsPR18_DynBootSoft_real_out_SI1_SD42'
#res_path = 'metricsPR18_Mixup_real_out_SI1_SD42'
#res_path = 'metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'

#res_path = 'metricsPR18_Mixup_real_in_SI1_SD42'
#res_path = 'metricsPR18_CE_random_in_CleanDet_SI1_SD42'
#res_path = 'metricsPR18_CE_real_in_CleanDet_SI1_SD42'
#res_path = 'metricsPR18_CE_random_out_CleanDet_SI1_SD42'
#res_path = 'metricsPR18_CE_real_out_CleanDet_SI1_SD42'


#res_path = 'metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'
#res_path = 'metricsPR18_Phase3_SSL_real_in_SI1_SD42'
#res_path = 'metricsPR18_Phase5_SSL2_real_in_SI1_SD42'

# res_path = 'metricsPR18_Phase1_Relabeling_random_in_SI1_SD42'
# res_path = 'metricsPR18_Phase3_SSL_random_in_SI1_SD42'
# res_path = 'metricsPR18_Phase5_SSL2_random_in_SI1_SD42'

# res_path = 'metricsPR18_Phase5_SSL2_random_in_SI1_SD42'
# res_path = 'metricsPR18_Phase7_SSL3_random_in_SI1_SD42'


# res_path = 'metricsPR18_Phase1_Relabeling_real_in_SI1_SD42'
# res_path = 'metricsPR18_Phase3_SSL_real_in_SI1_SD42'
# res_path = 'metricsPR18_Phase5_SSL2_real_in_SI1_SD42'

# res_path = 'metricsPR18_CrossEntropy_random_in_SI1_SD42'
# res_path = 'metricsPR18_DMI_real_in_SI1_SD42'
res_path = 'metricsPR18_DMI_random_in3_SI1_SD42'



# res_path = 'pencilServer/metricsPR18_PENCIL_real_in_SI1_SD42'
# res_path = 'metricsPR18_PENCIL_random_in_SI1_SD42'
# res_path = 'pencilServer/metricsPR18_PENCIL_random_in_SI1_SD42'
# res_path = 'pencilServer/metricsPR18_PENCIL_random_in_lr01_SI1_SD42'

# res_path = 'lccnServer/metricsPR18_LCCN_real_in_SI1_SD42'




# res_path = 'metricsPR18_GCE_real_in_NOprune_SI1_SD42'
# res_path = 'metricsPR18_LCCN_random_in_SI1_SD42'



noise_levels = np.array([0.6])
# noise_levels = np.array([0.5])
#noise_levels = np.array([0.2, 0.4, 0.6])

numNoise = noise_levels.size


loss_clean_vec = np.zeros((1,numNoise))
loss_noisy_vec = np.zeros((1,numNoise))

nRows = 1
nCols = 1
#figCN, axesCN = plt.subplots(nrows=4, ncols=2)
fig1 = plt.figure(1)
fig2 = plt.figure(2)

w_ewa = [0.01]
for ewa_i in w_ewa:
    for i in range(numNoise):

        res_noise_path = os.path.join(res_path,str(noise_levels[i]))
        ##Accuracy
        acc_train = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_accuracy_per_epoch_train.npy')
        acc_val = np.load(res_noise_path + '/' + str(noise_levels[i]) +'_accuracy_per_epoch_val.npy')

        #Loss per epoch
        loss_train = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_LOSS_epoch_train.npy')
        loss_val = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_LOSS_epoch_val.npy')

        numEpochs = len(acc_train)
        epochs = range(numEpochs)

        if i==0:
            loss_train_vec = np.zeros((numNoise, numEpochs))
            loss_val_vec = np.zeros((numNoise, numEpochs))

            acc_train_vec = np.zeros((numNoise, numEpochs))
            acc_val_vec = np.zeros((numNoise, numEpochs))

            # loss_samples_train_clean = np.zeros((epochs, numNoise))
            # loss_samples_train_noisy = np.zeros((epochs, numNoise))

        acc_train_vec[numNoise-i-1,:] = acc_train
        acc_val_vec[numNoise-i-1, :] = acc_val

        loss_train_vec[numNoise-i-1, :] = loss_train
        loss_val_vec[numNoise-i-1, :] = loss_val

        #Load clean and noisy samples

        loss_tr = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_losses_per_epoch_train.npy')
        loss_tr_t = np.transpose(loss_tr)
        noisy_labels = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_diff_labels.npy')
        labels = np.array(range(loss_tr.shape[1]))
        clean_labels = np.setdiff1d(labels, noisy_labels)

        #BMM_prob045_noisy = BMM_prob045[noisy_labels]
        #BMM_prob045_clean = BMM_prob045[clean_labels]

        # BMM_prob060_noisy = BMM_prob060[noisy_labels]
        # BMM_prob060_clean = BMM_prob060[clean_labels]

        # plt.figure(101)
        # num_bins = 100
        # #plt.hist(BMM_prob045_clean, num_bins, label='clean')
        # #plt.hist(BMM_prob045_noisy, num_bins, label='noisy')
        # plt.hist(BMM_prob060_clean, num_bins, label='clean')
        # plt.hist(BMM_prob060_noisy, num_bins, label='noisy')
        # plt.legend()
        # plt.show()


        # SoftMax_probs = np.load(res_noise_path + '/' + str(noise_levels[i]) + '.0' + '_probs_per_epoch_train.npy')
        # z = np.argmax(SoftMax_probs, 2)

        # plt.figure(51)
        #
        # plt.plot(epochs, acc_val, label='Validation, max acc: ' + str(np.max(acc_val)))
        # plt.plot(epochs, acc_train, label='Train, max acc: ' + str(np.max(acc_train)))
        # plt.ylabel('Acc test')
        # plt.xlabel('Epoch')
        # plt.legend(loc='lower right')
        # plt.show()

        noisy_labels = noisy_labels.astype(int)
        avg_clean = loss_tr_t[clean_labels].mean(axis=0)
        avg_noisy = loss_tr_t[noisy_labels].mean(axis=0)
        std_clean = loss_tr_t[clean_labels].std(axis=0)
        std_noisy = loss_tr_t[noisy_labels].std(axis=0)

        quart25_clean = np.quantile(loss_tr_t[clean_labels], 0.25, axis=0)
        quart75_clean = np.quantile(loss_tr_t[clean_labels], 0.75, axis=0)
        median_clean = np.quantile(loss_tr_t[clean_labels], 0.5, axis=0)

        if noise_levels[i]!=0:
            quart25_noisy = np.quantile(loss_tr_t[noisy_labels], 0.25, axis=0)
            quart75_noisy = np.quantile(loss_tr_t[noisy_labels], 0.75, axis=0)
            median_noisy = np.quantile(loss_tr_t[noisy_labels], 0.5, axis=0)


        #x = np.linspace(0, len(avg_right), len(avg_right))
        x = np.linspace(0, len(avg_clean),len(avg_clean))

        ax = fig1.add_subplot(str(nRows)+str(nCols)+str(i+1))
        ax.set_title('Noise level: ' + str(noise_levels[i]), y=1.08)
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Loss')
        # ax.plot(x, avg_clean, 'b-', label='Clean')
        # ax.fill_between(x, avg_clean - std_clean, avg_clean + std_clean, alpha=0.2, color='b')
        ax.plot(x, median_clean, 'b-', label='Clean')
        ax.fill_between(x, quart25_clean, quart75_clean, alpha=0.2, color='b')

        #plt.fill_between(x, avg_right - std_right * 2, avg_right + std_right * 2, alpha=0.2, color='b')
        if noise_levels[i] != 0:
            # ax.plot(x, avg_noisy, 'r-', label='Noisy')
            # ax.fill_between(x, avg_noisy - std_noisy, avg_noisy + std_noisy, alpha=0.2, color='r')
            ax.plot(x, median_noisy, 'r-', label='Noisy')
            ax.fill_between(x, quart25_noisy, quart75_noisy, alpha=0.2, color='r')
            #plt.fill_between(x, avg_wrong - std_wrong * 2, avg_wrong + std_wrong * 2, alpha=0.2, color='r')
            ax.legend(loc='upper right')

        lossEpochX = loss_tr_t[:,-1] #Epoch 40
        lossEpochX_clean = lossEpochX[clean_labels]
        lossEpochX_noisy = lossEpochX[noisy_labels]

        ax2 = fig2.add_subplot(str(nRows) + str(nCols) + str(i + 1))
        bins = np.linspace(0, 7, 100)

        ax2.hist(lossEpochX_clean, 50, alpha=0.5, label='Clean')
        ax2.hist(lossEpochX_noisy,50, alpha=0.5, label='Noisy')
        #ax2.hist(np.concatenate((lossEpochX_clean,lossEpochX_noisy)), 50, alpha=0.5, label='All')
        ax2.legend(loc='upper right')


        plt.figure(51)
        # plt.plot(epochs, acc_val, label = 'Final temp: ' + str(ewa_i))
        # plt.plot(epochs, acc_val, label = 'Noise level ' + str(noise_levels[i]))
        plt.plot(epochs, acc_val, label='Validation, max acc: ' + str(np.max(acc_val)))
        plt.plot(epochs, acc_train, label='Train, max acc: ' + str(np.max(acc_train)))
        plt.ylabel('Acc test')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.figure(52)
        # plt.plot(epochs, acc_val, label = 'Final temp: ' + str(ewa_i))
        # plt.plot(epochs, acc_val, label = 'Noise level ' + str(noise_levels[i]))
        plt.plot(epochs, acc_train, label='Max acc: ' + str(np.max(acc_train)))
        plt.ylabel('Acc train')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')

        plt.figure(53)
        # plt.plot(epochs, acc_val, label = 'Final temp: ' + str(ewa_i))
        # plt.plot(epochs, acc_val, label = 'Noise level ' + str(noise_levels[i]))
        plt.plot(epochs, loss_train, label='Min loss: ' + str(np.min(loss_train)))
        plt.ylabel('Loss train')
        plt.xlabel('Epoch')
        plt.legend(loc='lower right')



    # plt.figure(52)
    # plt.plot(epochs, loss_val, label='Noise level ' + str(noise_levels[i]))
    # plt.ylabel('Training loss')
    # plt.xlabel('Epoch')
    # plt.legend(loc='upper right')



# plt.show()

noise_type = "random_in"
noise_level = 0.8



bmm_noisy_probs2 = np.load('BMM_probs/BMM_probs_CIFAR-10_R_' + noise_type + '_noise_' + str(noise_level) + '_1_42.npy')
# bmm_noisy_probs2 = np.load('BMM_probs/BMM_probs_CIFAR-10_ssl_' + noise_type + '_noise_' + str(noise_level) + '_1_42.npy')
# bmm_noisy_probs2 = np.load('BMM_probs/BMM_probs_CIFAR-10_ssl2_' + noise_type + '_noise_' + str(noise_level) + '_1_42.npy')
#
# all_labels = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_all_labels.npy')
#
#
noisy_idx = np.load(res_noise_path + '/' + str(noise_levels[i]) + '_diff_labels.npy')
clean_idx = np.ones((len(bmm_noisy_probs2))).astype(bool)
clean_idx[noisy_idx] = False
#






# #### Plot BMM probs
clean_idx = np.ones((len(bmm_noisy_probs2))).astype(bool)
clean_idx[noisy_idx] = False
noisy_probs = bmm_noisy_probs2[noisy_idx]
clean_probs = bmm_noisy_probs2[clean_idx]
plt.figure(11)
plt.hist(clean_probs, 50, color='b', alpha = 0.5)
plt.hist(noisy_probs, 50, color='r', alpha = 0.5)

plt.figure(12)
# plt.hist(clean_probs, 50, color='b')
plt.hist(noisy_probs[noisy_probs<0.05], 200, color='r')
#
#
noisy05 = sum(bmm_noisy_probs2[noisy_idx]<0.5)
noisy_in = noisy05/len(bmm_noisy_probs2[noisy_idx])
print(noisy_in*100)
# embed()



plt.show()

print("Percentage of noise:", 100 * (sum(noisy_labels) / float(len(noisy_labels))))





#Plot loss
# fig, axes = plt.subplots(nrows=2, ncols=1)
# for k, ax in enumerate(axes.flat):
#     if k == 0:
#         im = ax.imshow(loss_train_vec)
#         ax.set_title('Train loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Noise')
#     elif k == 1:
#         im = ax.imshow(loss_val_vec)
#         ax.set_title('Validation loss')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Noise')
#
# plt.sca(axes[0])
# plt.yticks(range(numNoise))
# #plt.yticks(range(numNoise), ('60', '50', '40', '30', '20', '10', '0'))
# plt.yticks(range(numNoise), np.flip(noise_levels,0))
#
#
# plt.sca(axes[1])
# plt.yticks(range(numNoise))
# #plt.yticks(range(numNoise), ('60', '50', '40', '30', '20', '10', '0'))
# plt.yticks(range(numNoise), np.flip(noise_levels,0))
#
# fig.subplots_adjust(right=0.8)
# cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
# fig.colorbar(im, cax=cbar_ax)
#
# plt.show()
#
# #Plot accuracy
# fig1, axes1 = plt.subplots(nrows=2, ncols=1)
# for k, ax in enumerate(axes1.flat):
#     if k == 0:
#         im2 = ax.imshow(acc_train_vec)
#         ax.set_title('Train accuracy')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Noise')
#     elif k == 1:
#         im2 = ax.imshow(acc_val_vec)
#         ax.set_title('Validation accuracy')
#         ax.set_xlabel('Epoch')
#         ax.set_ylabel('Noise')
#
# plt.sca(axes1[0])
# plt.yticks(range(numNoise))
# plt.yticks(range(numNoise), ('60', '50', '40', '30', '20', '10', '0'))
#
# plt.sca(axes1[1])
# plt.yticks(range(numNoise))
# plt.yticks(range(numNoise), ('60', '50', '40', '30', '20', '10', '0'))
#
# fig1.subplots_adjust(right=0.8)
# cbar_ax = fig1.add_axes([0.85, 0.15, 0.05, 0.7])
# fig1.colorbar(im2, cax=cbar_ax)
#
# plt.show()



    #
    #
    # plt.figure()
    # plt.plot(epochs,acc_train, label = 'Train')
    # plt.plot(epochs,acc_val, label = 'Validation')
    # plt.ylabel('Acc')
    # plt.xlabel('Epoch')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    # # Loss per epoch
    # plt.figure()
    # plt.plot(epochs,loss_train_accum, label = 'Train')
    # plt.plot(epochs,loss_val_accum, label = 'Validation')
    # plt.ylabel('Loss')
    # plt.xlabel('Epoch')
    # plt.legend(loc='upper right')
    # plt.show()
    #
    #
    # # Loss per epoch tracked training
