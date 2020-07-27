

############################ Example to run uniform noise

#Set an argument "--download True" in stage 1 if dataset is not downloaded

#Label Noise detection: stage 1
python3 train_ssl.py --epoch 100 --num_classes 100 --epoch_begin 40 --M 45 --M 80 --noise_ratio 0.80 --download True \
--network "PR18" --dataset "CIFAR-100" --method "SoftRelabeling" --noise_type "random_in_noise" \
--save_BMM_probs "True" --experiment_name Phase1_Relabeling_random_in --cuda_dev 0

# Label Noise detection: stage 2 (Semi-supervised learning - warm-up)
python3 train_ssl.py --epoch 10 --num_classes 100 --noise_ratio 0.80 --noise_type "random_in_noise" \
--network "PR18" --dataset "CIFAR-100" --bmm_th 0.1 --random_relab 0.0 --method "ssl" --ssl_warmup "True"  --alpha 1 \
--experiment_name Phase2_Wup_random_in --cuda_dev 0

 #Label Noise detection: stage 2 (Semi-supervised learning)
python3 train_ssl.py --epoch 175 --initial_epoch 10 --num_classes 100 --M 100 --M 150 --noise_ratio 0.80 \
--noise_type "random_in_noise" --network "PR18" --dataset "CIFAR-100" --method "ssl" --bmm_th 0.1 \
--random_relab 0.2 --ssl_warmup "False" --alpha 1  \
--experiment_name Phase3_SSL_random_in --save_BMM_probs 'True' --cuda_dev 0

# Final training (Semi-supervised learning - warm-up)
python3 train_ssl.py --epoch 10 --num_classes 100 --noise_ratio 0.80 --noise_type "random_in_noise" \
--network "PR18" --dataset "CIFAR-100" --method "ssl2" --bmm_th 0.5 --random_relab 0.0 \
--ssl_warmup "True"  --alpha 1 \
--experiment_name Phase4_Wup2_random_in --cuda_dev 0

# Final training (Semi-supervised learning)
python3 train_ssl.py --epoch 300 --initial_epoch 10 --num_classes 100 --M 150 --M 225 --noise_ratio 0.80 \
--noise_type "random_in_noise" --network "PR18" --dataset "CIFAR-100" --method "ssl2" --bmm_th 0.5 \
--random_relab 0.2 --ssl_warmup "False" --alpha 1  \
--experiment_name Phase5_SSL2_random_in --save_BMM_probs 'True' --cuda_dev 0


############################ Example to run non-uniform noise

#Label Noise detection: stage 1
python3 train_ssl.py --epoch 100 --num_classes 100 --epoch_begin 40 --M 45 --M 80 --noise_ratio 0.40 \
--network "PR18" --dataset "CIFAR-100" --method "SoftRelabeling" --noise_type "real_in_noise" \
--save_BMM_probs "True" --experiment_name Phase1_Relabeling_real_in --cuda_dev 0

# Label Noise detection: stage 2 (Semi-supervised learning - warm-up)
python3 train_ssl.py --epoch 10 --num_classes 100 --noise_ratio 0.40 --noise_type "real_in_noise" \
--network "PR18" --dataset "CIFAR-100" --bmm_th 0.1 --random_relab 0.0 --method "ssl" --ssl_warmup "True"  --alpha 1 \
--experiment_name Phase2_Wup_real_in --cuda_dev 0
#
 #Label Noise detection: stage 2 (Semi-supervised learning)
python3 train_ssl.py --epoch 175 --initial_epoch 10 --num_classes 100 --M 100 --M 150 --noise_ratio 0.40 \
--noise_type "real_in_noise" --network "PR18" --dataset "CIFAR-100" --method "ssl" --bmm_th 0.1 \
--random_relab 0.2 --ssl_warmup "False" --alpha 1  \
--experiment_name Phase3_SSL_real_in --save_BMM_probs 'True' --cuda_dev 0

# Final training (Semi-supervised learning - warm-up)
python3 train_ssl.py --epoch 10 --num_classes 100 --noise_ratio 0.40 --noise_type "real_in_noise" \
--network "PR18" --dataset "CIFAR-100" --method "ssl2" --bmm_th 0.5 --random_relab 0.0 \
--ssl_warmup "True"  --alpha 1 \
--experiment_name Phase4_Wup2_real_in --cuda_dev 0

# Final training (Semi-supervised learning)
python3 train_ssl.py --epoch 300 --initial_epoch 10 --num_classes 100 --M 150 --M 225 --noise_ratio 0.40 \
--noise_type "real_in_noise" --network "PR18" --dataset "CIFAR-100" --method "ssl2" --bmm_th 0.5 \
--random_relab 0.2 --ssl_warmup "False" --alpha 1  \
--experiment_name Phase5_SSL2_real_in --save_BMM_probs 'True' --cuda_dev 0




