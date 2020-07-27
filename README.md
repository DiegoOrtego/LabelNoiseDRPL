# Code for paper <a href="https://arxiv.org/abs/1912.08741" target="_blank">"Towards Robust Learning with Different Label Noise Distributions"</a> 

Our approach robustly learns CNNs for Image Classification in the presence of label noise. 

We provide examples to run our method in:

- CIFAR-10/100: "cifar10" and "cifar100" folders contain code used to simulate label noise with 2 different label noise distributions: uniform (random_in noise type) and non-uniform noise (real_in noise type). We provide example scripts to run our approach for both noise types: "RunScripts_cifar10.sh" and "RunScripts_cifar100.sh". Both datasets are downloaded automatically when setting "--download True". The dataset have to be placed in cifar10/data/ folder (should be done automatically).
- ImageNet32/64: "ImageNet32_64" folder contains code used to simulate label noise with 4 different label noise distributions: uniform and non-uniform for both in-distribution and ouy-of-distribution noise. We provide an example script "RunScripts_Im32.sh" to run the method in ImageNet32. To run it in ImageNet64, change the dataset argument from "ImageNet32" to "ImageNet64". ImageNet32 and ImageNet64 requiere download from http://www.image-net.org. After download they have to be placed in ImageNet32_64/data/ folder. To facilitate selecting the same 100 in-distribution classes used in our experiments we provide txt files with the lists of in-distribution and out-of-distribution classes and image indexes, which could be replaced by the ones selected randomly the dataset class.

Main requirements:

- Python 3.5.2
- Pytorch 0.4.1 (torchvision 0.2.1)
- Numpy 1.15.3
- scikit-learn 0.21.3
- cuda 9.0

(also tested in Pytorch 1.5)


