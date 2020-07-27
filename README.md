# Code for paper <a href="https://arxiv.org/abs/1912.08741" target="_blank">"Towards Robust Learning with Different Label Noise Distributions"</a> 

Please note that the code is not cleaned and it will be further revised before sharing it online. This is just an example of how to run our method in CIFAR-10, CIFAR-100, ImageNet32 and ImageNet64.
- CIFAR-10 and CIFAR-100 are downloaded automatically when setting "--download True". The dataset have to be placed in data folder (should be done automatically). We provide all code used to simulate label noise with 2 different distributions and provide example scripts to run our approach for each of the noise types.

- ImageNet32 and ImageNet64 requiere download from http://www.image-net.org. After download they have to be placed in data folder. We provide all code used to simulate label noise with 4 different distributions and provide example scripts to run our approach for each of the noise types. To facilitate selecting the same 100 in-distribution classes used in our experiments we provide txt files with the lists of in-distribution and out-of-distribution classes and image indexes. To run ImageNet64, use the script provided for ImageNet32 and change the dataset argument from "ImageNet32" to "ImageNet64". 

Requirements:

Python 3.5.2
Pytorch 0.4.1 (torchvision 0.2.1)
Numpy 1.15.3
scikit-learn 0.21.3
cuda 9.0




