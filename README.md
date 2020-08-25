# Code for paper <a href="https://arxiv.org/abs/1912.08741" target="_blank">"Towards Robust Learning with Different Label Noise Distributions"</a> 

#### Abstract:

Noisy labels are an unavoidable consequence of labeling processes and detecting them is an important step towards preventing performance degradations in Convolutional Neural Networks. Discarding noisy labels avoids a harmful memorization, while the associated image content can still be exploited in a semi-supervised learning (SSL) setup. Clean samples are usually identified using the small loss trick, i.e. they exhibit a low loss. However, we show that different noise distributions make the application of this trick less straightforward and propose to continuously relabel all images to reveal a discriminative loss against multiple distributions. SSL is then applied twice, once to improve the clean-noisy detection and again for training the final model. We design an experimental setup based on ImageNet32/64 for better understanding the consequences of representation learning with differing label noise distributions and find that non-uniform out-of-distribution noise better resembles real-world noise and that in most cases intermediate features are not affected by label noise corruption. Experiments in CIFAR-10/100, ImageNet32/64 and WebVision (real-world noise) demonstrate that the proposed label noise Distribution Robust Pseudo-Labeling (DRPL) approach gives substantial improvements over recent state-of-the-art. 

#### Examples to run our method:

- CIFAR-10/100: "cifar10" and "cifar100" folders contain the code to run our method with 2 different label noise distributions: uniform (random_in noise type) and non-uniform noise (real_in noise type). We provide example scripts to run our approach for both noise types: "RunScripts_cifar10.sh" and "RunScripts_cifar100.sh". Both datasets are downloaded automatically when setting "--download True". The dataset have to be placed in cifar10/data/ folder (should be done automatically).
- ImageNet32/64: "ImageNet32_64" folder contains the code to run our method with 4 different label noise distributions: uniform and non-uniform for both in-distribution and ouy-of-distribution noise. We provide an example script "RunScripts_Im32.sh" to run the method in ImageNet32. To run it in ImageNet64, change the dataset argument from "ImageNet32" to "ImageNet64". ImageNet32 and ImageNet64 requiere download from http://www.image-net.org. After download they have to be placed in ImageNet32_64/data/ folder. To facilitate selecting the same 100 in-distribution classes used in our experiments we provide txt files with the lists of in-distribution and out-of-distribution classes and image indexes, which could be replaced by the ones selected randomly the dataset class.

#### Main requirements:

- Python 3.7.7
- Pytorch 1.5.1 (torchvision 0.6.1)
- Numpy 1.18.5
- scikit-learn 0.23.1
- cuda 9.2


#### Examples of noisy samlpes detected in WebVision

![couldn't find image](https://github.com/DiegoOrtego/LabelNoiseDRPL/blob/master/noisy_examples.png)


#### Test Accuracy


|Non-uniform noise|0%|10%|30%|40%|
|----|----|----|----|----|
|CIFAR-10|94.47|95.70|93.65|93.14|
|CIFAR-100|72.27 |72.40 |69.30 |65.86 |

|Uniform noise|0%|20%|40%|60%|80%|
|----|----|----|----|----|----|
|CIFAR-10|94.47|94.20|92.92|89.21|64.35|
|CIFAR-100|72.27 |71.25 |73.13 |68.71 |53.04|


#### Please consider citing the following paper if you find this work useful for your research.

```
 @article{2020_arXiv_DRPL,
  title = {Towards Robust Learning with Different Label Noise Distributions},
  authors = {Diego Ortego and Eric Arazo and Paul Albert and Noel E O'Connor and Kevin McGuinness},
  year={2020},
  journal={arXiv: 2007.11866},
 } 
```

Diego Ortego, Eric Arazo, Paul Albert, Noel E. O'Connor, Kevin McGuinness. "Towards Robust Learning with Different Label Noise Distributions", arXiv, 2020.
