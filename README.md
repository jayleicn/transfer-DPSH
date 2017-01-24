# transfer-DPSH

- Setup MatConvNet
```
run ~/tools/ext_matconvnet/matlab/vl_setupnn.m
```
- Download the Pretrained CNN model VGG-F from the website
```
wget http://www.vlfeat.org/matconvnet/models/imagenet-vgg-f.mat
```
- Run the experiment
```
[B_dataset, B_test, map] = DPSH(32, 'cifar10', ratio); % ratio in [0.0 ~ 1.0] 
```
