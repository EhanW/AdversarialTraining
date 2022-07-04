# Adversarial Training---Pytorch

## Networks
ResNet, PreAct-ResNet, Resnet for CIFAR.
[He K, Zhang X, Ren S, et al.](https://arxiv.org/abs/1512.03385)

Wide-ResNet
[Zagoruyko S, Komodakis N.](https://arxiv.org/abs/1605.07146)

VGG
[Simonyan K, Zisserman A.](https://arxiv.org/abs/1409.1556)

MobileNet
[Howard A G, Zhu M, Chen B, et al.](https://arxiv.org/abs/1704.04861)
## 

## Adversarial Training Results
Attack: PGD with epsilon 8/255, alpha 2/255, steps 10 and random start.

Optimizer: SGD with learning rate 0.1, momentume 0.9 and weight decay 5e-4.

Scheduler: MultiStepLR with milestones [100, 105] and gamma 0.1.

|Arch|clean accuracy| adv accuracy|
|----|----|----|
|ResNet18|0|0|
