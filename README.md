# Adversarial Training---Pytorch

## Networks
ResNet, PreAct-ResNet, Resnet for CIFAR.
[He K, Zhang X, Ren S, et al.](https://arxiv.org/abs/1512.03385)

Wide-ResNet.
[Zagoruyko S, Komodakis N.](https://arxiv.org/abs/1605.07146)

VGG.
[Simonyan K, Zisserman A.](https://arxiv.org/abs/1409.1556)

MobileNet.
[Howard A G, Zhu M, Chen B, et al.](https://arxiv.org/abs/1704.04861)
## 

## Adversarial Training
### Settings
Attack: PGD with epsilon 8/255, alpha 2/255, steps 10 and random start.

Optimizer: SGD with learning rate 0.1, momentume 0.9 and weight decay 5e-4.

Scheduler: MultiStepLR with milestones [100, 105] and gamma 0.1.
### Results
|Arch|clean accuracy| adv accuracy|
|----|----|----|
|ResNet18|84.55|52.77|
|ResNet20|74.38|45.43|
|ResNet32|77.21|48.24|
|ResNet44|78.21|49.54|
|ResNet56|79.62|49.54|

## Adversarial Distillation and Black-Box Attack
### Settings
Models: a pre-trained target model and a proxy model.

Knowledge distillation temperature: 5.

Loss function: the cross entropy loss between the proxy model logits and groud truth labels; 
the KL-Divergence loss between the distilled proxy model logits and the distilled target model logtis;
a hyper-parameter lambda to balance the above two losses.

Attack: PGD with epsilon 8/255, alpha 2/255, steps 10 and random start.

Optimizer: SGD with learning rate 0.1, momentume 0.9 and weight decay 5e-4.

Scheduler: MultiStepLR with milestones [100, 105] and gamma 0.1.

### Results
Collums: target models.

Rows: proxy models.

Report: (proxy model clean acc, proxy model adv acc, transfer adv acc from proxy model to target model).

|Arch|ResNet18| ResNet34|
|----|----|----|
|ResNet18||84.25, 52.69, 60.59|

