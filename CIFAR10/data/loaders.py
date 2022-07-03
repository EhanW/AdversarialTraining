from torchvision import transforms as T
from torchvision.datasets import CIFAR10
from torch.utils.data import DataLoader


def get_train_loader(batch_size, num_workers=4):
	train_transforms = T.Compose([T.RandomCrop(32, 4), T.RandomHorizontalFlip(), T.ToTensor()])
	train_full = CIFAR10('datasets', train=True, transform=train_transforms, download=True)
	train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
	return train_loader


def get_test_loader(batch_size, num_workers=4):
	test_full = CIFAR10('datasets', train=False, transform=T.ToTensor(), download=True)
	test_loader = DataLoader(test_full, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
	return test_loader


