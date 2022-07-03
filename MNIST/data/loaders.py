from torchvision import transforms as T
from torchvision.datasets import MNIST
from torch.utils.data import DataLoader


def get_train_loader(batch_size, resize=28, num_workers=4):
    train_full = MNIST('datasets', train=True, transform=T.Compose([T.Resize(resize), T.ToTensor()]), download=True)
    train_loader = DataLoader(train_full, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    return train_loader


def get_test_loader(batch_size, resize=28, num_workers=4):
    test_full = MNIST('datasets', train=False, transform=T.Compose([T.Resize(resize), T.ToTensor()]), download=True)
    test_loader = DataLoader(test_full, batch_size=batch_size, num_workers=num_workers, pin_memory=True)
    return test_loader
