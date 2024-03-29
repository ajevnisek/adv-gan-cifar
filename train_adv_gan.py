import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from adv_gan import AdvGAN_Attack
from models import CIFAR_target_net
from simple_dla_classifier import SimpleDLA

use_cuda = True
image_nc = 3
epochs = 120
batch_size = 128
BOX_MIN = 0
BOX_MAX = 1
# epsilon = 0.3
epsilon = 8.0 / 256

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# pretrained_model = "./CIFAR_target_model.pth"
# targeted_model = CIFAR_target_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
targeted_model = SimpleDLA().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# MNIST train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.CIFAR10('./dataset',
                                             train=True,
                                             transform=transforms.ToTensor(),
                                             download=True)
dataloader = DataLoader(mnist_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=1)
advGAN = AdvGAN_Attack(device,
                       targeted_model,
                       model_num_labels,
                       image_nc,
                       BOX_MIN,
                       BOX_MAX,
                       epsilon)

advGAN.train(dataloader, epochs)
