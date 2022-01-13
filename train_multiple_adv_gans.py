import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from common import Permutation
from adv_gan import AdvGAN_Attack
from models import CIFAR_target_net

use_cuda = True
image_nc = 3
epochs = 20
batch_size = 256
BOX_MIN = 0
BOX_MAX = 1
epsilon = 8.0 / 256.0
nof_branches = 2

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# pretrained_model = "./CIFAR_target_model.pth"
# targeted_model = CIFAR_target_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
from simple_dla_classifier import SimpleDLA
targeted_model = SimpleDLA().to(device)
targeted_model.load_state_dict(torch.load(pretrained_model))
targeted_model.eval()
model_num_labels = 10

# CIFAR train dataset and dataloader declaration
mnist_dataset = torchvision.datasets.CIFAR10('./dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
dataloader = DataLoader(mnist_dataset,
                        batch_size=batch_size,
                        shuffle=True,
                        num_workers=4)
mnist_test_dataset = torchvision.datasets.CIFAR10('./dataset',
                                           train=False,
                                           transform=transforms.ToTensor(),
                                           download=True)
test_dataloader = DataLoader(mnist_test_dataset,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=4)


permutations = []
labels = torch.tensor(list(range(10))).to(device)
for i in range(1, nof_branches):
    new_labels = (labels + i) % 10
    permutations.append(Permutation(f'plus_{i}_permutation', new_labels,
                                    [str(i.item()) for i in labels]))
for permutation in permutations:
    target_path = f"{permutation.name}"
    os.makedirs(target_path, exist_ok=True)
    perm_func = permutation.permutation_function
    advGAN = AdvGAN_Attack(device,
                           targeted_model,
                           model_num_labels,
                           image_nc,
                           BOX_MIN,
                           BOX_MAX,
                           epsilon=epsilon,
                           permutation_function=perm_func,
                           target_path=target_path,
                           is_cifar10=True)

    advGAN.train(dataloader, epochs, test_dataloader)
