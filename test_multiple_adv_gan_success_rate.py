import os
import torch
import numpy as np
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch import nn
from torch.nn.functional import one_hot
from torch.utils.data import DataLoader

import models
from common import Permutation, pgd, pgd_linf, pgd_linf_targ, fgsm
from common import plot_images_cifar as plot_images
from common import plot_predictions_cifar as plot_predictions
from models import CIFAR_target_net


use_cuda = True
image_nc = 3
batch_size = 32
generator_epochs = 20
epsilon = 256.0 * 0.5
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
attack = pgd_linf_targ
root_path = 'visualization'
nof_branches = 2

# test adversarial examples in CIFAR training dataset
cifar_dataset = torchvision.datasets.CIFAR10('./dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_dataloader = DataLoader(cifar_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1)
# test adversarial examples in CIFAR testing dataset
cifar_dataset_test = torchvision.datasets.CIFAR10('./dataset',
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
test_dataloader = DataLoader(cifar_dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)



permutations = []
labels = torch.tensor(list(range(10))).to(device)
for i in range(1, nof_branches):
    new_labels = (labels + i) % 10
    permutations.append(Permutation(f'plus_{i}_permutation', new_labels,
                                [str(l.item()) for l in labels]))


from models import CombinedModel, ClassificationMode

# load the pretrained model
# pretrained_model = "./MNIST_target_model.pth"
# classifier = MNIST_target_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
from simple_dla_classifier import SimpleDLA
classifier = SimpleDLA().to(device)
classifier.load_state_dict(torch.load(pretrained_model))
classifier.eval()
generator_models = nn.ModuleList([])

for perm_index, permutation in enumerate(permutations):
    # load the generator of adversarial examples
    pretrained_generator_path = os.path.join(
        permutation.name, 'models', f"netG_epoch_{generator_epochs}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    generator_models.append(pretrained_G)

asrs_list = []
for netG, permutation in zip(generator_models, permutations):
    nof_correct = 0
    nof_samples = 0
    for data in test_dataloader:
        images, labels = data
        images, labels = images.to(device), labels.to(device)
        perturbation = netG(images)
        # add a clipping trick
        adv_images = torch.clamp(perturbation, -epsilon, epsilon) + images
        adv_images = torch.clamp(adv_images, 0, 1)
        logits_model = classifier(adv_images)
        predictions = torch.argmax(logits_model, axis=1)
        backward_permuted_preds = permutation.backward_permutation_function(
            predictions)
        nof_correct += (labels == backward_permuted_preds).sum()
        nof_samples += predictions.shape[0]
    accuracy = nof_correct / nof_samples * 100.0
    asrs_list.append(accuracy)

plt.stem(range(1, len(asrs_list) + 1), asrs_list)
plt.grid(True)
plt.xlabel('permutation #')
plt.ylabel('ASR[%]')
plt.title('Attack Success Rate vs permutation index')
plt.show()