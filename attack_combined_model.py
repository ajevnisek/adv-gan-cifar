import os
import torch
import matplotlib.pyplot as plt
import torch.nn. functional as F

import torchvision.datasets
import torchvision.transforms as transforms

from enum import Enum, auto

from torch import nn
from torch.utils.data import DataLoader

import models
from common import Permutation, pgd, pgd_linf, pgd_linf_targ, fgsm
from common import plot_images_cifar as plot_images
from common import plot_predictions_cifar as plot_predictions
from models import CIFAR_target_net, CombinedModel, ClassificationMode


use_cuda = True
image_nc = 3
batch_size = 128
epsilon = 8.0/256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
attack = pgd_linf_targ


# test adversarial examples in CIFAR10 training dataset
cifar_dataset = torchvision.datasets.CIFAR10('./dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_dataloader = DataLoader(cifar_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1)
# test adversarial examples in CIFAR10 testing dataset
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
for i in range(1, 10):
    new_labels = (labels + i) % 10
    permutations.append(Permutation(f'plus_{i}_permutation', new_labels,
                                [str(i.item()) for i in labels]))


# load the pretrained model
pretrained_model = "./CIFAR_target_model.pth"
target_model = CIFAR_target_net().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()
generator_models = nn.ModuleList([])

for perm_index, permutation in enumerate(permutations):
    # load the generator of adversarial examples
    pretrained_generator_path = os.path.join(permutation.name,
                                             'models',
                                             f"netG_epoch_{60}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    generator_models.append(pretrained_G)


combined_model = CombinedModel(
    target_model, permutations, generator_models,
    classification_mode=ClassificationMode.AVERAGE_LOGITS)


# Original test set performance:
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    pred_lab = torch.argmax(target_model(test_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of pristine imgs for basic model: {accuracy:.2f}[%]')


# Original test set performance on combined model:
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    pred_lab = torch.argmax(combined_model(test_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of pristine imgs for combined model: {accuracy:.2f}[%]')

# test the target model resilience to adversarial attacks:
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    target_labels = 2 * torch.ones_like(test_label).to(device)
    target_labels[test_label == 2] = 3
    test_adv_img = test_img + attack(target_model, test_img, test_label,
                                     epsilon=0.3, alpha=1e-2,
                                     num_iter=40, y_targ=target_labels)
    pred_lab = torch.argmax(target_model(test_adv_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of attacked imgs for combined model where  target:'
      f' {accuracy:.2f}['
      f'%]')

# test the combined model resilience to adversarial attacks:
num_correct = 0
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    target_labels = 2 * torch.ones_like(test_label).to(device)
    target_labels[test_label == 2] = 3
    test_adv_img = test_img + attack(combined_model, test_img, test_label,
                                     epsilon=0.3, alpha=1e-2,
                                     num_iter=40, y_targ=target_labels)
    pred_lab = torch.argmax(combined_model(test_adv_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of attacked imgs for combined model: {accuracy:.2f}[%]')

