import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

import models
from common import Permutation, pgd, pgd_linf, pgd_linf_targ, fgsm
from common import plot_images_cifar as plot_images
from common import plot_predictions_cifar as plot_predictions
from models import CIFAR_target_net


use_cuda = True
image_nc = 3
batch_size = 128
epsilon = 8.0 / 256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
attack = pgd_linf_targ
nof_branches = 10
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

# load the pretrained model
# pretrained_model = "./CIFAR_target_model.pth"
# target_model = CIFAR_target_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
from simple_dla_classifier import SimpleDLA
target_model = SimpleDLA().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# Original test set performance:
num_correct = 0
test_set_labels = []
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    pred_lab = torch.argmax(target_model(test_img), 1)
    test_set_labels.append(pred_lab.cpu())
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of pristine imgs in testing set: {accuracy:.2f}[%]')
test_set_labels = torch.cat(test_set_labels)


permutations = []
labels = torch.tensor(list(range(10))).to(device)
for i in range(1, nof_branches):
    new_labels = (labels + i) % 10
    permutations.append(Permutation(f'plus_{i}_permutation', new_labels,
                                [str(i.item()) for i in labels]))

test_sample_to_adv_prediction = torch.zeros((len(cifar_dataset_test),
                                             len(permutations)))

for perm_index, permutation in enumerate(permutations):
    # load the generator of adversarial examples
    # pretrained_generator_path = './models/netG_epoch_20.pth'
    pretrained_generator_path = os.path.join(permutation.name,
                                             'models',
                                             f"netG_epoch_{60}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test perturbation adversarial attack effectiveness on test set:
    num_correct = 0
    for i, data in enumerate(test_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        test_sample_to_adv_prediction[i*batch_size:(i+1)*batch_size,
        perm_index] = pred_lab


labels_shift = torch.tensor(list(range(1, nof_branches)))
new_labels = ((test_sample_to_adv_prediction - labels_shift) % 10).median(
    axis=1)[0]
accuracy = (torch.tensor(cifar_dataset_test.targets) == new_labels).sum() / len(
    cifar_dataset_test.targets) * 100.0
print(f"Accuracy best solely on adversarial permutation {accuracy:.2f} [%]")

agreement = (test_set_labels == new_labels).sum() / len(new_labels)
print(f"Agreement rate between AdvGAN to just classification: "
      f"{agreement * 100:.2f} [%]")

generator_models = nn.ModuleList([])

for perm_index, permutation in enumerate(permutations):
    # load the generator of adversarial examples
    # pretrained_generator_path = './models/netG_epoch_20.pth'
    pretrained_generator_path = os.path.join(permutation.name,
                                             'models',
                                             f"netG_epoch_{60}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    generator_models.append(pretrained_G)

# test perturbation adversarial attack effectiveness on test set:
adv_test_sample_to_adv_prediction = torch.zeros((len(cifar_dataset_test),
                                                 len(permutations)))
adv_test_sample_to_prediction = []
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    test_adv_img = test_img + attack(target_model, test_img, test_label,
                                     epsilon=0.3, alpha=1e-2,
                                     num_iter=40, y_targ=2)
    # inference through C(x)
    with torch.no_grad():
        pred_lab = torch.argmax(target_model(test_adv_img), 1)
        adv_test_sample_to_prediction.append(pred_lab)
    for perm_index, pretrained_G in enumerate(generator_models):
        with torch.no_grad():
            perturbation = pretrained_G(test_adv_img)
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_img = perturbation + test_adv_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        adv_test_sample_to_adv_prediction[i*batch_size:(i+1)*batch_size,
        perm_index] = pred_lab

# evaluate AdvGAN outputs on PGD attacked images:
labels_shift = torch.tensor(list(range(1, nof_branches)))
adv_test_adv_prediction_new_labels = (
        (adv_test_sample_to_adv_prediction - labels_shift) % 10).median(
    axis=1)[0]
accuracy = (torch.tensor(cifar_dataset_test.targets) ==
            adv_test_adv_prediction_new_labels).sum() / len(
    cifar_dataset_test.targets) * 100.0
print(f"Accuracy best solely on adversarial permutation of PGD attacked images"
      f" {accuracy:.2f} [%]")

# test perturbation adversarial attack effectiveness on test set:
adv_test_sample_to_prediction = torch.cat(adv_test_sample_to_prediction).cpu()
accuracy = (torch.tensor(cifar_dataset_test.targets) ==
            adv_test_sample_to_prediction).sum() / len(
    cifar_dataset_test.targets) * 100.0
print(f"Accuracy when images are attacked with targeted PGD: {accuracy:.2f} ["
      f"%]")

agreement = (adv_test_adv_prediction_new_labels ==
             adv_test_sample_to_prediction).sum() / len(adv_test_adv_prediction_new_labels)
print(f"Agreement rate between AdvGAN to just classification for attacked "
      f"images: {agreement * 100:.2f} [%]")
