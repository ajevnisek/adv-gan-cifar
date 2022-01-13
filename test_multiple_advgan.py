import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader

import models
from common import Permutation
from models import CIFAR_target_net


use_cuda = True
image_nc = 3
batch_size = 128
nof_branches = 2
epsilon = 8.0 / 256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

from common import plot_images_cifar as plot_images
from common import plot_predictions_cifar as plot_predictions

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
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    pred_lab = torch.argmax(target_model(test_img), 1)
    num_correct += torch.sum(pred_lab == test_label, 0)

print('num_correct: ', num_correct.item())
print('num_correct: ', num_correct.item())
accuracy = num_correct.item() / len(cifar_dataset_test) * 100.0
print(f'accuracy of pristine imgs in testing set: {accuracy:.2f}[%]')


permutations = []
labels = torch.tensor(list(range(10))).to(device)
for i in range(1, nof_branches):
    new_labels = (labels + i) % 10
    permutations.append(Permutation(f'plus_{i}_permutation', new_labels,
                                [str(i.item()) for i in labels]))

for permutation in permutations:
    # load the generator of adversarial examples
    # pretrained_generator_path = './models/netG_epoch_20.pth'
    pretrained_generator_path = os.path.join(permutation.name,
                                             'models',
                                             f"netG_epoch_{20}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()

    # test perturbation adversarial attack effectiveness on train set
    num_correct = 0
    for i, data in enumerate(train_dataloader, 0):
        test_img, test_label = data
        test_img, test_label = test_img.to(device), test_label.to(device)
        perturbation = pretrained_G(test_img)
        perturbation = torch.clamp(perturbation, -epsilon, epsilon)
        adv_img = perturbation + test_img
        adv_img = torch.clamp(adv_img, 0, 1)
        pred_lab = torch.argmax(target_model(adv_img),1)
        num_correct += torch.sum(pred_lab==test_label,0)

    print('CIFAR training dataset:')
    print('num_correct: ', num_correct.item())
    accuracy = num_correct.item()/len(cifar_dataset)
    print(f'accuracy of adv imgs in training set: {accuracy:.2f}[%]\n')

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
        num_correct += torch.sum(pred_lab==test_label,0)

    print('num_correct: ', num_correct.item())
    accuracy = num_correct.item()/len(cifar_dataset_test) * 100.0
    print(f'accuracy of adv imgs in testing set: {accuracy:.2f}[%]\n')


    data = next(iter(test_dataloader))
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    perturbation = pretrained_G(test_img)
    perturbation = torch.clamp(perturbation, -epsilon, epsilon)
    adv_img = perturbation + test_img
    adv_img = torch.clamp(adv_img, 0, 1)
    plt.clf()
    # plot pristine images
    plot_images(test_img.detach(), test_label, 3, 6, suptitle='Test images')
    base_dir = os.path.join(permutation.name, "creating_auto_advs")
    os.makedirs(base_dir, exist_ok=True)
    plt.savefig(os.path.join(base_dir,  'test_images.png'))
    plt.savefig(os.path.join(base_dir,  'test_images.pdf'))
    # plot perturbations
    from common import plot_images_mnist as plot_one_dim_predictions
    plot_one_dim_predictions(perturbation.detach(), test_label, 3, 6,
                             suptitle='Perturbations')
    plt.savefig(os.path.join(base_dir,  'perturbations.png'))
    plt.savefig(os.path.join(base_dir,  'perturbations.pdf'))
    # plot pristine images with the classifier predictions
    predictions = target_model(test_img)
    plot_predictions(test_img.detach(), test_label, predictions, 3, 6,
                     suptitle='Classified Test images')
    plt.savefig(os.path.join(base_dir,  'classified_test_images.png'))
    plt.savefig(os.path.join(base_dir,  'classified_test_images.pdf'))
    # plot adversarial images with the classifier predictions
    predictions = target_model(adv_img)
    plot_predictions(adv_img.detach(), test_label, predictions, 3, 6,
                     suptitle='Classified Adversarial images')
    plt.savefig(os.path.join(base_dir,  'classified_adversarial_images.png'))
    plt.savefig(os.path.join(base_dir,  'classified_adversarial_images.pdf'))


