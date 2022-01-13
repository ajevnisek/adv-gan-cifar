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
generator_epochs = 120
epsilon = 24.0 / 256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
attack = pgd_linf_targ
root_path = 'visualization'
nof_branches = 10

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
                                [str(i.item()) for i in labels]))


from models import CombinedModel, ClassificationMode

# load the pretrained model
# pretrained_model = "./CIFAR_target_model.pth"
# target_model = CIFAR_target_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
from simple_dla_classifier import SimpleDLA
target_model = SimpleDLA().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()
generator_models = nn.ModuleList([])

for perm_index, permutation in enumerate(permutations):
    # load the generator of adversarial examples
    pretrained_generator_path = os.path.join(
        permutation.name, 'models', f"netG_epoch_{generator_epochs}.pth")
    pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
    pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
    pretrained_G.eval()
    generator_models.append(pretrained_G)


combined_model = CombinedModel(target_model, permutations, generator_models,
                               classification_mode=ClassificationMode.AVERAGE_LOGITS)
nof_correct = 0
nof_samples = 0
for data in test_dataloader:
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    predictions = torch.argmax(combined_model(test_img), axis=1)
    nof_correct += (predictions == test_label).sum()
    nof_samples += predictions.shape[0]

accuracy = nof_correct / nof_samples * 100.0
print(f"Combined model accuracy: {accuracy:.2f}[%]")



# test the combined model adversarial detection capabilities:
nof_permutations = nof_branches
nof_classes = 10
adversarial_examples_predictions_matrix = torch.zeros(
    (len(cifar_dataset_test), nof_permutations, nof_classes))
pristine_examples_predictions_matrix = torch.zeros(
    (len(cifar_dataset_test), nof_permutations, nof_classes))
for i, data in enumerate(test_dataloader, 0):
    test_img, test_label = data
    test_img, test_label = test_img.to(device), test_label.to(device)
    target_labels = 2 * torch.ones_like(test_label).to(device)
    target_labels[test_label == 2] = 3
    test_adv_img = test_img + attack(combined_model, test_img, test_label,
                                     epsilon=0.3, alpha=1e-2,
                                     num_iter=40, y_targ=target_labels)
    # adversarial inference:
    results_dict = combined_model.explicit_forward(test_adv_img)
    logits = results_dict['permutation_index_to_classification'].values()
    logits_tensor = torch.cat([l.cpu().detach().unsqueeze(0) for l in logits])
    adversarial_examples_predictions_matrix[
    i * batch_size:(i + 1) * batch_size, 1:, ...] = logits_tensor.permute(1, 0,
                                                                          2)
    adversarial_examples_predictions_matrix[
    i * batch_size:(i + 1) * batch_size, 0, ...] = results_dict[
        'basic_classification'].cpu().detach()
    # pristine inference:
    results_dict = combined_model.explicit_forward(test_img)
    logits = results_dict['permutation_index_to_classification'].values()
    logits_tensor = torch.cat([l.cpu().detach().unsqueeze(0) for l in logits])
    pristine_examples_predictions_matrix[
    i * batch_size:(i + 1) * batch_size, 1:, ...] = logits_tensor.permute(1, 0,
                                                                          2)
    pristine_examples_predictions_matrix[
    i * batch_size:(i + 1) * batch_size, 0, ...] = results_dict[
        'basic_classification'].cpu().detach()

pristine_predictions = torch.argmax(
    pristine_examples_predictions_matrix.mean(axis=1), axis=1)
gt_labels = torch.tensor(cifar_dataset_test.targets)
pristine_accuracy = (pristine_predictions == gt_labels).sum().item() / len(
    gt_labels) * 100.0
print(f"Combined model accuracy: {pristine_accuracy:.2f}[%]")
pristine_predictions_every_branch = torch.argmax(
    pristine_examples_predictions_matrix, axis=2)
pristine_len_unique_samples = [len(x.unique()) for x in
                               pristine_predictions_every_branch]
adversarial_predictions_every_branch = torch.argmax(
    adversarial_examples_predictions_matrix, axis=2)
adversarial_len_unique_samples = [len(x.unique())
                                  for x in adversarial_predictions_every_branch]
fig1 = plt.figure()
plt.hist(pristine_len_unique_samples, alpha=0.5)
plt.hist(adversarial_len_unique_samples, alpha=0.5)
plt.legend(['pristine', 'adversarial'])
plt.title('The number of different classification labels across barnches')
plt.grid(True)
plt.savefig(
    os.path.join(root_path,
                 'number_of_different_classification_labels_across_branches'
                 '.png'))
plt.savefig(
    os.path.join(root_path,
                 'number_of_different_classification_labels_across_branches'
                 '.pdf'))
pristine_predictions_every_branch = torch.argmax(
    pristine_examples_predictions_matrix, axis=2)
adversarial_predictions_every_branch = torch.argmax(
    adversarial_examples_predictions_matrix, axis=2)


one_hot_pristine = torch.cat([
    one_hot(x, 10).float().mean(axis=0).sort(descending=True)[0].unsqueeze(0)
    for x in pristine_predictions_every_branch])
d_pristine = torch.tensor([torch.nn.L1Loss()(x[0], x[1:].mean())
                           for x in one_hot_pristine])
one_hot_adversarial = torch.cat([
    one_hot(x, 10).float().mean(axis=0).sort(descending=True)[0].unsqueeze(0)
    for x in adversarial_predictions_every_branch])
d_adversarial = torch.tensor([torch.nn.L1Loss()(x[0], x[1:].mean())
                              for x in one_hot_adversarial])
fig2 = plt.figure()
plt.hist(d_pristine.numpy(), bins=int(np.sqrt(d_pristine.shape[0])), alpha=0.5)
plt.hist(d_adversarial.numpy(), bins=int(np.sqrt(d_adversarial.shape[0])), alpha=0.5)
plt.legend(['pristine', 'adversarial'])
from sklearn.metrics import roc_auc_score
auc = roc_auc_score(torch.cat([torch.zeros_like(d_pristine),
                               torch.ones_like(d_adversarial)]),
                    torch.cat([d_pristine, d_adversarial]))
auc = auc if auc > 0.5 else 1-auc
plt.title(f'L1 distance of basic classification from mean classification over '
          f'permuted branches, AuC: {auc * 100.0:.2f}[%]')
plt.grid(True)
fig2.set_size_inches((12, 6))
plt.savefig(
    os.path.join(root_path,
                 'L1_distance_of_basic_classification_from_mean_classification_over_permuted_branches'
                 '.png'))
plt.savefig(
    os.path.join(root_path,
                 'L1_distance_of_basic_classification_from_mean_classification_over_permuted_branches'
                 '.pdf'))
print(f"Detection AuC: {auc * 100.0:.2f}[%]")
