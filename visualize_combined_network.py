import os
import torch
import torchvision.datasets
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

from torch import nn
from torch.utils.data import DataLoader

import models
from common import Permutation, pgd, pgd_linf, pgd_linf_targ, fgsm, CIFAR10_CLASSES
from common import plot_images_cifar as plot_images
from common import plot_predictions_cifar as plot_predictions
from models import CIFAR_target_net


use_cuda = True
image_nc = 3
batch_size = 32
nof_branches = 10
epsilon = 8.0 / 256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")
attack = pgd_linf_targ
root_dir = 'visualization'

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

# load the pretrained model
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
from simple_dla_classifier import SimpleDLA
target_model = SimpleDLA().to(device)
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

from models import CombinedModel, ClassificationMode

combined_model = CombinedModel(target_model, permutations, generator_models,
                               classification_mode=ClassificationMode.AVERAGE_LOGITS)
images, labels = next(iter(test_dataloader))
results_dict = combined_model.explicit_forward(images.to(device))
all_classifications = [
    results_dict['permutation_index_to_classification'][k][16]
    for k in results_dict['permutation_index_to_classification']]
basic_classification = results_dict['basic_classification'][16]
all_generated_images = [
    results_dict['permutation_index_to_permuted_image'][k][16]
    for k in results_dict['permutation_index_to_permuted_image']]

plt.subplot(5, 1, 1)
plt.imshow(images[16].cpu().detach().permute(1, 2, 0).numpy())
plt.xticks([]); plt.yticks([]); plt.title(f"label="
                                          f"{CIFAR10_CLASSES[labels[16]]} "
                                          f"({labels[16]})")
for i in range(9):
    plt.subplot(5, 3, i + 1 + 3*2)
    plt.stem(all_classifications[i].cpu().detach().numpy())
    # plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.xticks(list(range(10)))
    argmax = torch.argmax(all_classifications[i].cpu().detach()).item()
    plt.title(f'permutation #{i+1}, argmax={argmax}')
plt.subplot(5, 1, 2); plt.stem(basic_classification.cpu().detach().numpy())
plt.title('basic classification');
plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)],
           rotation=0)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(os.path.join(root_dir, 'permutation_classification.png'))
plt.savefig(os.path.join(root_dir, 'permutation_classification.pdf'))
plt.clf()

plt.suptitle(f'How targeted attack (to class label 2 = {CIFAR10_CLASSES[2]}) '
             f'affects image generation & classification:')
plt.subplot(4, 1, 1)
plt.imshow(images[16].cpu().detach().permute(1, 2, 0).numpy())
plt.xticks([]); plt.yticks([])
plt.title(f"Original image, label={CIFAR10_CLASSES[labels[16]]} ({labels[16]})")
for i in range(9):
    plt.subplot(4, 3, i + 1 + 3)
    plt.imshow(all_generated_images[i].cpu().detach().permute(1, 2, 0).numpy())
    result_label_idx = torch.argmax(
        all_classifications[i].cpu().detach()).item()
    result_label = CIFAR10_CLASSES[result_label_idx]
    plt.xticks([])
    plt.yticks([])
    plt.title(f'permutation #{i+1}, classification: {result_label} ({result_label_idx})')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(os.path.join(root_dir, 'permutation_generated_images.png'))
plt.savefig(os.path.join(root_dir, 'permutation_generated_images.pdf'))
plt.clf()
# Attack only the basic classifier:
test_img, test_label = images.to(device), labels.to(device)
test_adv_img = test_img + attack(target_model, test_img, test_label,
                                 epsilon=0.3, alpha=1e-2,
                                 num_iter=40, y_targ=2)
results_dict = combined_model.explicit_forward(test_adv_img.to(device))
all_classifications = [results_dict['permutation_index_to_classification'][k][0]
                       for k in results_dict['permutation_index_to_classification']]
basic_classification = results_dict['basic_classification'][0]
plt.suptitle(f'How targeted attack (to class label 2 = {CIFAR10_CLASSES[2]}'
             f') affects classification:')
plt.subplot(5, 1, 1);
plt.imshow(test_adv_img[0].cpu().detach().permute(1,2,0).numpy())
plt.xticks([]); plt.yticks([]);
plt.title(f"label={CIFAR10_CLASSES[labels[0]]}({labels[0]})")
for i in range(9):
    plt.subplot(5, 3, i + 1 + 3*2)
    plt.stem(all_classifications[i].cpu().detach().numpy())
    plt.xticks(list(range(10)))
    # plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.title(f'permutation #{i+1}')
plt.subplot(5, 1, 2); plt.stem(basic_classification.cpu().detach().numpy())
plt.title('basic classification')
plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_classification.png'))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_classification.pdf'))
plt.clf()

all_generated_images = [
    results_dict['permutation_index_to_permuted_image'][k][0]
    for k in results_dict['permutation_index_to_permuted_image']]
plt.suptitle(f'How targeted attack (to class label 2 = {CIFAR10_CLASSES[2]}) '
             f'affects image generation & classification:')
plt.subplot(4, 1, 1)
plt.imshow(test_adv_img[0].cpu().detach().permute(1, 2, 0).numpy())
plt.xticks([]); plt.yticks([]);
plt.title(f"label={CIFAR10_CLASSES[labels[0]]} ({labels[0]})")
for i in range(9):
    plt.subplot(4, 3, i + 1 + 3)
    plt.imshow(all_generated_images[i].cpu().detach().permute(1, 2, 0).numpy())
    result_label_idx = torch.argmax(
        all_classifications[i].cpu().detach()).item()
    result_label = CIFAR10_CLASSES[result_label_idx]
    plt.xticks([])
    plt.yticks([])
    plt.title(f'permutation #{i+1}, classification: '
              f'{result_label} ({result_label_idx})')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_generated_image.png'))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_generated_image.pdf'))
plt.clf()

# Attack the combined classifier:
test_img, test_label = images.to(device), labels.to(device)
test_adv_img = test_img + attack(combined_model, test_img, test_label,
                                 epsilon=0.3, alpha=1e-2,
                                 num_iter=40, y_targ=2)
results_dict = combined_model.explicit_forward(test_adv_img.to(device))
all_classifications = [
    results_dict['permutation_index_to_classification'][k][0]
    for k in results_dict['permutation_index_to_classification']]
basic_classification = results_dict['basic_classification'][0]
plt.suptitle('How targeted attack (to class label 2) on the combined model '
             'affects classification:')
plt.subplot(6, 1, 1)
plt.imshow(test_adv_img[0].cpu().detach().permute(1,2, 0).numpy())
plt.xticks([]); plt.yticks([]);
plt.title(f"label={CIFAR10_CLASSES[labels[0]]} ({labels[0]})")
logits = combined_model(test_adv_img.to(device))[0]
plt.subplot(6, 1, 2); plt.stem(logits.cpu().detach().numpy())
total_classification = torch.argmax(logits).item()
plt.title(f'total classification: {CIFAR10_CLASSES[total_classification]} '
          f'({total_classification})')
for i in range(9):
    plt.subplot(6, 3, i + 1 + 3*3)
    logits = all_classifications[i].cpu().detach()
    plt.stem(logits.numpy())
    classification = torch.argmax(logits).item()
    plt.xticks(list(range(10)))
    # plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
    plt.title(f'permutation #{i+1}, classification: '
              f'{CIFAR10_CLASSES[classification]} ({classification})')
plt.subplot(6, 1, 3)
basic_classification_logits = basic_classification.cpu().detach()
plt.stem(basic_classification_logits.numpy())
basic_classification_label = torch.argmax(basic_classification_logits).item()
plt.title(f'basic classification: '
          f'{CIFAR10_CLASSES[basic_classification_label]} '
          f'({basic_classification_label})')
plt.xticks(list(range(10)), labels=[CIFAR10_CLASSES[i] for i in range(10)], rotation=45)
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(
    os.path.join(root_dir,
                 'targeted_attack_permutation_classification_on_combined_model.png'))
plt.savefig(
    os.path.join(root_dir,
                 'targeted_attack_permutation_classification_on_combined_model.pdf'))
plt.clf()

all_generated_images = [
    results_dict['permutation_index_to_permuted_image'][k][0]
    for k in results_dict['permutation_index_to_permuted_image']]
plt.suptitle(f'How targeted attack (to class label {CIFAR10_CLASSES[2]} (2) '
             'affects image generation & classification:')
plt.subplot(4, 1, 1)
plt.imshow(test_adv_img[0].cpu().detach().permute(1, 2, 0).numpy())
plt.xticks([]); plt.yticks([]);
plt.title(f"label={CIFAR10_CLASSES[labels[0]]} ({labels[0]})")
for i in range(9):
    plt.subplot(4, 3, i + 1 + 3)
    plt.imshow(all_generated_images[i].cpu().detach().permute(1, 2, 0).numpy())
    result_label_idx = torch.argmax(
        all_classifications[i].cpu().detach()).item()
    result_label = CIFAR10_CLASSES[result_label_idx]
    plt.xticks([])
    plt.yticks([])
    plt.title(f'permutation #{i+1}, classification: {result_label}')
plt.tight_layout()
fig = plt.gcf()
fig.set_size_inches((12, 12))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_generated_image_on_combined_model.png'))
plt.savefig(os.path.join(root_dir,
                         'targeted_attack_permutation_generated_image_on_combined_model.pdf'))
plt.clf()

