import torch
import torchvision.datasets
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
import models
from models import CIFAR_target_net
from simple_dla_classifier import SimpleDLA

use_cuda = True
image_nc = 3
batch_size = 128
# epsilon = 0.3
epsilon = 8.0/256

gen_input_nc = image_nc

# Define what device we are using
print("CUDA Available: ",torch.cuda.is_available())
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")

# load the pretrained model
# pretrained_model = "./CIFAR_target_model.pth"
# target_model = CIFAR_targ
# et_net().to(device)
pretrained_model = "./simple_dla_cifar_net_no_normalization_92_acc.pth"
target_model = SimpleDLA().to(device)
target_model.load_state_dict(torch.load(pretrained_model))
target_model.eval()

# load the generator of adversarial examples
pretrained_generator_path = './models/netG_epoch_60.pth'
pretrained_G = models.Generator(gen_input_nc, image_nc).to(device)
pretrained_G.load_state_dict(torch.load(pretrained_generator_path))
pretrained_G.eval()

# test adversarial examples in CIFAR10 training dataset
cifar_dataset = torchvision.datasets.CIFAR10('./dataset',
                                           train=True,
                                           transform=transforms.ToTensor(),
                                           download=True)
train_dataloader = DataLoader(cifar_dataset,
                              batch_size=batch_size,
                              shuffle=False,
                              num_workers=1)
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

print('CIFAR10 training dataset:')
print('num_correct: ', num_correct.item())
print('accuracy of adv imgs in training set: %f\n'%(num_correct.item()/len(cifar_dataset)))

# test adversarial examples in CIFAR10 testing dataset
cifar_dataset_test = torchvision.datasets.CIFAR10('./dataset',
                                                train=False,
                                                transform=transforms.ToTensor(),
                                                download=True)
test_dataloader = DataLoader(cifar_dataset_test,
                             batch_size=batch_size,
                             shuffle=False,
                             num_workers=1)
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
print('accuracy of adv imgs in testing set: %f\n'%(num_correct.item()/len(cifar_dataset_test)))


import matplotlib.pyplot as plt


def plot_images(X,y,M,N, suptitle=None):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N*1.3,M*1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j].cpu().permute(1, 2, 0).numpy())
            title = ax[i][j].set_title("Label: {}".format(y[i*N+j]))
            plt.setp(title, color='b')
            ax[i][j].set_axis_off()
    # plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle)


def plot_predictions(X,y,yp,M,N, suptitle=None):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N*1.3,M*1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j].cpu().permute(1, 2, 0).numpy())
            title = ax[i][j].set_title("Pred: {}".format(yp[i*N+j].max(dim=0)[1]))
            plt.setp(title, color=('g' if yp[i*N+j].max(dim=0)[1] == y[i*N+j] else 'r'))
            ax[i][j].set_axis_off()
    if suptitle:
        plt.suptitle(suptitle)


def plot_preturbations(X,y,M,N, suptitle=None):
    f,ax = plt.subplots(M,N, sharex=True, sharey=True, figsize=(N*1.3,M*1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i*N+j][0].cpu().numpy(), cmap='gray')
            title = ax[i][j].set_title("Label: {}".format(y[i*N+j]))
            plt.setp(title, color='b')
            ax[i][j].set_axis_off()
    # plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle)



data = next(iter(test_dataloader))
test_img, test_label = data
test_img, test_label = test_img.to(device), test_label.to(device)
perturbation = pretrained_G(test_img)
perturbation = torch.clamp(perturbation, -epsilon, epsilon)
adv_img = perturbation + test_img
adv_img = torch.clamp(adv_img, 0, 1)
# plot pristine images
plot_images(test_img.detach(), test_label, 3, 6, suptitle='Test images')
# plot perturbations
plot_images(perturbation.detach(), test_label, 3, 6, suptitle='Perturbations')
plot_preturbations(perturbation.detach(), test_label, 3, 6,
                   suptitle='R-Channel Perturbations')
# plot pristine images with the classifier predictions
predictions = target_model(test_img)
plot_predictions(test_img.detach(), test_label, predictions, 3, 6,
                 suptitle='Classified Test images')
# plot adversarial images with the classifier predictions
predictions = target_model(adv_img)
plot_predictions(adv_img.detach(), test_label, predictions, 3, 6,
                 suptitle='Classified Adversarial images')
plt.show()
