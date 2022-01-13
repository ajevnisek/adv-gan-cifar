import torch.nn as nn
import torch
import numpy as np
import models
import torch.nn.functional as F
import torchvision
import os

models_path = './models/'


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)


def permutation(labels, model_num_labels):
    return (labels + 1) % model_num_labels


class AdvGAN_Attack:
    def __init__(self,
                 device,
                 model,
                 model_num_labels,
                 image_nc,
                 box_min,
                 box_max,
                 epsilon,
                 target_attack_label=0,
                 permutation_function=permutation,
                 target_path='',
                 is_cifar10=False):
        output_nc = image_nc
        self.device = device
        self.model_num_labels = model_num_labels
        self.model = model
        self.input_nc = image_nc
        self.output_nc = output_nc
        self.box_min = box_min
        self.box_max = box_max
        self.epsilon = epsilon
        self.target_attack_label = target_attack_label
        self.permutation_function = permutation_function
        self.target_path = target_path
        self.is_cifar10 = is_cifar10

        self.gen_input_nc = image_nc
        if is_cifar10:
            self.netG = models.Generator(self.gen_input_nc, image_nc).to(device)
        else:
            self.netG = models.Autoencoder(base_channel_size=32,
                                           latent_dim=256).to(device)
        self.netDisc = models.Discriminator(image_nc).to(device)

        # initialize all weights
        self.netG.apply(weights_init)
        self.netDisc.apply(weights_init)

        # initialize optimizers
        self.optimizer_G = torch.optim.Adam(self.netG.parameters(),
                                            lr=0.001)
        if self.is_cifar10:
            self.scheduler_G = torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer_G,
                mode='min',
                factor=0.2,
                patience=20,
                min_lr=5e-5)
        self.optimizer_D = torch.optim.Adam(self.netDisc.parameters(),
                                            lr=0.001)
        os.makedirs(os.path.join(self.target_path, models_path), exist_ok=True)

    def train_batch(self, x, labels):
        # optimize D
        perturbation = self.netG(x)

        # add a clipping trick
        adv_images = torch.clamp(perturbation, -self.epsilon, self.epsilon) + x
        adv_images = torch.clamp(adv_images, self.box_min, self.box_max)

        self.optimizer_D.zero_grad()
        pred_real = self.netDisc(x)
        loss_D_real = F.mse_loss(pred_real, torch.ones_like(pred_real, device=self.device))
        loss_D_real.backward()

        pred_fake = self.netDisc(adv_images.detach())
        loss_D_fake = F.mse_loss(pred_fake, torch.zeros_like(pred_fake, device=self.device))
        loss_D_fake.backward()
        loss_D_GAN = loss_D_fake + loss_D_real
        self.optimizer_D.step()

        # optimize G
        self.optimizer_G.zero_grad()

        # cal G's loss in GAN
        pred_fake = self.netDisc(adv_images)
        loss_G_fake = F.mse_loss(pred_fake, torch.ones_like(pred_fake, device=self.device))
        loss_G_fake.backward(retain_graph=True)

        # calculate perturbation norm
        C = 0.1
        loss_perturb = torch.mean(torch.norm(perturbation.view(perturbation.shape[0], -1), 2, dim=1))
        # loss_perturb = torch.max(loss_perturb - C, torch.zeros(1, device=self.device))

        # cal adv loss
        logits_model = self.model(adv_images)
        # probs_model = F.softmax(logits_model, dim=1)
        # onehot_labels = torch.eye(self.model_num_labels, device=self.device)[labels]

        # C&W loss function
        # real = torch.sum(onehot_labels * probs_model, dim=1)
        # other, _ = torch.max(
        #     (1 - onehot_labels) * probs_model - onehot_labels * 10000, dim=1)
        # zeros = torch.zeros_like(other)
        # loss_adv = torch.max(real - other, zeros)
        # loss_adv = torch.sum(loss_adv)
        # import ipdb; ipdb.set_trace()

        # maximize cross_entropy loss
        # loss_adv = -F.mse_loss(logits_model, onehot_labels)
        # loss_adv = -F.cross_entropy(logits_model, labels)

        # one target label
        # loss_adv = nn.CrossEntropyLoss()(
        #     logits_model, torch.ones_like(labels) * self.target_attack_label)
        # permutation
        target_labels = self.permutation_function(labels, self.model_num_labels)
        loss_adv = nn.CrossEntropyLoss()(logits_model, target_labels)
        # import ipdb; ipdb.set_trace()
        adv_lambda = 10
        pert_lambda = 1
        # import ipdb; ipdb.set_trace()
        loss_G = adv_lambda * loss_adv + pert_lambda * loss_perturb

        loss_G.backward()
        self.optimizer_G.step()

        return loss_D_GAN.item(), loss_G_fake.item(), loss_perturb.item(), loss_adv.item()

    def train(self, train_dataloader, epochs, test_loader=None):
        for epoch in range(1, epochs+1):
            loss_D_sum = 0
            loss_G_fake_sum = 0
            loss_perturb_sum = 0
            loss_adv_sum = 0
            for i, data in enumerate(train_dataloader, start=0):
                images, labels = data
                images, labels = images.to(self.device), labels.to(self.device)

                loss_D_batch, loss_G_fake_batch, loss_perturb_batch, loss_adv_batch = \
                    self.train_batch(images, labels)
                loss_D_sum += loss_D_batch
                loss_G_fake_sum += loss_G_fake_batch
                loss_perturb_sum += loss_perturb_batch
                loss_adv_sum += loss_adv_batch

            if self.is_cifar10:
                self.scheduler_G.step(loss_G_fake_sum)
            # print statistics
            num_batch = len(train_dataloader)
            print("epoch %d:\nloss_D: %.3f, loss_G_fake: %.3f,\
             \nloss_perturb: %.3f, loss_adv: %.3f, \n" %
                  (epoch, loss_D_sum/num_batch, loss_G_fake_sum/num_batch,
                   loss_perturb_sum/num_batch, loss_adv_sum/num_batch))

            # save generator
            if epoch % 20 == 0:
                # netG_file_name = models_path + 'netG_epoch_' + str(epoch) + '.pth'
                # torch.save(self.netG.state_dict(), netG_file_name)
                netG_file_name = os.path.join(
                    self.target_path, models_path, f'netG_epoch_{epoch}.pth')
                torch.save(self.netG.state_dict(), netG_file_name)
                if test_loader:
                    nof_correct = 0
                    nof_samples = 0
                    for data in test_loader:
                        images, labels = data
                        images, labels = images.to(self.device), labels.to(
                            self.device)
                        perturbation = self.netG(images)
                        # add a clipping trick
                        adv_images = torch.clamp(perturbation, -self.epsilon,
                                                 self.epsilon) + images
                        adv_images = torch.clamp(adv_images, self.box_min,
                                                 self.box_max)
                        logits_model = self.model(adv_images)
                        predictions = torch.argmax(logits_model, axis=1)
                        # target_labels = self.permutation_function(
                        #     labels, self.model_num_labels)
                        target_labels = torch.ones_like(labels) * self.target_attack_label
                        nof_correct += (predictions == target_labels).sum()
                        nof_samples += predictions.shape[0]
                    accuracy = nof_correct / nof_samples * 100.0
                    print(f"epoch: {epoch}, Generator succeeds to fool the "
                          f"classifier in {accuracy:.2f}[%]")



