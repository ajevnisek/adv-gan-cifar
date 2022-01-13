import torch.nn as nn
import torch.nn.functional as F


# Target Model definition
class CIFAR_target_net(nn.Module):
    def __init__(self):
        super(CIFAR_target_net, self).__init__()
        self.conv1 = nn.Conv2d(3, 32, kernel_size=3)
        self.conv2 = nn.Conv2d(32, 32, kernel_size=3)
        self.conv3 = nn.Conv2d(32, 128, kernel_size=3)
        self.conv4 = nn.Conv2d(128, 128, kernel_size=3)

        self.fc1 = nn.Linear(128*5*5, 512)
        self.fc2 = nn.Linear(512, 512)
        self.fc3 = nn.Linear(512, 256)
        self.logits = nn.Linear(256, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, 128*5*5)
        x = F.relu(self.fc1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = self.logits(x)
        return x


class Discriminator(nn.Module):
    def __init__(self, image_nc):
        super(Discriminator, self).__init__()
        # MNIST: 1*28*28
        model = [
            nn.Conv2d(image_nc, 8, kernel_size=4, stride=2, padding=0, bias=True),
            nn.LeakyReLU(0.2),
            # 8*13*13
            nn.Conv2d(8, 16, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(16),
            nn.LeakyReLU(0.2),
            # 16*5*5
            nn.Conv2d(16, 32, kernel_size=4, stride=2, padding=0, bias=True),
            nn.BatchNorm2d(32),
            nn.LeakyReLU(0.2),
            nn.Conv2d(32, 1, 1),
            nn.Sigmoid()
            # 32*1*1
        ]
        self.model = nn.Sequential(*model)

    def forward(self, x):
        output = self.model(x).squeeze()
        return output


class Generator(nn.Module):
    def __init__(self,
                 gen_input_nc,
                 image_nc,
                 ):
        super(Generator, self).__init__()

        encoder_lis = [
            # MNIST:1*28*28
            nn.Conv2d(gen_input_nc, 8, kernel_size=3, stride=1, padding=0, bias=True),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # 8*26*26
            nn.Conv2d(8, 16, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # 16*12*12
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=0, bias=True),
            nn.InstanceNorm2d(32),
            nn.ReLU(),
            # 32*5*5
        ]

        bottle_neck_lis = [ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),
                       ResnetBlock(32),]

        decoder_lis = [
            nn.ConvTranspose2d(32, 16, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(16),
            nn.ReLU(),
            # state size. 16 x 11 x 11
            nn.ConvTranspose2d(16, 8, kernel_size=3, stride=2, padding=0, bias=False),
            nn.InstanceNorm2d(8),
            nn.ReLU(),
            # state size. 8 x 23 x 23
            nn.ConvTranspose2d(8, image_nc, kernel_size=6, stride=1, padding=0, bias=False),
            nn.Tanh()
            # state size. image_nc x 28 x 28
        ]

        self.encoder = nn.Sequential(*encoder_lis)
        self.bottle_neck = nn.Sequential(*bottle_neck_lis)
        self.decoder = nn.Sequential(*decoder_lis)

    def forward(self, x):
        x = self.encoder(x)
        x = self.bottle_neck(x)
        x = self.decoder(x)
        return x


class Encoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of input channels of the image. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the first convolutional layers. Deeper layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the encoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.net = nn.Sequential(
            nn.Conv2d(num_input_channels, c_hid, kernel_size=3, padding=1,
                      stride=2),  # 32x32 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            # 16x16 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1, stride=2),
            # 8x8 => 4x4
            act_fn(),
            nn.Flatten(),  # Image grid to single feature vector
            nn.Linear(2 * 16 * c_hid, latent_dim)
        )

    def forward(self, x):
        return self.net(x)


class Decoder(nn.Module):

    def __init__(self,
                 num_input_channels: int,
                 base_channel_size: int,
                 latent_dim: int,
                 act_fn: object = nn.GELU):
        """
        Inputs:
            - num_input_channels : Number of channels of the image to reconstruct. For CIFAR, this parameter is 3
            - base_channel_size : Number of channels we use in the last convolutional layers. Early layers might use a duplicate of it.
            - latent_dim : Dimensionality of latent representation z
            - act_fn : Activation function used throughout the decoder network
        """
        super().__init__()
        c_hid = base_channel_size
        self.linear = nn.Sequential(
            nn.Linear(latent_dim, 2 * 16 * c_hid),
            act_fn()
        )
        self.net = nn.Sequential(
            nn.ConvTranspose2d(2 * c_hid, 2 * c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            # 4x4 => 8x8
            act_fn(),
            nn.Conv2d(2 * c_hid, 2 * c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(2 * c_hid, c_hid, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            # 8x8 => 16x16
            act_fn(),
            nn.Conv2d(c_hid, c_hid, kernel_size=3, padding=1),
            act_fn(),
            nn.ConvTranspose2d(c_hid, num_input_channels, kernel_size=3,
                               output_padding=1, padding=1, stride=2),
            # 16x16 => 32x32
            nn.Tanh()
            # The input images is scaled between -1 and 1, hence the output has to be bounded as well
        )

    def forward(self, x):
        x = self.linear(x)
        x = x.reshape(x.shape[0], -1, 4, 4)
        x = self.net(x)
        return x


class Autoencoder(nn.Module):

    def __init__(self,
                 base_channel_size: int,
                 latent_dim: int,
                 encoder_class = Encoder,
                 decoder_class = Decoder,
                 num_input_channels: int = 3,
                 width: int = 32,
                 height: int = 32):
        super().__init__()
        # Saving hyperparameters of autoencoder
        self.save_hyperparameters()
        # Creating encoder and decoder
        self.encoder = encoder_class(num_input_channels, base_channel_size,
                                     latent_dim)
        self.decoder = decoder_class(num_input_channels, base_channel_size,
                                     latent_dim)
        # Example input array needed for visualizing the graph of the network
        self.example_input_array = torch.zeros(2, num_input_channels, width,
                                               height)

    def forward(self, x):
        """
        The forward function takes in an image and returns the reconstructed image
        """
        x -= 0.5
        x /= 0.5
        z = self.encoder(x)
        x_hat = self.decoder(z)
        x_hat += 1
        x /= 2.0
        return x_hat



# Define a resnet block
# modified from https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/master/models/networks.py
class ResnetBlock(nn.Module):
    def __init__(self, dim, padding_type='reflect', norm_layer=nn.BatchNorm2d, use_dropout=False, use_bias=False):
        super(ResnetBlock, self).__init__()
        self.conv_block = self.build_conv_block(dim, padding_type, norm_layer, use_dropout, use_bias)

    def build_conv_block(self, dim, padding_type, norm_layer, use_dropout, use_bias):
        conv_block = []
        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim),
                       nn.ReLU(True)]
        if use_dropout:
            conv_block += [nn.Dropout(0.5)]

        p = 0
        if padding_type == 'reflect':
            conv_block += [nn.ReflectionPad2d(1)]
        elif padding_type == 'replicate':
            conv_block += [nn.ReplicationPad2d(1)]
        elif padding_type == 'zero':
            p = 1
        else:
            raise NotImplementedError('padding [%s] is not implemented' % padding_type)

        conv_block += [nn.Conv2d(dim, dim, kernel_size=3, padding=p, bias=use_bias),
                       norm_layer(dim)]

        return nn.Sequential(*conv_block)

    def forward(self, x):
        out = x + self.conv_block(x)
        return out


import torch
import torch.nn. functional as F


from enum import Enum, auto

from torch import nn


use_cuda = True
image_nc = 3
batch_size = 128
epsilon = 8.0 / 256.0
gen_input_nc = image_nc
device = torch.device("cuda" if (use_cuda and torch.cuda.is_available()) else "cpu")


class ClassificationMode(Enum):
    AVERAGE_LOGITS = auto
    MAX_ENTROPY_BRANCH = auto


class Entropy(nn.Module):
    def __init__(self):
        super(Entropy, self).__init__()

    def forward(self, x):
        b = F.softmax(x, dim=2) * F.log_softmax(x, dim=2)
        b = -1.0 * b.sum(axis=1)
        return b


class CombinedModel(nn.Module):
    def __init__(self, classifier, permutations, generators,
                 classification_mode=ClassificationMode.AVERAGE_LOGITS):
        super(CombinedModel, self).__init__()
        self.classifier = classifier
        self.permutations = permutations
        self.generators = generators
        self.classification_mode = classification_mode
        self.entropy_criterion = Entropy()

    def forward(self, x):
        classifications = []
        basic_classification = self.classifier(x)
        classifications.append(basic_classification.unsqueeze(0))
        for permutation, generator in zip(self.permutations,
                                          self.generators):
            perturbation = generator(x)
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            transformed_image = x + perturbation
            logits = self.classifier(transformed_image)
            normalized_logits = permutation.backward_permute_logits(logits)
            classifications.append(normalized_logits.unsqueeze(0))

        # all_classifications shape:
        # nof_permutations X batch_size X nof_classes
        all_classifications = torch.cat(classifications)

        if self.classification_mode == ClassificationMode.AVERAGE_LOGITS:
            total_classification = all_classifications.mean(axis=0)
            return total_classification
        elif self.classification_mode == ClassificationMode.MAX_ENTROPY_BRANCH:
            entropy_per_classification = self.entropy_criterion(
                all_classifications.permute(1, 0, 2))
            maximal_entropy_branch = torch.argmax(entropy_per_classification,
                                                  axis=1)
            total_classification = torch.cat(
                [all_classifications[x, i, ...].unsqueeze(0) for x, i in
                 zip(maximal_entropy_branch,
                     range(all_classifications.shape[1]))])
            return total_classification
        else:
            total_classification = torch.cat(classifications).mean(axis=0)
            return total_classification

    def explicit_forward(self, x):
        basic_classification = self.classifier(x)
        permutation_index_to_classification = {}
        permutation_index_to_permuted_image = {}
        for idx, (permutation, generator) in enumerate(zip(self.permutations,
                                                           self.generators)):
            perturbation = generator(x)
            perturbation = torch.clamp(perturbation, -epsilon, epsilon)
            transformed_image = x + perturbation
            logits = self.classifier(transformed_image)
            normalized_logits = permutation.backward_permute_logits(logits)
            permutation_index_to_classification[idx] = normalized_logits
            permutation_index_to_permuted_image[idx] = x + perturbation
        results_dict = {'basic_classification': basic_classification,
                        'permutation_index_to_classification':
                            permutation_index_to_classification,
                        'permutation_index_to_permuted_image':
                            permutation_index_to_permuted_image}
        return results_dict
