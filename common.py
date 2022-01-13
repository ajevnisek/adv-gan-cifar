import torch
import torch.nn as nn
import matplotlib.pyplot as plt


CIFAR10_CLASSES = ('plane', 'car', 'bird', 'cat',
                   'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


class Permutation:
    def __init__(self, name, new_labels, class_names):
        self.name = name
        self.new_labels = new_labels
        self.num_labels = len(new_labels)
        self.permutation_function = lambda labels, num_labels: \
            self.new_labels[labels]
        self.class_names = class_names

    def backward_permutation_function(self, future_labels):
        reverse_map = torch.argsort(self.new_labels)
        old_labels = reverse_map[future_labels]
        return old_labels

    def backward_permute_logits(self, future_logits):
        old_logits = future_logits[:, self.new_labels]
        return old_logits

    def __repr__(self):
        repr_string = ",\n".join([f"{label}->{nl}" for label, nl in zip(
            self.class_names,
            [self.class_names[i] for i in self.new_labels])])
        return repr_string


def pgd_linf_targ(model, X, y, epsilon, alpha, num_iter, y_targ):
    """ Construct targeted adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        yp = model(X + delta)
        loss = (yp[:,y_targ] - yp.gather(1,y[:,None])[:,0]).sum()
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def fgsm(model, X, y, epsilon):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    loss = nn.CrossEntropyLoss()(model(X + delta), y)
    loss.backward()
    return epsilon * delta.grad.detach().sign()


def pgd(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + X.shape[0]*alpha*delta.grad.data).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def pgd_linf(model, X, y, epsilon, alpha, num_iter):
    """ Construct FGSM adversarial examples on the examples X"""
    delta = torch.zeros_like(X, requires_grad=True)
    for t in range(num_iter):
        loss = nn.CrossEntropyLoss()(model(X + delta), y)
        loss.backward()
        delta.data = (delta + alpha*delta.grad.detach().sign()).clamp(-epsilon,epsilon)
        delta.grad.zero_()
    return delta.detach()


def plot_images_mnist(X, y, M, N, suptitle=None):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True,
                         figsize=(N * 1.3, M * 1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i * N + j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title("Label: {}".format(y[i * N + j]))
            plt.setp(title, color='b')
            ax[i][j].set_axis_off()
    # plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle)


def plot_predictions_mnist(X, y, yp, M, N, suptitle=None):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True,
                         figsize=(N * 1.3, M * 1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i * N + j][0].cpu().numpy(), cmap="gray")
            title = ax[i][j].set_title(
                "Pred: {}".format(yp[i * N + j].max(dim=0)[1]))
            plt.setp(title, color=(
                'g' if yp[i * N + j].max(dim=0)[1] == y[i * N + j] else 'r'))
            ax[i][j].set_axis_off()
    if suptitle:
        plt.suptitle(suptitle)


def plot_images_cifar(X, y, M, N, suptitle=None):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True,
                         figsize=(N * 1.3, M * 1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i * N + j].permute(1, 2, 0).cpu().numpy(),
                            cmap="gray")
            label = CIFAR10_CLASSES[y[i * N + j]]
            title = ax[i][j].set_title(f"Label: {label}")
            plt.setp(title, color='b')
            ax[i][j].set_axis_off()
    # plt.tight_layout()
    if suptitle:
        plt.suptitle(suptitle)


def plot_predictions_cifar(X, y, yp, M, N, suptitle=None):
    f, ax = plt.subplots(M, N, sharex=True, sharey=True,
                         figsize=(N * 1.3, M * 1.5))
    for i in range(M):
        for j in range(N):
            ax[i][j].imshow(X[i * N + j].permute(1, 2, 0).cpu().numpy(),
                            cmap="gray")
            label = CIFAR10_CLASSES[yp[i * N + j].max(dim=0)[1]]
            title = ax[i][j].set_title(f"Pred: {label}")
            plt.setp(title, color=(
                'g' if yp[i * N + j].max(dim=0)[1] == y[i * N + j] else 'r'))
            ax[i][j].set_axis_off()
    if suptitle:
        plt.suptitle(suptitle)
