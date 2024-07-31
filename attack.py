'''
07/28/2024 Vladimir Frants
Attacks are implemented following original papers and reference
implementations from https://github.com/Harry24k/adversarial-attacks-pytorch
Attacks on image processing techniques are implemented following the approach
from https://github.com/Xiaofeng-life/AADN_Dehazing.git

We use predicted image as "label", so the goal is to get an adversarial
example that gets us as far from the initial predicted image as possible.
'''

import torch
import torch.nn as nn
import torch.autograd
import torch.optim as optim

def clamp(input, min=None, max=None):
    if min is not None and max is not None:
        return torch.max(torch.min(input, max), min)
    elif min is not None:
        return torch.max(input, min)
    elif max is not None:
        return torch.min(input, max)
    else:
        return input

def fgsm_attack(model, images, eps=8/255):
    """
    FGSM attack as described in 'Explaining and Harnessing Adversarial Examples'
    https://arxiv.org/pdf/1412.6572

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        labels (torch.Tensor): true labels of the images.
        eps (float): maximum perturbation

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    images = images.clone().detach().to(CUR_DEVICE)
    labels = labels.clone().detach().to(CUR_DEVICE)

    images.requires_grad = True
    outputs = model(images)
    loss = nn.MSELoss(reduction='sum')(outputs, images)
    model.zero_grad()
    loss.backward()

    grad = images.grad.data
    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    return adv_images

def ifgsm_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10):
    """
    I-FGSM attack as described in 'Iterative Methods for Adversarial Attacks'

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        labels (torch.Tensor): true labels of the images.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    images = images.clone().detach().to(CUR_DEVICE)
    labels = labels.clone().detach().to(CUR_DEVICE)

    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = nn.MSELoss(reduction='sum')(outputs, images)
        model.zero_grad()
        loss.backward()

        grad = adv_images.grad.data
        adv_images = adv_images + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images

def mi_fgsm_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, decay=1.0):
    """
    MI-FGSM attack as described in 'Boosting Adversarial Attacks with Momentum'

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        labels (torch.Tensor): true labels of the images.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of iterations. (Default: 10)
        decay (float): momentum factor. (Default: 1.0)

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    images = images.clone().detach().to(CUR_DEVICE)
    labels = labels.clone().detach().to(CUR_DEVICE)

    momentum = torch.zeros_like(images).detach().to(CUR_DEVICE)
    loss = nn.MSELoss(reduction='sum')

    adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        cost = loss(outputs, images)

        # Update the gradient
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]
        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        momentum = decay * momentum + grad

        # Update the adversarial images
        adv_images = adv_images.detach() + alpha * momentum.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def pgd_attack(model, images, labels, eps=8 / 255, alpha=2 / 255, steps=10, random_start=True):
    """
    PGD attack as described in 'Towards Deep Learning Models Resistant to Adversarial Attacks'

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        labels (torch.Tensor): true labels of the images.
        eps (float): maximum perturbation. (Default: 8/255)
        alpha (float): step size. (Default: 2/255)
        steps (int): number of steps. (Default: 10)
        random_start (bool): using random initialization of delta. (Default: True)

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    images = images.clone().detach().to(CUR_DEVICE)
    labels = labels.clone().detach().to(CUR_DEVICE)

    loss = nn.CrossEntropyLoss()
    adv_images = images.clone().detach()

    if random_start:
        # Starting at a uniformly random point
        adv_images = adv_images + torch.empty_like(adv_images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        cost = loss(outputs, labels)

        # Update adversarial images
        grad = torch.autograd.grad(cost, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def cw_attack(model, images, c=1, kappa=0, steps=50, lr=0.01):
    """
    C&W attack as described in 'Towards Evaluating the Robustness of Neural Networks'

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        c (float): parameter for box-constraint. (Default: 1)
        kappa (float): confidence parameter. (Default: 0)
        steps (int): number of steps. (Default: 50)
        lr (float): learning rate of the Adam optimizer. (Default: 0.01)

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    images = images.clone().detach().to(CUR_DEVICE)

    # Initialize w as the inverse tanh of images
    w = 0.5 * torch.log((1 + images) / (1 - images))
    w.requires_grad = True

    optimizer = optim.Adam([w], lr=lr)

    MSELoss = nn.MSELoss(reduction='sum')

    best_adv_images = images.clone().detach()
    best_L2 = float('inf')
    prev_cost = float('inf')

    for step in range(steps):
        adv_images = torch.tanh(w) * 0.5 + 0.5
        outputs = model(adv_images)

        # L2 loss
        L2_loss = MSELoss(adv_images, images)

        # f_loss as per the paper
        f_loss = torch.clamp(outputs - kappa, min=0).sum()

        cost = L2_loss + c * f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Update best adversarial images
        if L2_loss < best_L2:
            best_L2 = L2_loss
            best_adv_images = adv_images.clone().detach()

        # Early stop if the cost is not decreasing
        if step % max(steps // 10, 1) == 0:
            if cost.item() > prev_cost:
                break
            prev_cost = cost.item()

    return best_adv_images



# Example usage:
# model: your neural network model
# images: the input images to attack
# labels: the ground-truth labels for the images

# FGSM Attack
adv_images_fgsm = fgsm_attack(model, images, labels, eps=8/255)

# I-FGSM Attack
adv_images_ifgsm = ifgsm_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10)

# PGD Attack
adv_images_pgd = pgd_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10)

# MI-FGSM Attack
adv_images_mifgsm = mi_fgsm_attack(model, images, labels, eps=8/255, alpha=2/255, steps=10, decay=1.0)

# C&W Attack
adv_images_cw = cw_attack(model, images, c=1, kappa=0, steps=50, lr=0.01)

