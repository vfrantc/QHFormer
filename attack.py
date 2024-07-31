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

def fgsm_attack(model, images, eps=8/255):
    """
    FGSM attack as described in 'Explaining and Harnessing Adversarial Examples'
    https://arxiv.org/pdf/1412.6572

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        eps (float): maximum perturbation

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    images.requires_grad = True
    labels = model(images)

    loss = nn.MSELoss(reduction='sum')(labels, images)
    model.zero_grad()
    loss.backward()
    grad = images.grad.data

    adv_images = images + eps * grad.sign()
    adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    return adv_images

def ifgsm_attack(model, images, eps=8/255, alpha=2/255, attack_iters=5):
    """
    IFGSM attack as described in 'Adversarial Machine Learning at Scale'
    https://arxiv.org/pdf/1611.01236

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        attack_iters (int): number of attack iterations.

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    upper_limit, lower_limit = 1, 0
    device = next(model.parameters()).device
    delta = torch.zeros_like(images).to(device)
    delta.uniform_(-eps, eps)
    delta = torch.clamp(delta, lower_limit - images, upper_limit - images)
    delta.requires_grad = True
    with torch.no_grad():
      labels = model(images)
    for _ in range(attack_iters):
        robust_output = model((images + delta))
        loss = nn.MSELoss(reduction='sum')(robust_output, labels)
        grad = torch.autograd.grad(loss, [delta])[0].detach()
        d = delta
        g = grad
        x = images
        d = torch.clamp(d + alpha * torch.sign(g), min=-eps, max=eps)
        d = torch.clamp(d, lower_limit - x, upper_limit - x)
        delta.data = d
    max_delta = delta.detach()
    return max_delta + images

def pgd_attack(model, images, eps=8/255, alpha=2/255, steps=10, random_start=True):
    """
    PGD attack as described in 'Towards Deep Learning Models Resistant to Adversarial Attacks'
    https://arxiv.org/abs/1706.06083

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        steps (int): number of steps.
        random_start (bool): using random initialization of delta.

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    labels = model(images).detach()

    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    else:
        adv_images = images.clone().detach()

    for _ in range(steps):
        adv_images.requires_grad = True
        outputs = model(adv_images)

        loss = nn.MSELoss(reduction='sum')(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]

        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images