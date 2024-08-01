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
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.stats as st

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

def mifgsm_attack(model, images, eps=8 / 255, attack_iters=10, decay=1.0, random_start=True):
    """
    MIFGSM attack as described in 'Boosting Adversarial Attacks with Momentum'
    https://arxiv.org/abs/1710.06081

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        eps (float): maximum perturbation.
        alpha (float): step size.
        steps (int): number of iterations.
        decay (float): momentum factor.
        random_start (bool): using random initialization of delta.

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    alpha = eps / attack_iters
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)

    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-eps, eps)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    else:
        adv_images = images.clone().detach()

    images.requires_grad = True
    with torch.no_grad():
        labels = model(images).detach()

    momentum = torch.zeros_like(images).detach()
    loss_func = nn.MSELoss(reduction='mean')

    for _ in range(attack_iters):
        adv_images.requires_grad = True
        outputs = model(adv_images)
        loss = loss_func(outputs, labels)
        grad = torch.autograd.grad(loss, adv_images, retain_graph=False, create_graph=False)[0]
        print(grad.min(), grad.max())

        grad = grad / torch.mean(torch.abs(grad), dim=(1, 2, 3), keepdim=True)
        grad = grad + momentum * decay

        momentum = grad
        adv_images = adv_images.detach() + alpha * grad.sign()
        delta = torch.clamp(adv_images - images, min=-eps, max=eps)
        adv_images = torch.clamp(images + delta, min=0, max=1).detach()

    return adv_images


def cw_attack(model, images, c=1, kappa=0, steps=50, lr=0.01):
    """
    C&W attack as described in 'Towards Evaluating the Robustness of Neural Networks'
    https://arxiv.org/abs/1608.04644

    Arguments:
        model (nn.Module): model to attack.
        images (torch.Tensor): original images to attack.
        c (float): parameter for box-constraint.
        kappa (float): confidence parameter.
        steps (int): number of steps.
        lr (float): learning rate of the Adam optimizer.

    Returns:
        adv_images (torch.Tensor): adversarial images.
    """
    device = next(model.parameters()).device
    images = images.clone().detach().to(device)
    labels = model(images).detach()

    def tanh_space(x): # I am not sure why this important. I havent read the paper
        return 0.5 * (torch.tanh(x) + 1) # 0.5 * (torch.tanh(x) + 1)

    def inverse_tanh_space(x):
        return 0.5 * torch.log((1 + x) / (1 - x)) # 0.5 * torch.log((1 + x) / (1 - x))

    w = inverse_tanh_space(images).detach()
    w.requires_grad = True

    best_adv_images = images.clone().detach()
    best_L2 = 1e10 * torch.ones((len(images))).to(device)
    prev_cost = 1e10

    MSELoss = nn.MSELoss(reduction="none")
    Flatten = nn.Flatten()

    optimizer = optim.Adam([w], lr=lr)

    for step in range(steps):
        adv_images = tanh_space(w)

        current_L2 = MSELoss(Flatten(adv_images), Flatten(images)).sum(dim=1)
        L2_loss = current_L2.sum()

        outputs = model(adv_images)
        f_loss = torch.max(torch.zeros_like(labels), torch.max(outputs, dim=1)[0] - labels.max(dim=1)[0] + kappa).sum()

        cost = L2_loss + c * f_loss

        optimizer.zero_grad()
        cost.backward()
        optimizer.step()

        # Update adversarial images
        condition = (current_L2.detach() < best_L2).float()
        best_L2 = condition * current_L2.detach() + (1 - condition) * best_L2

        condition = condition.view([-1] + [1] * (len(images.shape) - 1))
        best_adv_images = condition * adv_images.detach() + (1 - condition) * best_adv_images

        # Early stop when loss does not converge
        if step % max(steps // 10, 1) == 0:
            if cost.item() > prev_cost:
                return best_adv_images
            prev_cost = cost.item()

    return best_adv_images

def pifgsmpp_attack(model, images, max_epsilon=8 / 255, num_iter_set=10, momentum=1.0, amplification=10.0, prob=0.7,
                    project_factor=0.8, random_start=True):
    def clip_by_tensor(t, t_min, t_max):
        result = (t >= t_min).float() * t + (t < t_min).float() * t_min
        result = (result <= t_max).float() * result + (result > t_max).float() * t_max
        return result

    def project_noise(images, P_kern, padding_size):
        images = F.conv2d(images, P_kern, padding=(padding_size, padding_size), groups=3)
        return images

    def gaussian_kern(kernlen=21, nsig=3):
        """Returns a 2D Gaussian kernel array."""
        x = np.linspace(-nsig, nsig, kernlen)
        kern1d = st.norm.pdf(x)
        kernel_raw = np.outer(kern1d, kern1d)
        kernel = kernel_raw / kernel_raw.sum()
        kernel = kernel.astype(np.float32)
        stack_kernel = np.stack([kernel, kernel, kernel]).swapaxes(2, 0)
        stack_kernel = np.expand_dims(stack_kernel, 1)
        return stack_kernel

    def project_kern(kern_size, device):
        kern = np.ones((kern_size, kern_size), dtype=np.float32) / (kern_size ** 2 - 1)
        kern[kern_size // 2, kern_size // 2] = 0.0
        kern = kern.astype(np.float32)
        stack_kern = np.stack([kern, kern, kern])
        stack_kern = np.expand_dims(stack_kern, 1)
        stack_kern = torch.tensor(stack_kern).to(device)
        return stack_kern, kern_size // 2

    device = next(model.parameters()).device
    images = images.clone().detach().to(device)

    with torch.no_grad():
        labels = model(images).detach()

    images_min = clip_by_tensor(images - max_epsilon, t_min=0, t_max=1)
    images_max = clip_by_tensor(images + max_epsilon, t_min=0, t_max=1)

    if random_start:
        adv_images = images + torch.empty_like(images).uniform_(-max_epsilon, max_epsilon)
        adv_images = torch.clamp(adv_images, min=0, max=1).detach()
    else:
        adv_images = images.clone().detach()

    eps = max_epsilon
    num_iter = num_iter_set
    alpha = eps / num_iter
    alpha_beta = alpha * amplification
    gamma = alpha_beta * project_factor

    P_kern, padding_size = project_kern(3, device)
    T_kern = gaussian_kern(3, 3)
    T_kern = torch.tensor(T_kern).to(device)

    amplification = torch.zeros_like(images).to(device)

    for _ in range(num_iter):
        adv_images.requires_grad = True

        outputs = model(adv_images)
        loss = F.mse_loss(outputs, labels)
        loss.backward()

        noise = adv_images.grad.data
        noise = F.conv2d(noise, T_kern, padding=(padding_size, padding_size), groups=3)

        amplification += alpha_beta * torch.sign(noise)
        cut_noise = torch.clamp(torch.abs(amplification) - eps, 0, 1e10) * torch.sign(amplification)
        projection = gamma * torch.sign(project_noise(cut_noise, P_kern, padding_size))

        adv_images = adv_images + alpha_beta * torch.sign(noise) + projection

        adv_images = clip_by_tensor(adv_images, images_min, images_max)
        adv_images = adv_images.detach().requires_grad_(True)

    return adv_images.detach()

