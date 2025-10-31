import torch
import torch.nn.functional as F


def contrastive_loss(image_features, text_features, temperature=0.07):
    image_features = F.normalize(image_features, dim=1)
    text_features = F.normalize(text_features, dim=1)
    logits = torch.matmul(image_features, text_features.T)
    labels = torch.arange(logits.size(0), device=image_features.device)
    loss_i2t = F.cross_entropy(logits / temperature, labels)
    loss_t2i = F.cross_entropy(logits.T / temperature, labels)
    return (loss_i2t + loss_t2i) / 2


class DWA:
    def __init__(self, num_tasks, T=2.0):
        self.num_tasks = num_tasks
        self.T = T
        self.prev_losses = None
        self.current_losses = None

    def compute_weights(self, current_losses):
        if self.prev_losses is None:
            self.prev_losses = current_losses.detach().clone()
            self.current_losses = current_losses.detach().clone()
            return torch.ones(self.num_tasks, device=current_losses.device) / self.num_tasks

        r = current_losses / (self.prev_losses + 1e-7)
        weights = torch.exp(r / self.T)
        weights = weights / torch.sum(weights)

        self.prev_losses = self.current_losses.clone()
        self.current_losses = current_losses.detach().clone()
        return weights


def gradient_based_dynamic_weighting(losses, shared_params):
    grads = []
    for loss in losses:
        grad = torch.autograd.grad(loss, shared_params, retain_graph=True, create_graph=True)
        grad_flat = torch.cat([g.reshape(-1) for g in grad if g is not None])
        grads.append(grad_flat)
    grads = torch.stack(grads)
    grad_norms = grads.norm(p=2, dim=1)
    avg_norm = grad_norms.mean().detach()
    weights = avg_norm / (grad_norms + 1e-6)
    weights = weights / weights.sum().detach()
    return weights.detach()