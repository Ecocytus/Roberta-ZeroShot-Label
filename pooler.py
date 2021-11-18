import torch
from torch import nn

class ClassWisePool(nn.Module):
    def __init__(self, num_maps):
        super(ClassWisePool, self).__init__()
        self.num_maps = num_maps
    
    def forward(self, input):
        batch_size, num_channels, s = input.size()
        num_outputs = int(num_channels / self.num_maps)
        x = input.view(batch_size, num_outputs, self.num_maps, s)
        output = torch.sum(x, 2)
        return output.view(batch_size, num_outputs, s) / self.num_maps


class MaxMinPool1d(nn.Module):
    def __init__(self, kmax=1, kmin=None, alpha=1, beta=2, random_drop=0.1):
        super(MaxMinPool1d, self).__init__()
        self.kmax = kmax
        self.kmin = kmin
        if self.kmin is None:
            self.kmin = self.kmax
        self.alpha = alpha
        self.beta = beta
        self.random_drop = random_drop

    def get_positive_k(self, k, n):
        if k <= 0:
            return 0
        elif k < 1:
            return round(k * n)
        elif k > n:
            return int(n)
        else:
            return int(k)

    def forward(self, input, labels):
        batch_size, num_channels, n = input.size()
        kmax = self.get_positive_k(self.kmax, n)
        kmin = self.get_positive_k(self.kmin, n)
        
        kmax_input, _ = input.topk(dim=2, largest=True, sorted=True, k=kmax)
        kmin_input, _ = input.topk(dim=2, largest=False, sorted=True, k=kmin)

        logits = torch.pow(kmax_input.sigmoid(), exponent=self.beta).detach()
        # random drop
        if self.training:
            mask = torch.zeros_like(logits, device=logits.device).uniform_() > self.random_drop
            indices = torch.ones((logits.size(0) * logits.size(1), logits.size(2)), 
                        device=logits.device).multinomial(num_samples=1).reshape(logits.size(0), logits.size(1), 1)
            # force to have at least one result
            mask.scatter_(-1, indices, torch.ones_like(mask))
            logits = logits * mask
        kmax_weight = logits / logits.sum(2).unsqueeze(2)

        logits = torch.pow((-1 * kmin_input).sigmoid(), exponent=self.beta).detach()
        # random drop
        # mask = torch.zeros_like(logits, device=logits.device).uniform_() > self.random_drop
        # logits = logits * mask
        kmin_weight = logits / logits.sum(2).unsqueeze(2)

        extra_loss = None

        if labels is not None:
            extra_loss = kmin_input.sigmoid().pow(exponent=2).sum(2).div_(kmin).sum()
            # extra_loss += (labels.unsqueeze(2).expand(batch_size, num_channels, kmax) - kmax_input.sigmoid()).pow(exponent=2).sum(2).div_(kmax).sum()
            # extra_loss += (labels.unsqueeze(2).expand(batch_size, num_channels, 1) - kmax_input[:, :, 0].sigmoid()).pow(exponent=2).sum(2).div_(1).sum()

        kmax_input = kmax_input * kmax_weight
        output = kmax_input.sum(2)
        if kmin != 0:
            kmin_input = kmin_input * kmin_weight
            output += self.alpha * kmin_input.sum(2)
        return output, extra_loss