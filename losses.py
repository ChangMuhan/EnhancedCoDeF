import torch
import torch.nn as nn

class MSELoss(nn.Module):
    def __init__(self, coef=1):
        super().__init__()
        self.coef = coef
        self.loss = nn.MSELoss(reduction='mean')

    def forward(self, inputs, targets):
        loss = self.loss(inputs, targets)
        return self.coef * loss

def compute_flow_loss(pred_flow, gt_flow):
    return torch.mean(torch.abs(pred_flow - gt_flow))