import torch
import torch.nn as nn
import torch.nn.functional as F

class OrdinalCrossEntropyLoss(nn.Module):
    def __init__(self, num_classes=3, weight=None, penalty_strength=0.5):
        super().__init__()
        self.num_classes = num_classes
        self.weight = weight
        self.penalty_strength = penalty_strength
        self.distance_matrix = torch.abs(
            torch.arange(num_classes).view(-1, 1) - torch.arange(num_classes).view(1, -1)
        ).float()

    def forward(self, logits, targets):
        device = logits.device
        self.distance_matrix = self.distance_matrix.to(device)
        if self.weight is not None:
            self.weight = self.weight.to(device)
            
        ce_loss = F.cross_entropy(logits, targets, weight=self.weight)
        probs = F.softmax(logits, dim=1)
        target_distances = self.distance_matrix[targets]
        ordinal_penalty = torch.mean(torch.sum(probs * target_distances, dim=1))
        
        return ce_loss + (self.penalty_strength * ordinal_penalty)
