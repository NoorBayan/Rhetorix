import torch
from transformers import Trainer
import torch.nn as nn
from .losses import OrdinalCrossEntropyLoss

class OrdinalTrainer(Trainer):
    def __init__(self, class_weights=None, use_ordinal_loss=True, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.class_weights = torch.tensor(class_weights, dtype=torch.float32) if class_weights is not None else None
        self.use_ordinal_loss = use_ordinal_loss

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.pop("labels")
        outputs = model(**inputs)
        logits = outputs.logits
        
        if self.use_ordinal_loss:
            loss_fct = OrdinalCrossEntropyLoss(num_classes=self.model.config.num_labels, weight=self.class_weights)
        else:
            loss_fct = nn.CrossEntropyLoss(weight=self.class_weights.to(self.args.device) if self.class_weights is not None else None)
            
        loss = loss_fct(logits.view(-1, self.model.config.num_labels), labels.view(-1))
        
        return (loss, outputs) if return_outputs else loss
