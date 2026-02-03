"""Learning rate scheduling and training utilities."""

import torch
import math
from torch.optim.lr_scheduler import (
    CosineAnnealingLR,
    StepLR,
    PolynomialLR,
    LinearLR,
)


class WarmupCosineAnnealingLR:
    """Cosine annealing with linear warmup.
    
    Learning rate schedule:
    1. Linear warmup from 0 to initial_lr over warmup_epochs
    2. Cosine annealing decay from initial_lr to min_lr over remaining epochs
    """
    
    def __init__(self, optimizer, max_epochs, warmup_epochs=5, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.warmup_epochs = warmup_epochs
        self.min_lr = min_lr
        self.current_epoch = 0
        self.base_lr = optimizer.defaults['lr']
    
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        
        if self.current_epoch <= self.warmup_epochs:
            # Linear warmup
            lr = self.base_lr * (self.current_epoch / self.warmup_epochs)
        else:
            # Cosine annealing
            progress = (self.current_epoch - self.warmup_epochs) / (
                self.max_epochs - self.warmup_epochs
            )
            lr = self.min_lr + (self.base_lr - self.min_lr) * 0.5 * (
                1 + math.cos(math.pi * progress)
            )
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class PolynomialDecayLR:
    """Polynomial decay learning rate schedule.
    
    lr = (initial_lr - min_lr) * (1 - epoch / max_epochs)^power + min_lr
    """
    
    def __init__(self, optimizer, max_epochs, power=1.0, min_lr=1e-6):
        self.optimizer = optimizer
        self.max_epochs = max_epochs
        self.power = power
        self.min_lr = min_lr
        self.current_epoch = 0
        self.base_lr = optimizer.defaults['lr']
    
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        progress = self.current_epoch / self.max_epochs
        lr = (self.base_lr - self.min_lr) * (1 - progress) ** self.power + self.min_lr
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


class StepDecayLR:
    """Step decay learning rate schedule.
    
    Reduces LR by a factor every step_size epochs.
    """
    
    def __init__(self, optimizer, step_size=20, gamma=0.5):
        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.current_epoch = 0
        self.base_lr = optimizer.defaults['lr']
    
    def step(self):
        """Update learning rate."""
        self.current_epoch += 1
        num_steps = self.current_epoch // self.step_size
        lr = self.base_lr * (self.gamma ** num_steps)
        
        for param_group in self.optimizer.param_groups:
            param_group['lr'] = lr
    
    def get_last_lr(self):
        return [group['lr'] for group in self.optimizer.param_groups]


def get_scheduler(optimizer, max_epochs, schedule_type='warmup_cosine', **kwargs):
    """Factory function to get learning rate scheduler.
    
    Args:
        optimizer: Optimizer instance
        max_epochs: Total number of epochs
        schedule_type: 'warmup_cosine', 'polynomial', 'step', 'cosine'
        **kwargs: Additional arguments for specific schedulers
    
    Returns:
        Scheduler instance with step() method
    """
    if schedule_type == 'warmup_cosine':
        warmup_epochs = kwargs.get('warmup_epochs', 5)
        min_lr = kwargs.get('min_lr', 1e-6)
        return WarmupCosineAnnealingLR(optimizer, max_epochs, warmup_epochs, min_lr)
    
    elif schedule_type == 'polynomial':
        power = kwargs.get('power', 1.0)
        min_lr = kwargs.get('min_lr', 1e-6)
        return PolynomialDecayLR(optimizer, max_epochs, power, min_lr)
    
    elif schedule_type == 'step':
        step_size = kwargs.get('step_size', 20)
        gamma = kwargs.get('gamma', 0.5)
        return StepDecayLR(optimizer, step_size, gamma)
    
    elif schedule_type == 'cosine':
        return CosineAnnealingLR(
            optimizer,
            T_max=max_epochs,
            eta_min=kwargs.get('min_lr', 1e-6)
        )
    
    else:
        raise ValueError(f"Unknown scheduler type: {schedule_type}")


# Recommended training configurations
TRAINING_CONFIGS = {
    'baseline_brats': {
        'batch_size': 2,           # Can afford larger batch with single model
        'learning_rate': 5e-4,
        'max_epochs': 50,
        'warmup_epochs': 5,
        'weight_decay': 1e-5,
        'scheduler_type': 'warmup_cosine',
    },
    'lsnet_brats': {
        'batch_size': 1,           # LSNet is heavier, keep batch=1
        'learning_rate': 1e-4,     # Lower LR for two-branch model
        'max_epochs': 50,
        'warmup_epochs': 10,       # Longer warmup for complexity
        'weight_decay': 1e-5,
        'scheduler_type': 'warmup_cosine',
    },
}
