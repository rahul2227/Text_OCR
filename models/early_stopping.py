import os

import numpy as np
import torch

from utils.utils import get_project_root


class EarlyStopping:
    """
    Early stops the training if training loss doesn't improve after a given patience.
    """

    def __init__(self, patience=3, verbose=False, delta=0.0, path='checkpoint.pt'):
        """
        Args:
            patience (int): How long to wait after last time training loss improved.
            verbose (bool): If True, prints a message for each training loss improvement.
            delta (float): Minimum change in the monitored metric to qualify as an improvement.
            path (str): Path for the checkpoint to be saved to.
        """
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.path = path

        self.counter = 0
        self.best_score = None
        self.early_stop = False
        self.best_loss = np.inf

    def __call__(self, train_loss, model):
        """
        Call method to check if training should stop.

        Args:
            train_loss (float): Current epoch's training loss.
            model (torch.nn.Module): The model being trained.
        """
        score = -train_loss  # Since we want to minimize training loss

        if self.best_score is None:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
        elif score < self.best_score + self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = score
            self.save_checkpoint(train_loss, model)
            self.counter = 0

    def save_checkpoint(self, train_loss, model):
        """
        Saves model when training loss decreases.

        Args:
            train_loss (float): Current epoch's training loss.
            model (torch.nn.Module): The model being trained.
        """
        if self.verbose:
            print(f'Training loss decreased ({self.best_loss:.6f} --> {train_loss:.6f}).  Saving model ...')

        # Ensure the directory exists
        models_dir = os.path.join(get_project_root(), 'models')
        os.makedirs(models_dir, exist_ok=True)

        model_save_path = os.path.join(models_dir, self.path)
        torch.save(model.state_dict(), model_save_path)
        self.best_loss = train_loss