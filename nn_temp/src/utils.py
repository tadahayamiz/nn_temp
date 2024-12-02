# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

utilities

@author: tadahaya
"""
import datetime
import numpy as np
import pandas as pd
import random
import torch
import logging
import matplotlib.pyplot as plt
from sklearn import metrics
from torchinfo import summary

# Seed fixing for reproducibility
def fix_seed(seed: int, fix_gpu: bool = False) -> None:
    """
    Fix the random seed for reproducibility.
    
    Args:
        seed (int): Random seed value.
        fix_gpu (bool): Whether to fix GPU-related randomness.
    """
    random.seed(seed)
    torch.manual_seed(seed)
    np.random.seed(seed)
    if fix_gpu:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

# Logger initialization
def init_logger(
    module_name: str, 
    outdir: str = '', 
    tag: str = '', 
    level_console: str = 'warning', 
    level_file: str = 'info'
) -> logging.Logger:
    """
    Initialize a logger with console and file handlers.
    
    Args:
        module_name (str): Name of the module using the logger.
        outdir (str): Directory to save the log file.
        tag (str): Tag for the log file name.
        level_console (str): Logging level for console output.
        level_file (str): Logging level for file output.
    
    Returns:
        logging.Logger: Configured logger object.
    """
    level_dic = {
        'critical': logging.CRITICAL,
        'error': logging.ERROR,
        'warning': logging.WARNING,
        'info': logging.INFO,
        'debug': logging.DEBUG,
        'notset': logging.NOTSET,
    }
    if not tag:
        tag = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    logging.basicConfig(
        level=level_dic[level_file],
        filename=f'{outdir}/log_{tag}.txt',
        format='[%(asctime)s] [%(levelname)s] %(message)s',
        datefmt='%Y%m%d-%H%M%S',
    )
    logger = logging.getLogger(module_name)
    sh = logging.StreamHandler()
    sh.setLevel(level_dic[level_console])
    fmt = logging.Formatter(
        "[%(asctime)s] [%(levelname)s] %(message)s",
        "%Y%m%d-%H%M%S"
    )
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


# Add information to logger
def to_logger(
    logger: logging.Logger, 
    name: str = '', 
    obj: object = None, 
    skip_keys: set = set(), 
    skip_hidden: bool = True
) -> None:
    """
    Log instance information.
    
    Args:
        logger (logging.Logger): Logger object.
        name (str): Name of the object being logged.
        obj (object): Object whose attributes will be logged.
        skip_keys (set): Keys to skip when logging.
        skip_hidden (bool): Whether to skip private attributes.
    """
    logger.info(name)
    for k, v in vars(obj).items():
        if k not in skip_keys:
            if skip_hidden and not k.startswith('_'):
                logger.info(f'  {k}: {v}')
            elif not skip_hidden:
                logger.info(f'  {k}: {v}')


# Save and summarize model
def summarize_model(model: torch.nn.Module, input_tensor: torch.Tensor, outdir: str) -> None:
    """
    Summarize and save a PyTorch model.
    
    Args:
        model (torch.nn.Module): PyTorch model.
        input_tensor (torch.Tensor): Input tensor for the model.
        outdir (str): Directory to save the summary and model weights.
    """
    try:
        with open(f'{outdir}/summary.txt', 'w') as writer:
            writer.write(repr(summary(model, input_tensor.size())))
    except ModuleNotFoundError:
        print('!! CAUTION: torchinfo not installed. Model summary was not saved !!')
    torch.save(model.state_dict(), f'{outdir}/model.pt')


# Plot learning progress
def plot_progress(train_loss: list, test_loss: list, num_epoch: int, outdir: str) -> None:
    """
    Plot learning progress of training and test loss.
    
    Args:
        train_loss (list): List of training loss values.
        test_loss (list): List of test loss values.
        num_epoch (int): Number of epochs.
        outdir (str): Directory to save the plot.
    """
    epochs = list(range(1, num_epoch + 1))
    fig, ax = plt.subplots()
    plt.rcParams['font.size'] = 18
    ax.plot(epochs, train_loss, c='purple', label='Train Loss')
    ax.plot(epochs, test_loss, c='orange', label='Test Loss')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.grid()
    ax.legend()
    plt.tight_layout()
    plt.savefig(f'{outdir}/progress.tif', dpi=100, bbox_inches='tight')
    plt.close(fig)


# Plot accuracy metrics
def plot_accuracy(scores: list, labels: list, outdir: str) -> tuple:
    """
    Plot ROC and PR curves and save them along with prediction results.
    
    Args:
        scores (list): Model prediction scores.
        labels (list): Ground truth labels.
        outdir (str): Directory to save the plots and results.
    
    Returns:
        tuple: AUROC and AUPR values.
    """
    fpr, tpr, _ = metrics.roc_curve(labels, scores)
    auroc = metrics.auc(fpr, tpr)
    precision, recall, _ = metrics.precision_recall_curve(labels, scores)
    aupr = metrics.auc(recall, precision)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 6), tight_layout=True)
    plt.rcParams['font.size'] = 18
    
    # ROC curve
    axes[0].plot(fpr, tpr, c='purple')
    axes[0].set_title(f'ROC Curve (AUC: {auroc:.3f})')
    axes[0].set_xlabel('False Positive Rate (FPR)')
    axes[0].set_ylabel('True Positive Rate (TPR)')
    axes[0].grid()
    
    # PR curve
    axes[1].plot(recall, precision, c='orange')
    axes[1].set_title(f'PR Curve (AUC: {aupr:.3f})')
    axes[1].set_xlabel('Recall')
    axes[1].set_ylabel('Precision')
    axes[1].grid()
    
    plt.savefig(f'{outdir}/accuracy.tif', dpi=100, bbox_inches='tight')
    plt.close(fig)
    
    # Save predictions
    df = pd.DataFrame({'Labels': labels, 'Predictions': scores})
    df.to_csv(f'{outdir}/predicted.txt', sep='\t', index=False)
    
    return auroc, aupr