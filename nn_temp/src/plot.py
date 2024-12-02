# -*- coding: utf-8 -*-
"""
Created on Fri 29 15:46:32 2022

Visualization module for plotting learning curves.

This script includes:
- A function for visualizing training and validation loss over epochs.

@author: tadahaya
"""
import matplotlib.pyplot as plt


def plot_curve(
    num_epochs: int, 
    train_loss_list: list, 
    valid_loss_list: list, 
    dir_name: str, 
    file_name: str = "curve.png"
) -> None:
    """
    Plots the learning curve for training and validation loss over epochs and saves the plot.

    Args:
        num_epochs (int): Total number of epochs.
        train_loss_list (list): List of training loss values, one for each epoch.
        valid_loss_list (list): List of validation loss values, one for each epoch.
        dir_name (str): Directory where the plot will be saved.
        file_name (str, optional): Name of the saved file. Defaults to 'curve.png'.

    Returns:
        None
    """
    # Set up the figure and axis
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # Configure font size for readability
    plt.rcParams['font.size'] = 18
    
    # Plot the training and validation loss curves
    ax.plot(range(num_epochs), train_loss_list, color='purple', label='Train Loss')
    ax.plot(range(num_epochs), valid_loss_list, color='orange', label='Validation Loss')
    
    # Label the axes and add a title
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss')
    ax.set_title('Training and Validation Loss')
    
    # Hide the top and right spines for a cleaner look
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    
    # Add a grid for better readability
    ax.grid()
    
    # Add a legend to distinguish the curves
    ax.legend()
    
    # Save the plot to the specified directory with the specified file name
    plt.savefig(f"{dir_name}/{file_name}", bbox_inches="tight")
    
    # Clear the figure to free memory
    plt.close(fig)