import json
import os
import torch

import matplotlib.pyplot as plt
import pandas as pd

from collections import defaultdict
from datetime import datetime
from pathlib import Path

#import seaborn as sns


def show_images(images: list, labels: list[str]=None, n_cols: int=3):
    """
    Show list of images
    
    Parameters
    ----------
    images : list
    
    """
    
    if labels is None:
        labels = [f"pic. {index}" for index in range(1, len(images) + 1)]
    else:
        assert len(images) == len(labels)
        
    n_rows = len(images) // n_cols + (1 if len(images) % n_cols else 0)
    
    fig = plt.figure()
    for index, (image, label) in enumerate(zip(images, labels)):
        plt.subplot(n_rows, n_cols, index+1)
        plt.tight_layout()
        plt.imshow(image[0], cmap='gray', interpolation='none')
        plt.title(label)
        plt.xticks([])
        plt.yticks([])
    
    plt.show();
    
        
def split_lists(list_of_tuples: list()) -> tuple():
    """
    List of tuples to tuple of lists.
    """
    
    assert len(list_of_tuples) > 0
    
    lists = [[] for index in range(len(list_of_tuples[0]))]
    for element in list_of_tuples:
        for index, subelement in enumerate(element):
            lists[index].append(subelement)
            
    return tuple(lists)



def get_outputs(model, dataloader, device) -> list():
    """
    Get outputs of the model.
    """
    
    # Exit training mode.
    was_in_training = model.training
    model.eval()
    
    outputs = []
    with torch.no_grad():
        for index, batch in enumerate(dataloader):
            x, y = batch
            
            outputs.append(model(x.to(device)).detach().cpu())
            
    outputs = torch.cat(outputs)
    
    # Return to the original mode.
    model.train(was_in_training)
    
    return outputs
