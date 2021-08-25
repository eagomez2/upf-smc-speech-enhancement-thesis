import os
import argparse
import torch
from typing import List


def get_args_subset(args: argparse.ArgumentParser, 
                    args_subset: List[str]):
    """ Return an argument subset or an ArgumentParser instance. 

    Args:
        args (argparse.ArgumentParser): An instance of ArgumentParser.
        args_subset (List[str]): A list with strings containing the variables
            to be extracted from the full args kwargs list.

    Returns:
        namespace (argparse.Namespace): A new namespace with the extracted
            arguments.
    """
    return argparse.Namespace(**{k: v for k, v in args._get_kwargs() 
        if k in args_subset})

def replace_denormals(x: torch.tensor, threshold=1e-10):
    """ Returns a tensor without denormal values under a certain threshold. 

    IMPORTANT: Please note that this function does not turn the denormal values
    into zeros, but replaces them by a threshold instead, thus preventing 
    tensors with values that may turn into a NaN during backpropagation.

    Args:
        x (torch.tensor): Input tensor.
        threshold (float): Thereshold under or above which values will be set
            to its value to avoid zeros.
    """
    y = x.clone()
    y[(x < threshold) & (x > -1.0 * threshold)] = threshold
    return y
