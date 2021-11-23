import math
import numpy as np
from scipy.stats import entropy

import torch


def check_torch(logits):

    if isinstance(logits, torch.tensor):
        logits = logits.numpy()
    return logits


def random_selection(logits):

    logits = check_torch(logits)


def least_confidence(logits):

    logits = check_torch(logits)

    most_conf = np.nanmax(logits, axis=1)
    num_labels = logits.shape[1]
    numerator = num_labels * (1 - most_conf)
    denominator = num_labels - 1
    return numerator / denominator


def margin_of_confidence(logits: np.ndarray):

    logits = check_torch(logits)

    part = np.partition(-logits, 1, axis=1)
    margin = -part[:, 0] + part[:, 1]
    return margin


def entropy(logits, dim: int, keepdim: bool = False):
    return -torch.sum((torch.exp(logits) * logits).double(), dim=dim, keepdim=keepdim)


def logit_mean(logits, dim: int, keepdim: bool = False):
    r"""Computes $\log \left ( \frac{1}{n} \sum_i p_i \right ) =
    \log \left ( \frac{1}{n} \sum_i e^{\log p_i} \right )$.
    We pass in logits.
    """
    return torch.logsumexp(logits, dim=dim, keepdim=keepdim) - math.log(
        logits.shape[dim]
    )


def mutual_information(logits_B_K_C):

    """
    logits_b_K_C: (batch, num_models, num_class)
    """

    # Need to calculate
    # Mean Entropy - Entropy of mean among models
    sample_entropies_B_K = entropy(logits_B_K_C, dim=-1)
    entropy_mean_B = torch.mean(sample_entropies_B_K, dim=1)

    logits_mean_B_C = logit_mean(logits_B_K_C, dim=1)
    mean_entropy_B = entropy(logits_mean_B_C, dim=-1)

    mutual_info_B = mean_entropy_B - entropy_mean_B
    return mutual_info_B


def bald_acquisition(logits):

    return mutual_information(logits)
