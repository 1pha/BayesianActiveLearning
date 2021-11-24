import math
import json
import logging
from pathlib import Path
import numpy as np
from scipy.stats import entropy

import torch

logging.basicConfig(
    format="[%(asctime)s] %(levelname)s - %(name)s: %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


class AcquisitionTool:
    def __init__(self, data_args, training_args):

        self.config = training_args
        self.method = training_args.acquisition

        with open(
            Path(f"{data_args.asset_dir}/{training_args.configuration_keys}"), "r"
        ) as f:
            configuration_keys = json.load(f)
        self.name = configuration_keys["acquisition"][self.method]
        self.acquisition = ACQUISITION_MAP[self.method]

    def __call__(self, logits):

        # Will return array of confidence

        return self.acquisition(logits)


def to_2d(logits):

    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.squeeze()
    elif logits.ndim == 3 and logits.shape[1] > 1:
        logger.warn("Single model should be used for uncertainty sampling.")
        raise

    return logits


def random_selection(logits):

    num_pool = logits.shape[1]
    indices = np.arange(num_pool)
    np.random.shuffle(indices)

    return indices


def least_confidence(logits):

    logits = to_2d(logits)

    most_conf = torch.max(logits, dim=1).values
    num_labels = logits.size()[1]
    numerator = num_labels * (1 - most_conf)
    denominator = num_labels - 1
    return numerator / denominator


def margin_of_confidence(logits):

    logits = to_2d(logits)
    sorted_logit = logits.sort(dim=1).values
    margin = 1 - (sorted_logit[:, -1] - sorted_logit[:, -2])
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


ACQUISITION_MAP = {
    "random": random_selection,
    "lc": least_confidence,
    "margin": margin_of_confidence,
    "entropy": entropy,
    "mnlp": "mnlp",
    "bald": bald_acquisition,
    "batchbald": "B-BALD",
}

if __name__ == "__main__":

    from config import parse_arguments
    import numpy as np

    def get_mixture_prob_dist(p1, p2, m):
        return (1.0 - m) * np.asarray(p1) + m * np.asarray(p2)

    data_args, training_args, model_args = parse_arguments()
    acquisition = AcquisitionTool(data_args, training_args)

    K = training_args.num_sampling

    p1 = [0.7, 0.1, 0.1, 0.1, 0.0]
    p2 = [0.3, 0.3, 0.2, 0.2, 0.0]
    y1_ws = [get_mixture_prob_dist(p1, p2, m) for m in np.linspace(0, 1, K)]

    p1 = [0.1, 0.6, 0.1, 0.1, 0.1]
    p2 = [0.2, 0.3, 0.3, 0.2, 0.0]
    y2_ws = [get_mixture_prob_dist(p1, p2, m) for m in np.linspace(0, 1, K)]

    p1 = [0.1, 0.1, 0.5, 0.1, 0.2]
    p2 = [0.2, 0.2, 0.3, 0.3, 0.0]
    y3_ws = [get_mixture_prob_dist(p1, p2, m) for m in np.linspace(0, 1, K)]

    p1 = [0.1, 0.1, 0.05, 0.9, 0.05]
    p2 = [0.3, 0.2, 0.2, 0.3, 0.0]
    y4_ws = [get_mixture_prob_dist(p1, p2, m) for m in np.linspace(0, 1, K)]

    def nested_to_tensor(l):
        return torch.stack(list(map(torch.as_tensor, l)))

    ys_ws = nested_to_tensor(
        [y1_ws, y2_ws, y3_ws, y4_ws]
    )  # (batch, num_models, num_class)

    print(ys_ws)

    print(acquisition(ys_ws))
