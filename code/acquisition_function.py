import math
import json
import logging
from pathlib import Path
from dataclasses import dataclass
from typing import List
from functools import partial

import numpy as np
from scipy.stats import entropy
from toma import toma
from tqdm.auto import tqdm
from batchbald_redux import joint_entropy

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

        if self.method != "batchbald":
            return self.acquisition(logits)

        elif self.method == "batchbald":
            return self.acquisition(
                logits,
                num_samples=self.config.increment_num,
                device="cuda" if self.config.use_gpu else "cpu",
            )


def to_2d(logits):

    if logits.ndim == 3 and logits.shape[1] == 1:
        logits = logits.squeeze()
    elif logits.ndim == 3 and logits.shape[1] > 1:
        logger.warn("Single model should be used for uncertainty sampling.")
        raise

    return logits


def random_selection(num_pool):

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

    """tried to use torch.tensor itself, but the operation was bottleneck. Chose to use numpy instead."""

    logits = to_2d(logits).detach().cpu().numpy()
    part = np.partition(-logits, 1, axis=1)
    margin = -part[:, 0] + part[:, 1]
    return torch.tensor(margin)


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


def compute_conditional_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Conditional Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        nats_n_K_C = log_probs_n_K_C * torch.exp(log_probs_n_K_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_K_C, dim=(1, 2)) / K)
        pbar.update(end - start)

    pbar.close()

    return entropies_N


def compute_entropy(log_probs_N_K_C: torch.Tensor) -> torch.Tensor:
    N, K, C = log_probs_N_K_C.shape

    entropies_N = torch.empty(N, dtype=torch.double)

    pbar = tqdm(total=N, desc="Entropy", leave=False)

    @toma.execute.chunked(log_probs_N_K_C, 1024)
    def compute(log_probs_n_K_C, start: int, end: int):
        mean_log_probs_n_C = torch.logsumexp(log_probs_n_K_C, dim=1) - math.log(K)
        nats_n_C = mean_log_probs_n_C * torch.exp(mean_log_probs_n_C)

        entropies_N[start:end].copy_(-torch.sum(nats_n_C, dim=1))
        pbar.update(end - start)

    pbar.close()

    return entropies_N


@dataclass
class CandidateBatch:
    scores: List[float]
    indices: List[int]


def batchbald_batch(
    log_probs_N_K_C: torch.Tensor,
    num_samples: int,
    dtype=None,
    device=None,
) -> CandidateBatch:
    N, K, C = log_probs_N_K_C.shape

    batch_size = N
    # batch_size = min(batch_size, N)

    candidate_indices = []
    candidate_scores = []

    if batch_size == 0:
        return candidate_scores

    conditional_entropies_N = compute_conditional_entropy(log_probs_N_K_C)

    batch_joint_entropy = joint_entropy.DynamicJointEntropy(
        num_samples, batch_size - 1, K, C, dtype=dtype, device=device
    )

    # We always keep these on the CPU.
    scores_N = torch.empty(N, dtype=torch.double, pin_memory=torch.cuda.is_available())

    for i in tqdm(range(batch_size), desc="BatchBALD", leave=False):
        if i > 0:
            latest_index = candidate_indices[-1]
            batch_joint_entropy.add_variables(
                log_probs_N_K_C[latest_index : latest_index + 1]
            )

        shared_conditinal_entropies = conditional_entropies_N[candidate_indices].sum()

        batch_joint_entropy.compute_batch(log_probs_N_K_C, output_entropies_B=scores_N)

        scores_N -= conditional_entropies_N + shared_conditinal_entropies
        scores_N[candidate_indices] = -float("inf")

        candidate_score, candidate_index = scores_N.max(dim=0)

        candidate_indices.append(candidate_index.item())
        candidate_scores.append(candidate_score.item())

    # return CandidateBatch(candidate_scores, candidate_indices)
    return candidate_scores, candidate_indices


ACQUISITION_MAP = {
    "random": random_selection,
    "lc": least_confidence,
    "margin": margin_of_confidence,
    "entropy": entropy,
    "mnlp": "mnlp",
    "bald": bald_acquisition,
    "batchbald": batchbald_batch,
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
