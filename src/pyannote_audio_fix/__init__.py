import itertools
import math
import textwrap
import warnings
import numpy as np
import scipy
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch_state_bridge as tb
from string import ascii_uppercase
from dataclasses import dataclass
from enum import Enum
from functools import cached_property, lru_cache
from pathlib import Path
from typing import Callable, Dict, List, Optional, Text, Tuple, Union
from tqdm import tqdm
from einops import rearrange
from huggingface_hub import hf_hub_download
from scipy.cluster.hierarchy import fcluster, linkage
from scipy.linalg import eigh
from scipy.optimize import linear_sum_assignment
from scipy.spatial.distance import cdist
from scipy.special import logsumexp, softmax
from sklearn.cluster import KMeans
from asteroid_filterbanks import Encoder, ParamSincFB
from safetensors.torch import load_file
from torch.utils.data import Dataset, DataLoader
from torchaudio.compliance.kaldi import fbank as kaldi_fbank
from pyannote.core import (
    Annotation,
    Segment,
    SlidingWindow,
    SlidingWindowFeature,
)

SAMPLE_RATE = 16000
def string_generator():
    r = 1
    while True:
        for c in itertools.product(ascii_uppercase, repeat=r):
            yield ''.join(c)
        r += 1

def VBx(
    X,
    Phi,
    Fa=1.0,
    Fb=1.0,
    pi=10,
    gamma=None,
    maxIters=10,
    epsilon=1e-4,
    alphaQInit=1.0,
    return_model=False,
    alpha=None,
    invL=None,
):
    D = X.shape[1]  # feature (e.g. x-vector) dimensionality

    if type(pi) is int:
        pi = np.ones(pi) / pi

    if gamma is None:
        # initialize gamma from flat Dirichlet prior with
        # concentration parameter alphaQInit
        gamma = np.random.gamma(alphaQInit, size=(X.shape[0], len(pi)))
        gamma = gamma / gamma.sum(1, keepdims=True)

    assert gamma.shape[1] == len(pi) and gamma.shape[0] == X.shape[0]

    G = -0.5 * (
        np.sum(X**2, axis=1, keepdims=True) + D * np.log(2 * np.pi)
    )  # per-frame constant term in (23)
    V = np.sqrt(Phi)  # between (5) and (6)
    rho = X * V  # (18)
    Li = []
    ELBO = None  ##
    for ii in range(maxIters):
        # Do not start with estimating speaker models if those are provided
        # in the argument
        if ii > 0 or alpha is None or invL is None:
            invL = 1.0 / (
                1 + Fa / Fb * gamma.sum(axis=0, keepdims=True).T * Phi
            )  # (17) for all speakers
            alpha = Fa / Fb * invL * gamma.T.dot(rho)  # (16) for all speakers
        log_p_ = Fa * (
            rho.dot(alpha.T) - 0.5 * (invL + alpha**2).dot(Phi) + G
        )  # (23) for all speakers

        # use GMM update
        eps = 1e-8
        lpi = np.log(pi + eps)
        log_p_x = logsumexp(log_p_ + lpi, axis=-1)  # marginal LLH of each data point
        log_pX_ = np.sum(
            log_p_x, axis=0
        )  # total LLH over all data points (to monitor ELBO)

        gamma = np.exp(log_p_ + lpi - log_p_x[:, None])  # responsibilities
        pi = np.sum(gamma, axis=0)

        pi = pi / pi.sum()

        ELBO = log_pX_ + Fb * 0.5 * np.sum(np.log(invL) - invL - alpha**2 + 1)  # (25)
        Li.append([ELBO])

        if ii > 0 and ELBO - Li[-2][0] < epsilon:
            if ELBO - Li[-2][0] < 0:
                print("WARNING: Value of auxiliary function has decreased!")
            break
    return (gamma, pi, Li) + ((alpha, invL) if return_model else ())

def cluster_vbx(ahc_init, fea, Phi, Fa, Fb, maxIters=20, init_smoothing=7.0):
    """ahc_init (T x N_clusters)"""
    qinit = np.zeros((len(ahc_init), ahc_init.max() + 1))
    qinit[range(len(ahc_init)), ahc_init.astype(int)] = 1.0
    qinit = qinit if init_smoothing < 0 else softmax(qinit * init_smoothing, axis=1)
    gamma, pi, _, _, _ = VBx(
        fea,
        Phi,
        Fa=Fa,
        Fb=Fb,
        pi=qinit.shape[1],
        gamma=qinit,
        maxIters=maxIters,
        return_model=True,
    )
    return gamma, pi

class VBxClustering:
    expects_num_clusters = False
    metric = 'cosine'
    constrained_assignment = True
    threshold = 0.6
    Fa = 0.07
    Fb = 0.8

    def __init__(self,plda):
        self.plda = plda

    def __call__(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature | None = None,
        num_clusters: int | None = None,
        min_clusters: int | None = None,
        max_clusters: int | None = None,
    ) -> np.ndarray:
        
        constrained_assignment = self.constrained_assignment

        train_embeddings, _, _ = self.filter_embeddings(
            embeddings, segmentations=segmentations
        )

        if train_embeddings.shape[0] < 2:
            # do NOT apply clustering when the number of training embeddings is less than 2
            num_chunks, num_speakers, _ = embeddings.shape
            hard_clusters = np.zeros((num_chunks, num_speakers), dtype=np.int8)
            soft_clusters = np.ones((num_chunks, num_speakers, 1))
            centroids = np.mean(train_embeddings, axis=0, keepdims=True)
            return hard_clusters, soft_clusters, centroids

        # AHC
        train_embeddings_normed = train_embeddings / np.linalg.norm(
            train_embeddings, axis=1, keepdims=True
        )
        dendrogram = linkage(
            train_embeddings_normed, method="centroid", metric="euclidean"
        )
        ahc_clusters = fcluster(dendrogram, self.threshold, criterion="distance") - 1
        _, ahc_clusters = np.unique(ahc_clusters, return_inverse=True)

        # VBx

        fea = self.plda(train_embeddings)
        q, sp = cluster_vbx(
            ahc_clusters,
            fea,
            self.plda.phi,
            Fa=self.Fa,
            Fb=self.Fb,
            maxIters=20,
        )

        num_chunks, num_speakers, dimension = embeddings.shape
        W = q[:, sp > 1e-7] # responsibilities of speakers that VBx kept
        centroids = W.T @ train_embeddings.reshape(-1, dimension) / W.sum(0, keepdims=True).T

        # (optional) K-Means
        # re-cluster with Kmeans only in case the automatically determined
        # number of clusters does not match the requested number of speakers
        # (either too low, or too high, or different from the requested number)
        auto_num_clusters, _ = centroids.shape
        if auto_num_clusters < min_clusters:
            num_clusters = min_clusters
        elif auto_num_clusters > max_clusters:
            num_clusters = max_clusters
        if num_clusters and num_clusters != auto_num_clusters:
            # disable constrained assignment when forcing number of clusters
            # as it might results in artificially increasing the number of clusters
            constrained_assignment = False
            kmeans_clusters = KMeans(
                n_clusters=num_clusters, n_init=3, random_state=42, copy_x=False
            ).fit_predict(train_embeddings_normed)
            centroids = np.vstack(
                [
                    np.mean(train_embeddings[kmeans_clusters == k], axis=0)
                    for k in range(num_clusters)
                ])

        # calculate distance
        e2k_distance = rearrange(
            cdist(
                rearrange(embeddings, "c s d -> (c s) d"),
                centroids,
                metric=self.metric,
            ),
            "(c s) k -> c s k",
            c=num_chunks,
            s=num_speakers,
        )
        soft_clusters = 2 - e2k_distance

        # assign each embedding to the cluster with the most similar centroid
        if constrained_assignment:
            const = soft_clusters.min() - 1.   # const < any_valid_score
            soft_clusters[segmentations.data.sum(1) == 0] = const
            hard_clusters = self.constrained_argmax(
                soft_clusters,
            )
        else:
            hard_clusters = np.argmax(soft_clusters, axis=2)

        hard_clusters = hard_clusters.reshape(num_chunks, num_speakers)
        
        return hard_clusters, soft_clusters, centroids

    def filter_embeddings(
        self,
        embeddings: np.ndarray,
        segmentations: SlidingWindowFeature,
        min_active_ratio: float = 0.2,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        _, num_frames, _ = segmentations.data.shape
        single_active_mask = (np.sum(segmentations.data, axis=2, keepdims=True) == 1)
        num_clean_frames = np.sum(segmentations.data * single_active_mask, axis=1)
        active = num_clean_frames >= min_active_ratio * num_frames
        valid = ~np.any(np.isnan(embeddings), axis=2)
        chunk_idx, speaker_idx = np.where(active * valid)
        return embeddings[chunk_idx, speaker_idx], chunk_idx, speaker_idx

    def constrained_argmax(self, soft_clusters: np.ndarray) -> np.ndarray:
        soft_clusters = np.nan_to_num(soft_clusters, nan=np.nanmin(soft_clusters))
        num_chunks, num_speakers, num_clusters = soft_clusters.shape
        hard_clusters = -2 * np.ones((num_chunks, num_speakers), dtype=np.int8)
        for c, cost in enumerate(soft_clusters):
            speakers, clusters = linear_sum_assignment(cost, maximize=True)
            for s, k in zip(speakers, clusters):
                hard_clusters[c, s] = k
        return hard_clusters

class Resolution(Enum):
    FRAME = 1  # model outputs a sequence of frames
    CHUNK = 2  # model outputs just one vector for the whole chunk

class Problem(Enum):
    BINARY_CLASSIFICATION = 0
    MONO_LABEL_CLASSIFICATION = 1
    MULTI_LABEL_CLASSIFICATION = 2
    REPRESENTATION = 3
    REGRESSION = 4

@dataclass
class Specifications:
    problem: Problem
    resolution: Resolution
    duration: float
    min_duration: Optional[float] = None
    warm_up: Optional[Tuple[float, float]] = (0.0, 0.0)
    classes: Optional[List[Text]] = None
    powerset_max_classes: Optional[int] = None
    permutation_invariant: bool = False

    @cached_property
    def powerset(self) -> bool:
        if self.powerset_max_classes is None:
            return False

        if self.problem != Problem.MONO_LABEL_CLASSIFICATION:
            raise ValueError(
                "`powerset_max_classes` only makes sense with multi-class classification problems."
            )

        return True

    @cached_property
    def num_powerset_classes(self) -> int:
        return int(
            sum(
                scipy.special.binom(len(self.classes), i)
                for i in range(0, self.powerset_max_classes + 1)
            )
        )
    def __len__(self):
        return 1
    def __iter__(self):
        yield self


class Powerset(nn.Module):
    def __init__(self, num_classes: int, max_set_size: int):
        super().__init__()
        self.num_classes = num_classes
        self.max_set_size = max_set_size
        self.register_buffer("mapping", self.build_mapping(), persistent=False)
        self.register_buffer("cardinality", self.build_cardinality(), persistent=False)


    @cached_property
    def powerset_classes(self) -> list[set[int]]:
        powerset_classes = []
        for set_size in range(0, self.max_set_size + 1):
            for current_set in itertools.combinations(range(self.num_classes), set_size):
                powerset_classes.append(set(current_set))
        return powerset_classes

    @cached_property
    def num_powerset_classes(self) -> int:
        return len(self.powerset_classes)

    def build_mapping(self) -> torch.Tensor:
        mapping = torch.zeros(self.num_powerset_classes, self.num_classes)
        powerset_k = 0
        for set_size in range(0, self.max_set_size + 1):
            for current_set in itertools.combinations(range(self.num_classes), set_size):
                mapping[powerset_k, current_set] = 1
                powerset_k += 1

        return mapping

    def build_cardinality(self) -> torch.Tensor:
        return torch.sum(self.mapping, dim=1)

    def to_multilabel(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        if soft:
            powerset_probs = torch.exp(powerset)
        else:
            powerset_probs = torch.nn.functional.one_hot(
                torch.argmax(powerset, dim=-1),
                self.num_powerset_classes,
            ).float()

        return torch.matmul(powerset_probs, self.mapping)

    def forward(self, powerset: torch.Tensor, soft: bool = False) -> torch.Tensor:
        return self.to_multilabel(powerset, soft=soft)

    def to_powerset(self, multilabel: torch.Tensor) -> torch.Tensor:
        return F.one_hot(
            torch.argmax(torch.matmul(multilabel, self.mapping.T), dim=-1),
            num_classes=self.num_powerset_classes,
        )

    def _permutation_powerset(
        self, multilabel_permutation: Tuple[int, ...]
    ) -> Tuple[int, ...]:
        permutated_mapping: torch.Tensor = self.mapping[:, multilabel_permutation]

        arange = torch.arange(
            self.num_classes, device=self.mapping.device, dtype=torch.int
        )
        powers_of_two = (2**arange).tile((self.num_powerset_classes, 1))

        # compute the encoding of the powerset classes in this 2**N space, before and after
        # permutation of the columns (mapping cols=labels, mapping rows=powerset classes)
        before = torch.sum(self.mapping * powers_of_two, dim=-1)
        after = torch.sum(permutated_mapping * powers_of_two, dim=-1)

        # find before-to-after permutation
        powerset_permutation = (before[None] == after[:, None]).int().argmax(dim=0)

        # return as tuple of indices
        return tuple(powerset_permutation.tolist())

    @cached_property
    def permutation_mapping(self) -> Dict[Tuple[int, ...], Tuple[int, ...]]:
        permutation_mapping = {}

        for multilabel_permutation in itertools.permutations(
            range(self.num_classes), self.num_classes
        ):
            permutation_mapping[
                tuple(multilabel_permutation)
            ] = self._permutation_powerset(multilabel_permutation)

        return permutation_mapping


































# --- Exact Logic Helpers ---
def conv1d_num_frames(num_samples, kernel_size=5, stride=1, padding=0, dilation=1) -> int:
    return 1 + (num_samples + 2 * padding - dilation * (kernel_size - 1) - 1) // stride

def conv1d_receptive_field_size(num_frames=1, kernel_size=5, stride=1, padding=0, dilation=1):
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return effective_kernel_size + (num_frames - 1) * stride - 2 * padding

def conv1d_receptive_field_center(frame=0, kernel_size=5, stride=1, padding=0, dilation=1) -> int:
    effective_kernel_size = 1 + (kernel_size - 1) * dilation
    return frame * stride + (effective_kernel_size - 1) // 2 - padding

def run_multi_conv(val, ks, ss, ps, ds, func, reverse=False):
    params = list(zip(ks, ss, ps, ds))
    if reverse: params = reversed(params)
    for k, s, p, d in params:
        val = func(val, kernel_size=k, stride=s, padding=p, dilation=d)
    return val

# --- SincNet ---
class SincNet(nn.Module):
    def __init__(self, stride: int = 10):
        super().__init__()
        self.stride = stride
        self.wav_norm1d = nn.InstanceNorm1d(1, affine=True)
        self.layers = nn.ModuleDict({
            "conv": nn.ModuleList([
                Encoder(ParamSincFB(80, 251, 10)),
                nn.Conv1d(80, 60, 5, stride=1),
                nn.Conv1d(60, 60, 5, stride=1)
            ]),
            "pool": nn.ModuleList([nn.MaxPool1d(3, stride=3) for _ in range(3)]),
            "norm": nn.ModuleList([nn.InstanceNorm1d(c, affine=True) for c in [80, 60, 60]])
        })

    def _get_params(self):
        return ([251, 3, 5, 3, 5, 3], [self.stride, 3, 1, 3, 1, 3], [0]*6, [1]*6)

    @lru_cache
    def num_frames(self, n: int) -> int:
        return run_multi_conv(n, *self._get_params(), conv1d_num_frames)

    def receptive_field_size(self, n: int = 1) -> int:
        return run_multi_conv(n, *self._get_params(), conv1d_receptive_field_size, reverse=True)

    def receptive_field_center(self, f: int = 0) -> int:
        return run_multi_conv(f, *self._get_params(), conv1d_receptive_field_center, reverse=True)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        x = self.wav_norm1d(waveforms)
        for i, (conv, pool, norm) in enumerate(zip(self.layers["conv"], self.layers["pool"], self.layers["norm"])):
            x = conv(x)
            if i == 0: x = torch.abs(x)
            x = F.leaky_relu(norm(pool(x)))
        return x

# --- PyanNet ---
class PyanNet(nn.Module):
    def __init__(self,specifications:Specifications=None):
        super().__init__()
        self.specifications = specifications
        self.sincnet = SincNet()
        self.lstm = nn.LSTM(60, 128, 4, bidirectional=True, batch_first=True)
        self.linears = nn.ModuleList([nn.Linear(128*2 if i==0 else 128, 128) for i in range(2)])
        self.classifier = nn.Linear(128, self.dimension)

    @property
    def dimension(self) -> int:
        return self.specifications.num_powerset_classes if getattr(self.specifications, 'powerset', False) else len(self.specifications.classes)

    def forward(self, waveforms: torch.Tensor) -> torch.Tensor:
        outputs = rearrange(self.sincnet(waveforms), "batch feature frame -> batch frame feature")
        outputs, _ = self.lstm(outputs)
        for lin in self.linears:
            outputs = F.leaky_relu(lin(outputs))
        return F.log_softmax(self.classifier(outputs), dim=-1)

    def num_frames(self, n): return self.sincnet.num_frames(n)
    def receptive_field_size(self, n=1): return self.sincnet.receptive_field_size(n)
    def receptive_field_center(self, f=0): return self.sincnet.receptive_field_center(f)
    @cached_property
    def receptive_field(self) -> SlidingWindow:
        """(Internal) frames"""

        receptive_field_size = self.receptive_field_size()
        receptive_field_step = (
            self.receptive_field_size(2) - receptive_field_size
        )
        receptive_field_start = (
            self.receptive_field_center() - (receptive_field_size - 1) / 2
        )
        return SlidingWindow(
            start=receptive_field_start / self.sincnet.sample_rate,
            duration=receptive_field_size / self.sincnet.sample_rate,
            step=receptive_field_step / self.sincnet.sample_rate,
        )


class Inference:
    def __init__(
        self,
        model:PyanNet,
        duration: Optional[float] = None,
        step: Optional[float] = None,
        pre_aggregation_hook: Callable[[np.ndarray], np.ndarray] = None,
        skip_aggregation: bool = False,
        skip_conversion: bool = False,
        batch_size: int = 32,
    ):
        self.model = model
        specifications = self.model.specifications
        self.duration = duration
        self.skip_conversion = skip_conversion
        conversion = [Powerset(len(s.classes), s.powerset_max_classes) if s.powerset and not skip_conversion else nn.Identity() for s in specifications]
        if isinstance(specifications, Specifications):
            self.conversion = conversion[0]
        else:
            self.conversion = nn.ModuleList(conversion)
        self.skip_aggregation = skip_aggregation
        self.pre_aggregation_hook = pre_aggregation_hook
        self.warm_up = next(iter(specifications)).warm_up
        step = step or (
            0.1 * self.duration if self.warm_up[0] == 0.0 else self.warm_up[0]
        )
        self.step = step
        self.batch_size = batch_size

    def infer(self, chunks: torch.Tensor) -> Union[np.ndarray, Tuple[np.ndarray]]:
        with torch.inference_mode():
            return self.conversion(self.model(chunks)).cpu().numpy()

    def slide(
        self,
        waveform: torch.Tensor,
        sample_rate: int,
    ) -> Union[SlidingWindowFeature, Tuple[SlidingWindowFeature]]:
        device = next(self.model.parameters()).device
        self.conversion.to(device)
        window_size: int = self.duration * SAMPLE_RATE
        step_size: int = round(self.step * sample_rate)
        _, num_samples = waveform.shape
        specs = self.model.specifications
        frames = self.model.receptive_field
        if num_samples >= window_size:
            chunks: torch.Tensor = rearrange(
                waveform.unfold(1, window_size, step_size),
                "channel chunk frame -> chunk channel frame",
            )
            num_chunks, _, _ = chunks.shape
        else:
            num_chunks = 0
        has_last_chunk = (num_samples < window_size) or (
            num_samples - window_size
        ) % step_size > 0
        if has_last_chunk:
            last_chunk: torch.Tensor = waveform[:, num_chunks * step_size :]
            _, last_window_size = last_chunk.shape
            last_pad = window_size - last_window_size
            last_chunk = F.pad(last_chunk, (0, last_pad))
        outputs = []
        for c in tqdm(range(0, num_chunks, self.batch_size)):
            batch = chunks[c:c+self.batch_size].to(device)
            batch_out = self.infer(batch)
            outputs.append(batch_out)
        if has_last_chunk:
            last_out = self.infer(last_chunk.to(device)[None])
            outputs.append(last_out)
        outputs = np.vstack(outputs)
        if (
            self.skip_aggregation
            or specs.resolution == Resolution.CHUNK
            or (specs.permutation_invariant and self.pre_aggregation_hook is None)
        ):
            frames = SlidingWindow(0.0, self.duration, self.step)
            result = SlidingWindowFeature(outputs, frames)
        else:
            if self.pre_aggregation_hook is not None:
                outputs = self.pre_aggregation_hook(outputs)
            result = self.aggregate(
                SlidingWindowFeature(
                    outputs,
                    SlidingWindow(0.0, self.duration, self.step),
                ),
                frames,
                warm_up=self.warm_up,
                hamming=True,
                missing=0.0,
            )
            if has_last_chunk:
                result.data = result.crop(
                    Segment(0.0, num_samples / sample_rate), mode="loose"
                )
        return result

    @staticmethod
    def aggregate(
        scores: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: Tuple[float, float] = (0.0, 0.0),
        epsilon: float = 1e-12,
        hamming: bool = False,
        missing: float = np.nan,
        skip_average: bool = False,
    ) -> SlidingWindowFeature:
        num_chunks, num_frames_per_chunk, num_classes = scores.data.shape

        chunks = scores.sliding_window
        frames = SlidingWindow(
            start=chunks.start,
            duration=frames.duration,
            step=frames.step,
        )

        # Hamming window used for overlap-add aggregation
        hamming_window = (
            np.hamming(num_frames_per_chunk).reshape(-1, 1)
            if hamming
            else np.ones((num_frames_per_chunk, 1))
        )

        # anything before warm_up_left (and after num_frames_per_chunk - warm_up_right)
        # will not be used in the final aggregation

        # warm-up windows used for overlap-add aggregation
        warm_up_window = np.ones((num_frames_per_chunk, 1))
        # anything before warm_up_left will not contribute to aggregation
        warm_up_left = round(
            warm_up[0] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[:warm_up_left] = epsilon
        # anything after num_frames_per_chunk - warm_up_right either
        warm_up_right = round(
            warm_up[1] / scores.sliding_window.duration * num_frames_per_chunk
        )
        warm_up_window[num_frames_per_chunk - warm_up_right :] = epsilon

        # aggregated_output[i] will be used to store the sum of all predictions
        # for frame #i
        num_frames = (
            frames.closest_frame(
                scores.sliding_window.start
                + scores.sliding_window.duration
                + (num_chunks - 1) * scores.sliding_window.step
                + 0.5 * frames.duration
            )
            + 1
        )
        aggregated_output: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # overlapping_chunk_count[i] will be used to store the number of chunks
        # that contributed to frame #i
        overlapping_chunk_count: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # aggregated_mask[i] will be used to indicate whether
        # at least one non-NAN frame contributed to frame #i
        aggregated_mask: np.ndarray = np.zeros(
            (num_frames, num_classes), dtype=np.float32
        )

        # loop on the scores of sliding chunks
        for chunk, score in scores:
            # chunk ~ Segment
            # score ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            # mask ~ (num_frames_per_chunk, num_classes)-shaped np.ndarray
            mask = 1 - np.isnan(score)
            np.nan_to_num(score, copy=False, nan=0.0)

            start_frame = frames.closest_frame(chunk.start + 0.5 * frames.duration)

            aggregated_output[start_frame : start_frame + num_frames_per_chunk] += (
                score * mask * hamming_window * warm_up_window
            )

            overlapping_chunk_count[
                start_frame : start_frame + num_frames_per_chunk
            ] += mask * hamming_window * warm_up_window

            aggregated_mask[start_frame : start_frame + num_frames_per_chunk] = (
                np.maximum(
                    aggregated_mask[start_frame : start_frame + num_frames_per_chunk],
                    mask,
                )
            )

        if skip_average:
            average = aggregated_output
        else:
            average = aggregated_output / np.maximum(overlapping_chunk_count, epsilon)

        average[aggregated_mask == 0.0] = missing

        return SlidingWindowFeature(average, frames)

    @staticmethod
    def trim(
        scores: SlidingWindowFeature,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:

        assert scores.data.ndim == 3, (
            "Inference.trim expects (num_chunks, num_frames, num_classes)-shaped `scores`"
        )
        _, num_frames, _ = scores.data.shape

        chunks = scores.sliding_window

        num_frames_left = round(num_frames * warm_up[0])
        num_frames_right = round(num_frames * warm_up[1])

        num_frames_step = round(num_frames * chunks.step / chunks.duration)
        if num_frames - num_frames_left - num_frames_right < num_frames_step:
            warnings.warn(
                f"Total `warm_up` is so large ({sum(warm_up) * 100:g}% of each chunk) "
                f"that resulting trimmed scores does not cover a whole step ({chunks.step:g}s)"
            )
        new_data = scores.data[:, num_frames_left : num_frames - num_frames_right]

        new_chunks = SlidingWindow(
            start=chunks.start + warm_up[0] * chunks.duration,
            step=chunks.step,
            duration=(1 - warm_up[0] - warm_up[1]) * chunks.duration,
        )

        return SlidingWindowFeature(new_data, new_chunks)

    def __call__(self, file):
        waveform, sample_rate = self.model.audio(file)
        return self.slide(waveform, sample_rate)


def detect_segments(scores, onset=0.5, offset=None):
    offset = onset if offset is None else offset

    data = scores.data
    timestamps = [scores.sliding_window[i].middle for i in range(len(data))]

    result = Annotation()
    track_gen = string_generator()

    for i, cls_scores in enumerate(data.T):
        label = i if scores.labels is None else scores.labels[i]
        track = next(track_gen)

        active = cls_scores[0] > onset
        start = timestamps[0]

        for t, val in zip(timestamps[1:], cls_scores[1:]):
            if active and val < offset:
                result[Segment(start, t), track] = label
                active = False
            elif not active and val > onset:
                start = t
                active = True

        if active:
            result[Segment(start, timestamps[-1]), track] = label

    return result

def set_num_speakers(
    num_speakers: Optional[int] = None,
    min_speakers: Optional[int] = None,
    max_speakers: Optional[int] = None,
):
    min_speakers = num_speakers or min_speakers or 1
    max_speakers = num_speakers or max_speakers or np.inf
    if min_speakers == max_speakers:
        num_speakers = min_speakers
    return num_speakers, min_speakers, max_speakers

def binarize(scores, onset=0.5, offset=None, initial_state=None):
    offset = offset or onset
    if isinstance(scores, SlidingWindowFeature):
        data = scores.data
        if data.ndim == 2:
            num_frames, num_classes = data.shape
            data = data.T
        elif data.ndim == 3:
            num_chunks, num_frames, num_classes = data.shape
            data = data.reshape(num_chunks * num_classes, num_frames)
        else:
            raise ValueError("Invalid shape")
        result = binarize(data, onset, offset, initial_state)
        if scores.data.ndim == 2:
            result = result.T
        else:
            result = result.reshape(num_chunks, num_frames, num_classes)
        return SlidingWindowFeature(result.astype(float), scores.sliding_window)

    # numpy case
    elif isinstance(scores, np.ndarray):
        batch_size, num_frames = scores.shape
        scores = np.nan_to_num(scores)
        if initial_state is None:
            initial_state = scores[:, 0] >= 0.5 * (onset + offset)
        if isinstance(initial_state, bool):
            initial_state = np.full(batch_size, initial_state)
        on = scores > onset
        off = scores < offset
        state = initial_state.copy()
        result = np.zeros_like(scores, dtype=bool)
        for t in range(num_frames):
            state = np.where(on[:, t], True, np.where(off[:, t], False, state))
            result[:, t] = state
        return result
    else:
        raise TypeError("Unsupported type")

def batchify(iterable, batch_size: int = 32, fillvalue=None):
    return itertools.zip_longest(*[iter(iterable)] * batch_size, fillvalue=fillvalue)

def l2_norm(x):
    n = np.linalg.norm(x, axis=-1, keepdims=True)
    return x / n

def vbx_setup(t_npz, p_npz):
    t, p = np.load(t_npz), np.load(p_npz)
    m1, m2, lda = t["mean1"], t["mean2"], t["lda"]
    mu, tr, psi = p["mu"], p["tr"], p["psi"]
    W = np.linalg.inv(tr.T @ tr)
    B = np.linalg.inv((tr.T / psi) @ tr)
    psi, tr = eigh(B, W)
    psi, tr = psi[::-1], tr.T[::-1]
    xvec_tf = lambda x: np.sqrt(lda.shape[1]) * l2_norm(
        (lda.T @ (np.sqrt(lda.shape[0]) * l2_norm(x - m1)).T).T - m2
    )
    plda_tf = lambda x, lda_dim=lda.shape[1]: ((x - mu) @ tr.T)[:, :lda_dim]
    return xvec_tf, plda_tf, psi


class PLDA:
    def __init__(self, transform_npz: str | Path, plda_npz: str | Path, lda_dimension=128):
        self._xvec_tf, self._plda_tf, self._psi = vbx_setup(transform_npz, plda_npz)
        self.lda_dimension = lda_dimension
    @property
    def phi(self):
        return self._psi[:self.lda_dimension]
    def __call__(self, emb: np.ndarray):
        return self._plda_tf(self._xvec_tf(emb), self.lda_dimension)

class TSTP(nn.Module):
    def __init__(self, in_dim=0):
        super().__init__()
        self.in_dim = in_dim

    def forward(self, features:torch.Tensor, weights=None):
        x = rearrange(features, "b d c f -> b (d c) f")  # (B, D, T)
        if weights is None:
            return torch.cat(
                [x.mean(dim=-1), x.std(dim=-1, correction=1)], dim=-1
            )
        if weights.dim() == 2:
            weights, has_speaker = weights.unsqueeze(1), False  # (B,1,T)
        else:
            has_speaker = True  # (B,S,T)
        if x.size(-1) != weights.size(-1):
            weights = F.interpolate(weights, size=x.size(-1), mode="nearest")
        w = weights.unsqueeze(2)                 # (B,S,1,T)
        x_exp = x.unsqueeze(1)                   # (B,1,D,T)
        v1 = w.sum(dim=-1) + 1e-8                # (B,S,1)
        mean = (x_exp * w).sum(dim=-1) / v1      # (B,S,D)
        dx2 = (x_exp - mean.unsqueeze(-1))**2    # (B,S,D,T)
        v2 = (w**2).sum(dim=-1)                  # (B,S,1)
        var = (dx2 * w).sum(dim=-1) / (v1 - v2 / v1 + 1e-8)  # (B,S,D)
        out = torch.cat([mean, torch.sqrt(var)], dim=2)      # (B,S,2D)
        return out.squeeze(1) if not has_speaker else out

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.shortcut = nn.Identity() if stride == 1 and in_planes == planes else nn.Sequential(
            nn.Conv2d(in_planes, planes, 1, stride, bias=False),
            nn.BatchNorm2d(planes)
        )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        return F.relu(out + self.shortcut(x))

class ResNet(nn.Module):
    def __init__(self):
        super().__init__()
        m, blocks = 32, [3, 4, 6, 3]
        self.in_planes = m

        self.stem = nn.Sequential(
            nn.Conv2d(1, m, 3, 1, 1, bias=False),
            nn.BatchNorm2d(m),
            nn.ReLU()
        )

        self.layer1 = self._make_layer(m,   blocks[0], 1)
        self.layer2 = self._make_layer(m*2, blocks[1], 2)
        self.layer3 = self._make_layer(m*4, blocks[2], 2)
        self.layer4 = self._make_layer(m*8, blocks[3], 2)

        stats_dim = (80 // 8) * m * 8
        self.pool = TSTP(stats_dim)
        self.fc = nn.Linear(stats_dim * 2, 256)

    def _make_layer(self, planes, n, stride):
        layers = [BasicBlock(self.in_planes, planes, stride)]
        self.in_planes = planes
        layers += [BasicBlock(planes, planes) for _ in range(n - 1)]
        return nn.Sequential(*layers)

    def forward(self, x, weights=None):
        x = x.permute(0, 2, 1).unsqueeze(1)
        x = self.stem(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        return self.fc(self.pool(x, weights=weights))

class WeSpeakerResNet34(nn.Module):
    def __init__(self):
        super().__init__()
        self.resnet = ResNet()

    def forward(self, wav, weights=None):
        wav = wav * (1 << 15)
        feat = torch.vmap(kaldi_fbank)(
            wav.unsqueeze(1),
            num_mel_bins=80,
            window_type="hamming"
        ).to(wav.device)

        feat = feat - feat.mean(dim=1, keepdim=True)
        return self.resnet(feat, weights=weights)

























































class EmbeddingDataset(Dataset):
    def __init__(
        self,
        wav,
        binary_segmentations: SlidingWindowFeature,
        clean_segmentations: SlidingWindowFeature,
        min_num_frames: int = -1,
    ):
        self.min_num_frames = min_num_frames
        self.sliding_window = binary_segmentations.sliding_window
        self.seg_data = binary_segmentations.data
        self.clean_data = clean_segmentations.data

        num_chunks, _, num_speakers = self.seg_data.shape
        self.indices = [
            (c, s)
            for c in range(num_chunks)
            for s in range(num_speakers)
        ]

        self.waveform = wav
        self.window_size = int(self.sliding_window.duration) * SAMPLE_RATE

    def __len__(self):
        return len(self.indices)

    def __getitem__(self, idx):
        chunk_idx, speaker_idx = self.indices[idx]

        # Disk I/O nahi — seedha RAM se slice!
        start_sample = round(
            self.sliding_window[chunk_idx].start * SAMPLE_RATE
        )
        end_sample = start_sample + self.window_size
        
        # Pad if needed (last chunk)
        waveform = self.waveform[:, start_sample:end_sample]  # (1, window_size)
        if waveform.shape[1] < self.window_size:
            waveform = F.pad(waveform, (0, self.window_size - waveform.shape[1]))

        mask = self.seg_data[chunk_idx, :, speaker_idx]
        clean_mask = self.clean_data[chunk_idx, :, speaker_idx]

        mask = np.nan_to_num(mask, nan=0.0).astype(np.float32)
        clean_mask = np.nan_to_num(clean_mask, nan=0.0).astype(np.float32)

        used_mask = (
            clean_mask
            if np.sum(clean_mask) > self.min_num_frames
            else mask
        )

        return (
            waveform.squeeze(0),             # (window_size,)
            torch.from_numpy(used_mask),     # (num_frames,)
            chunk_idx,
            speaker_idx,
        )

class SpeakerDiarization(nn.Module):
    def __init__(
        self,
        segmentation_step: float = 0.1,
        embedding_exclude_overlap: bool = True,
        embedding_batch_size: int = 32,
        der_variant: Optional[dict] = None,
    ):
        super().__init__()
        self._segmentation_model = PyanNet(specifications=Specifications(None,None, 10.0,classes=['speaker#1', 'speaker#2', 'speaker#3'], powerset_max_classes=2, permutation_invariant=True))
        self._embedding = WeSpeakerResNet34()
        self.segmentation_step = segmentation_step
        self.embedding_batch_size = embedding_batch_size
        self.embedding_exclude_overlap = embedding_exclude_overlap
        self._plda = PLDA(hf_hub_download("pyannote-community/speaker-diarization-community-1","plda/xvec_transform.npz"),hf_hub_download("pyannote-community/speaker-diarization-community-1","plda/plda.npz"))
        self.der_variant = der_variant or {"collar": 0.0, "skip_overlap": False}
        segmentation_duration = self._segmentation_model.specifications.duration
        self._segmentation = Inference(
            self._segmentation_model,
            duration=segmentation_duration,
            step=self.segmentation_step * segmentation_duration,
            skip_aggregation=True,
        )

        self.clustering = VBxClustering(self._plda)
        self._expects_num_speakers = self.clustering.expects_num_clusters

        self._segmentation_model.load_state_dict(tb.state_bridge(load_file(hf_hub_download("shethjenil/speaker-diarization","segmentation.safetensors")),"""
linear,linears
conv1d,layers.conv
.norm1d,.layers.norm
"""))
        self._embedding.load_state_dict(load_file(hf_hub_download("shethjenil/speaker-diarization","embedding.safetensors")))
        self.eval()
    @property
    def segmentation_batch_size(self) -> int:
        return self._segmentation.batch_size

    @segmentation_batch_size.setter
    def segmentation_batch_size(self, batch_size: int):
        self._segmentation.batch_size = batch_size

    def classes(self):
        speaker = 0
        while True:
            yield f"SPEAKER_{speaker:02d}"
            speaker += 1

    @cached_property
    def _embedding_min_num_samples(self) -> int:
        with torch.inference_mode():
            device = next(self.parameters()).device
            lower, upper = 2, round(0.5 * SAMPLE_RATE)
            middle = (lower + upper) // 2
            while lower + 1 < upper:
                try:
                    _ = self._embedding(torch.randn(1, 1, middle).to(device))
                    upper = middle
                except Exception:
                    lower = middle
                middle = (lower + upper) // 2
        return upper


    def get_embeddings(
        self,
        wav,
        binary_segmentations: SlidingWindowFeature,
        exclude_overlap: bool = False,
    ):
        device = next(self.parameters()).device
        num_chunks, num_frames, num_speakers = binary_segmentations.data.shape

        # --- Clean segmentations ---
        if exclude_overlap:
            min_num_samples = self._embedding_min_num_samples
            duration = binary_segmentations.sliding_window.duration
            num_samples = duration * SAMPLE_RATE
            min_num_frames = math.ceil(num_frames * min_num_samples / num_samples)
            clean_frames = 1.0 * (
                np.sum(binary_segmentations.data, axis=2, keepdims=True) < 2
            )
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data * clean_frames,
                binary_segmentations.sliding_window,
            )
        else:
            min_num_frames = -1
            clean_segmentations = SlidingWindowFeature(
                binary_segmentations.data.copy(),
                binary_segmentations.sliding_window,
            )

        # --- Dataset + DataLoader ---
        dataset = EmbeddingDataset(
            wav=wav,
            binary_segmentations=binary_segmentations,
            clean_segmentations=clean_segmentations,
            min_num_frames=min_num_frames,
        )

        loader = DataLoader(
            dataset,
            batch_size=self.embedding_batch_size,
            pin_memory=True,
        )
        embeddings = np.zeros(
            (num_chunks, num_speakers, self._embedding.resnet.fc.out_features),
            dtype=np.float32,
        )

        for waveforms, masks, chunk_idxs, speaker_idxs in tqdm(
            loader,
            desc="Embedding",
            total=len(loader),
        ):
            waveforms = waveforms.to(device)  # (B, N) — sahi shape
            masks = masks.to(device)                        # (B, num_frames)

            batch_embeddings = self._embedding(waveforms, masks)
            batch_embeddings = batch_embeddings.cpu().numpy()
            embeddings[chunk_idxs.numpy(), speaker_idxs.numpy()] = batch_embeddings


        return embeddings  # (num_chunks, num_speakers, dimension) — rearrange bhi nahi chahiye!


    def reconstruct(
        self,
        segmentations: SlidingWindowFeature,
        hard_clusters: np.ndarray,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        num_chunks, num_frames, _ = segmentations.data.shape
        num_clusters = np.max(hard_clusters) + 1
        clustered_segmentations = np.nan * np.zeros(
            (num_chunks, num_frames, num_clusters)
        )
        for c, (cluster, (_, segmentation)) in enumerate(
            zip(hard_clusters, segmentations)
        ):
            for k in np.unique(cluster):
                if k == -2:
                    continue
                clustered_segmentations[c, :, k] = np.max(
                    segmentation[:, cluster == k], axis=1
                )
        clustered_segmentations = SlidingWindowFeature(
            clustered_segmentations, segmentations.sliding_window
        )
        return self.to_diarization(clustered_segmentations, count)

    @staticmethod
    def to_diarization(
        segmentations: SlidingWindowFeature,
        count: SlidingWindowFeature,
    ) -> SlidingWindowFeature:
        activations = Inference.aggregate(
            segmentations,
            count.sliding_window,
            hamming=False,
            missing=0.0,
            skip_average=True,
        )
        _, num_speakers = activations.data.shape
        max_speakers_per_frame = np.max(count.data)
        if num_speakers < max_speakers_per_frame:
            activations.data = np.pad(
                activations.data, ((0, 0), (0, max_speakers_per_frame - num_speakers))
            )
        extent = activations.extent & count.extent
        activations = activations.crop(extent, return_data=False)
        count = count.crop(extent, return_data=False)
        sorted_speakers = np.argsort(-activations, axis=-1)
        binary = np.zeros_like(activations.data)
        for t, ((_, c), speakers) in enumerate(zip(count, sorted_speakers)):
            for i in range(c.item()):
                binary[t, speakers[i]] = 1.0
        return SlidingWindowFeature(binary, activations.sliding_window)

    @staticmethod
    def speaker_count(
        binarized_segmentations: SlidingWindowFeature,
        frames: SlidingWindow,
        warm_up: Tuple[float, float] = (0.1, 0.1),
    ) -> SlidingWindowFeature:
        trimmed = Inference.trim(binarized_segmentations, warm_up=warm_up)
        count = Inference.aggregate(
            np.sum(trimmed, axis=-1, keepdims=True),
            frames,
            hamming=False,
            missing=0.0,
            skip_average=False,
        )
        count.data = np.rint(count.data).astype(np.uint8)
        return count

    @torch.inference_mode()
    def forward(
        self,
        wav,
        num_speakers: Optional[int] = None,
        min_speakers: Optional[int] = None,
        max_speakers: Optional[int] = None,
    ):
        num_speakers, min_speakers, max_speakers = set_num_speakers(
            num_speakers=num_speakers,
            min_speakers=min_speakers,
            max_speakers=max_speakers,
        )
        segmentations = self._segmentation(wav)
        if self._segmentation_model.specifications.powerset:
            binarized_segmentations = segmentations
        else:
            binarized_segmentations: SlidingWindowFeature = binarize(
                segmentations,
                initial_state=False,
            )
        count = self.speaker_count(
            binarized_segmentations,
            self._segmentation_model.receptive_field,
            warm_up=(0.0, 0.0),
        )
        if np.nanmax(count.data) == 0.0:
            return

        embeddings = self.get_embeddings(
            wav,
            binarized_segmentations,
            exclude_overlap=self.embedding_exclude_overlap
        )
        hard_clusters, _, centroids = self.clustering(
            embeddings=embeddings,
            segmentations=binarized_segmentations,
            num_clusters=num_speakers,
            min_clusters=min_speakers,
            max_clusters=max_speakers,
        )
        num_different_speakers = np.max(hard_clusters) + 1
        if (
            num_different_speakers < min_speakers
            or num_different_speakers > max_speakers
        ):
            warnings.warn(
                textwrap.dedent(
                    f"""
                The detected number of speakers ({num_different_speakers}) for {file["uri"]} is outside
                the given bounds [{min_speakers}, {max_speakers}]. This can happen if the
                given audio file is too short to contain {min_speakers} or more speakers.
                Try to lower the desired minimal number of speakers.
                """
                )
            )
        count.data = np.minimum(count.data, max_speakers).astype(np.int8)
        inactive_speakers = np.sum(binarized_segmentations.data, axis=1) == 0
        hard_clusters[inactive_speakers] = -2
        discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        diarization = detect_segments(
            discrete_diarization,
        )
        count.data = np.minimum(count.data, 1).astype(np.int8)
        exclusive_discrete_diarization = self.reconstruct(
            segmentations,
            hard_clusters,
            count,
        )
        exclusive_diarization = detect_segments(exclusive_discrete_diarization)
        mapping = {
            label: expected_label
            for label, expected_label in zip(diarization.labels(), self.classes())
        }
        diarization = diarization.rename_labels(mapping=mapping)
        exclusive_diarization = exclusive_diarization.rename_labels(mapping=mapping)
        if centroids is None:
            return {
            "diarization":[{"start":i.start,"end":i.end,"speaker":int(s.replace("SPEAKER_",""))} for i,_,s in diarization.itertracks(True)],
            "exclusive_diarization":[{"start":i.start,"end":i.end,"speaker":int(s.replace("SPEAKER_",""))} for i,_,s in exclusive_diarization.itertracks(True)],
            }
        if len(diarization.labels()) > centroids.shape[0]:
            centroids = np.pad(
                centroids, ((0, len(diarization.labels()) - centroids.shape[0]), (0, 0))
            )
        inverse_mapping = {label: index for index, label in mapping.items()}
        centroids = centroids[
            [inverse_mapping[label] for label in diarization.labels()]
        ]
        
        return {
            "diarization":[{"start":i.start,"end":i.end,"speaker":int(s.replace("SPEAKER_",""))} for i,_,s in diarization.itertracks(True)],
            "exclusive_diarization":[{"start":i.start,"end":i.end,"speaker":int(s.replace("SPEAKER_",""))} for i,_,s in exclusive_diarization.itertracks(True)],
            "speaker_embeddings":centroids
        }

