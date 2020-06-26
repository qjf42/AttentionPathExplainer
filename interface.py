# coding: utf-8

from typing import Tuple, Dict
import numpy as np

# Graph and attention
EdgeIndices = np.ndarray    # [num_edges, 2]
EdgeWeights = np.ndarray    # [num_edges]
AttentionTensor = Tuple[EdgeIndices, EdgeWeights]   # attentions for single layer

# Attention paths and scores
Path = Tuple[int, ...]      # sequence of node ids
PathScores = Dict[Path, float]    # paths and scores
