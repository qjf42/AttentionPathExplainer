'''
Desc: Viterbi algorithm and util functions
Author: qjf42
Date: 2020-06-21
'''

import heapq
from typing import Tuple, List, Dict
from collections import OrderedDict
import numpy as np
from scipy.sparse import coo_matrix, csr_matrix
from tqdm import tqdm

from interface import EdgeIndices, EdgeWeights, AttentionTensor, Path, PathScores


def attention_weights_pooling(attention_weights, pooling_strategy='mean') -> EdgeWeights:
    '''Pooling edge attentions with multi heads
    Args:
        attention_weights: np array, [num_edges(, num_heads)]
        pooling_strategy: str, mean/avg/max
    Returns:
        attention_weights: np array, [num_edges]
    '''
    assert pooling_strategy in ('mean', 'avg', 'max')
    assert attention_weights.ndim <= 2, f'num of dims of attentions ({attention_weights.ndim}) should be no greater than 2'
    if attention_weights.ndim == 1:
        return attention_weights
    elif attention_weights.ndim == 2:
        if pooling_strategy in ('mean', 'avg'):
            return attention_weights.mean(-1)
        else:
            return attention_weights.max(-1)


def filter_weights(indices: EdgeIndices, values: EdgeWeights, min_value=1e-2) -> Tuple[EdgeIndices, csr_matrix]:
    '''Filter edges with small weights to reduce calculations
    Args:
        indices: np array: [num_edges, 2]
        values: np array: [num_edges], attention weights of edges
        min_value: float, minimum attention weight
    Returns:
        filtered_indices: np array: [num_edges, 2]
        filtered_attention_weights: csr_matrix, [num_nodes, num_nodes]
    '''
    assert indices.shape[0] == values.shape[0], \
        f'edges ({indices.shape[0]}) and attention weights ({values.shape[0]}) should have the same length'
    mask = values >= min_value
    indices = indices[mask,:]
    values = values[mask]
    return indices, coo_matrix((values, (indices[:,0], indices[:,1]))).tocsr()


def get_inv_edge_dict(edge_indicies: EdgeIndices) -> Dict[int, List[int]]:
    ret = {}
    for src, tgt in edge_indicies:
        ret.setdefault(tgt, []).append(src)
    return ret


def viterbi(attention_tensors: List[AttentionTensor], topk=1, min_attn_weight=0.01) -> Dict[int, PathScores]:
    '''Find the most important attention paths 
    Args:
        attention_tensors: [(EdgeIndices, csr_matrix), ...], attention edges and weights of each layer
        topk: int
        min_attn_weight: float
    Returns:
        top_scores, top_paths
    '''
    top_scores: Dict[int, List[float]] = {}
    top_paths: Dict[int, List[Path]] = {}
    for edge_indices, attn_weights in attention_tensors:
        # remove less important edges
        edge_indices, attn_matrix = filter_weights(edge_indices, attn_weights, min_attn_weight)
        # viterbi
        new_top_scores: Dict[int, List[float]] = {}
        new_top_paths: Dict[int, List[Path]] = {}
        for tgt, src_list in tqdm(get_inv_edge_dict(edge_indices).items()):
            scores = []
            paths = []
            for src in src_list:
                attn = attn_matrix[src,tgt]
                if src in top_scores:
                    scores += [s * attn for s in top_scores[src]]
                    paths += [tuple(list(p) + [tgt]) for p in top_paths[src]]
                else:
                    scores.append(attn)
                    paths.append((src, tgt))
            topk_pairs = heapq.nlargest(topk, zip(scores, paths))
            new_top_scores[tgt] = [_[0] for _ in topk_pairs]
            new_top_paths[tgt] = [_[1] for _ in topk_pairs]      
        top_scores = new_top_scores
        top_paths = new_top_paths

    # merge paths and scores
    ret: Dict[int, PathScores] = {}
    for tgt, scores in top_scores.items():
        ret[tgt] = OrderedDict(zip(top_paths[tgt], scores))
    return ret


def compress_path_scores(path_scores: PathScores) -> PathScores:
    '''Heuristic way to compress path: take all unique nodes on the path as the id of path, then aggregate scores
    Args:
        path_scores: {(int, ...): float}, path and scores
    Returns:
        merged version
    '''
    def _compress_path(path: Path):
        ret = []
        for node in reversed(path):
            if node not in ret:
                ret.append(node)
        return tuple(ret)

    ret = {}
    unique_path_dic = {}
    for path, score in path_scores.items():
        p = tuple(list(sorted(path)))
        path = unique_path_dic.setdefault(p, _compress_path(path))
        ret[path] = ret.get(path, 0) + score
    return ret
