'''
Desc: Explainers
Author: qjf42
Date: 2020-06-21
'''

from typing import Tuple, List, Dict, Union
import numpy as np

from interface import EdgeIndices, EdgeWeights, AttentionTensor, Path, PathScores
from explain_utils import viterbi, compress_path_scores
from visualize import Subgraph, Lattice


class ExplainerBase:
    def __init__(self):
        self.all_path_scores = None

    def fit(self, attention_tensors: List[AttentionTensor], topk=1, min_attn_weight=0.01):
        '''Run viterbi, get topk attention paths of all nodes
        Args:
            attention_tensors: [(EdgeIndices, EdgeWeights), ...], attention edges and weights of each layer
            topk: int
            min_attn_weight: float, minimum attention weight
        '''
        self.all_path_scores = viterbi(attention_tensors, topk, min_attn_weight)

    def _explain_single_node(self, node_id: int) -> PathScores:
        assert self.all_path_scores is not None, '`fit` method must be called before `explain`'
        path_scores = self.all_path_scores[node_id]
        return path_scores


class NodeClassificationExplainer(ExplainerBase):
    def explain(self, node_id: int, visualize=True):
        '''Explain single node
        Args:
            node_id: int
            visualize: bool
        Returns:
            path_scores: PathScores
            subgraph: networkx.Graph
        '''
        raw_path_scores = self._explain_single_node(node_id)
        path_scores = compress_path_scores(raw_path_scores)
        subgraph = Subgraph([node_id], {node_id: path_scores})
        if visualize:
            subgraph.draw()
        return raw_path_scores, subgraph.g


class LinkPredictionExplainer(ExplainerBase):
    def explain(self, node_pair: Tuple[int, int], visualize=True):
        '''Explain node pairs
        Args:
            node_pair: (int, int)
            visualize: bool
        Returns:
            path_scores: PathScores
            subgraph: networkx.Graph
        '''
        raw_path_scores = {}
        path_scores = {}
        for node_id in node_pair:
            raw = self._explain_single_node(node_id)
            raw_path_scores[node_id] = raw
            path_scores[node_id] = compress_path_scores(raw)
        subgraph = Subgraph(node_pair, path_scores)
        if visualize:
            subgraph.draw()
        return raw_path_scores, subgraph

    def _find_path(self):
        # TODO
        pass


class TransformerExplainer:
    def __init__(self, seq_len: int):
        super().__init__()
        self.seq_len = seq_len

    def explain(self, query, attention_tensors, topk=4, visualize=True):
        raw_path_scores = viterbi(attention_tensors, topk, min_attn_weight)
        path_scores = compress_path_scores(raw_path_scores)
        lattice = Lattice(len(query), attention_tensors, {node_id: path_scores})
        if visualize:
            lattice.draw()
        return raw_path_scores, lattice
