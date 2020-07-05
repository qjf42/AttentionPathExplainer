'''
Desc: Explainers
Author: qjf42
Date: 2020-06-21
'''

from typing import Tuple, List, Dict, Union, Iterable, Optional
import numpy as np

from interface import EdgeIndices, EdgeWeights, AttentionTensor, Path, PathScores
from explain_utils import viterbi, compress_path_scores, get_self_attention_tensor, attention_weights_pooling
from visualize import Subgraph, Lattice


class ExplainerBase:
    def __init__(self):
        self.all_path_scores = None

    def fit(self, attention_tensors: List[AttentionTensor], topk=1, min_attn_weight=0.01, show_progress=True):
        '''Run viterbi, get topk attention paths of all nodes
        Args:
            attention_tensors: [(EdgeIndices, EdgeWeights), ...], attention edges and weights of each layer
            topk: int
            min_attn_weight: float, minimum attention weight
            show_progress: show progress bar
        '''
        self.all_path_scores = viterbi(attention_tensors, topk, min_attn_weight, show_progress)

    def _explain_single_node(self, node_id: int) -> PathScores:
        assert self.all_path_scores is not None, '`fit` method must be called before `explain`'
        path_scores = self.all_path_scores[node_id]
        return path_scores


class NodeClassificationExplainer(ExplainerBase):
    def explain(self, node_id: int, visualize=True, **kwargs):
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
            subgraph.draw(**kwargs)
        return raw_path_scores, subgraph.g


class LinkPredictionExplainer(ExplainerBase):
    def explain(self, node_pair: Tuple[int, int], visualize=True, **kwargs):
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
            subgraph.draw(**kwargs)
        return raw_path_scores, subgraph

    def _find_path(self):
        # TODO
        pass


class BERTExplainer:
    def __init__(self, task_type: Union['seq_cls', 'token_cls'],
                 start_layer_index=0,
                 min_attention_weight=0.01,
                 multi_head_pooling='mean'):
        '''
        Args:
            task_type: seq_cls/token_cls
            start_layer_index
            min_attention_weight
        '''
        assert task_type.lower() in ('seq_cls', 'token_cls')
        self.task_type = task_type.lower()
        self.start_layer_index = start_layer_index
        self.min_attention_weight = min_attention_weight
        self.multi_head_pooling = multi_head_pooling

    def explain(self, tokens: List[str], attention_weights: Iterable[np.ndarray],
                seq_mask: Optional[Iterable[int]] = None, show_sep=False,
                topk=16, last_layer_token_mask: Iterable[int] = None,
                visualize=True, **kwargs):
        '''
        Args:
            tokens: tokens include `CLS` and `SEP`
            attention_weights: raw attention weights, list of array [#tokens, #tokens, #heads]
            seq_mask: same size as tokens
            show_sep: include `SEP` token or not
            topk: topk paths
            last_layer_token_mask: token mask of the last layer,
                                   e.g only [CLS] is used in sequence classfication tasks (default)
                                   if not set, all tokens are considered in token_cls tasks
            visualize: bool
            kwargs: see __init__
        '''
        start_layer_index = kwargs.get('start_layer_index') or self.start_layer_index
        min_attn_weight = kwargs.get('min_attention_weight') or self.min_attention_weight
        multi_head_pooling = kwargs.get('multi_head_pooling') or self.multi_head_pooling
        if not show_sep:
            if seq_mask:
                seq_mask[-1] = 0
                length = None
            else:
                length = len(tokens) - 1
        else:
            length = len(tokens)
        # get attentions
        attn_tensors = []
        for layer_idx, weights in enumerate(attention_weights):
            if layer_idx < start_layer_index:
                continue
            weights = attention_weights_pooling(weights, multi_head_pooling)
            if layer_idx < len(attention_weights) - 1:
                attn_tensors.append(get_self_attention_tensor(weights, seq_mask=seq_mask, length=length))
            else:
                if last_layer_token_mask is None:
                    if self.task_type == 'seq_cls':
                        last_layer_token_mask = np.array([1] + [0] * (len(tokens) - 1))
                    else:
                        last_layer_token_mask = np.ones(len(tokens), dtype=int)
                if not show_sep:
                    last_layer_token_mask[-1] = 0
                attn_tensors.append(get_self_attention_tensor(weights, seq_mask=last_layer_token_mask))

        all_path_scores = viterbi(attn_tensors, topk, min_attn_weight, show_progress=False)

        lattice = Lattice(tokens, attn_tensors)
        if visualize:
            lattice.draw(all_path_scores, **kwargs)
        return all_path_scores, lattice