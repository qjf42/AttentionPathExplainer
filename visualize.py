# coding: utf-8

from collections import defaultdict
from typing import List, Dict
import networkx as nx
import matplotlib.pyplot as plt
from interface import AttentionTensor, PathScores


class Subgraph:
    def __init__(self, center_nodes: List[int], all_path_scores: Dict[int, PathScores]):
        self.center_nodes = center_nodes
        center_path_scores = {node: all_path_scores[node] for node in center_nodes}
        compressed_graph = self._compress(center_path_scores)
        self.g = self._build(compressed_graph)
    
    def _build(self, edge_score_dic):
        g = nx.Graph()
        for (src, tgt), w in edge_score_dic.items():
            g.add_edge(src, tgt, weight=w)
        return g

    def _compress(self, center_path_scores: Dict[int, PathScores]):
        edges = {}
        for path_scores in center_path_scores.values():
            for path, score in path_scores.items():
                for i in range(len(path) - 1):
                    edge = path[i:i+2]
                    edges[edge] = edges.get(edge, 0) + score
        return edges

    def draw(self, center_node_color='red', center_node_size=800, **kwargs):
        default_node_color = kwargs.setdefault('node_color', '#1f78b4'); del kwargs['node_color']
        default_node_size = kwargs.setdefault('node_size', 500); del kwargs['node_size']
        node_colors = [center_node_color if n in self.center_nodes else default_node_color for n in self.g.nodes()]
        node_sizes = [center_node_size if n in self.center_nodes else default_node_size for n in self.g.nodes()]
        edge_colors = [_[-1] for _ in self.g.edges(data='weight')]
        node_labels = {n: n for n in self.g.nodes}

        edge_cmap = kwargs.setdefault('edge_cmap', 'Reds')
        if isinstance(edge_cmap, str):
            kwargs['edge_cmap'] = plt.get_cmap(edge_cmap)
        kwargs.setdefault('font_color', 'white')
        kwargs.setdefault('font_size', 10)
        plt.figure(figsize=kwargs.get('figsize', (10, 7)))
        nx.draw(self.g, pos=nx.spring_layout(self.g),
                labels=node_labels, node_color=node_colors, node_size=node_sizes,
                edge_color=edge_colors, **kwargs)
        plt.show()


class Lattice:
    def __init__(self, tokens: List, attn_tensors: List[AttentionTensor]):
        self.tokens = tokens
        self.num_layers = len(attn_tensors)
        self.g = self._build(attn_tensors)
    
    def _get_node(self, layer, i):
        return f'{layer}_{i}'

    def _build(self, attn_tensors: List[AttentionTensor]):
        g = nx.Graph()
        for layer_idx, (edges, weights) in enumerate(attn_tensors, 1):
            for (src, tgt), w in zip(edges, weights):
                src = self._get_node(layer_idx - 1, src)
                tgt = self._get_node(layer_idx, tgt)
                g.add_edge(src, tgt, weight=w)
        return g

    def draw(self, all_path_scores: Dict[int, PathScores], **kwargs):
        # get edges on paths
        path_edges = defaultdict(int)
        for path_scores in all_path_scores.values():
            for p, score in path_scores.items():
                for offset, (s, t) in enumerate(reversed(list(zip(p[:-1], p[1:])))):
                    src = self._get_node(self.num_layers - offset - 1, s)
                    tgt = self._get_node(self.num_layers - offset, t)
                    path_edges[(src, tgt)] += score
        # set positions for nodes, (node) top to bottom and (layer) left to right
        num_nodes = len({n.split('_')[1] for n in self.g.nodes})
        pos = {}
        for layer_idx in range(self.num_layers + 1):
            pos.update((self._get_node(layer_idx, i), (layer_idx, num_nodes - i)) for i in range(num_nodes))
        # set labels and colors
        node_labels = {n: self.tokens[int(n.split('_')[1])] for n in self.g.nodes}
        plt.figure(figsize=kwargs.get('figsize', (10, 7)))
        # draw
        #edge_colors = [e[-1] for e in self.g.edges(data='weight') if e[:2] not in path_edges]
        #nx.draw(self.g, pos=pos,
        #        labels=node_labels, node_size=800, font_color='white', font_size=10,
        #        edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'))
        edge_cmap = kwargs.setdefault('edge_cmap', 'Reds')
        if isinstance(edge_cmap, str):
            kwargs['edge_cmap'] = plt.get_cmap(edge_cmap)
        kwargs.setdefault('node_color', '#1f78b4')
        kwargs.setdefault('node_size', 800)
        kwargs.setdefault('font_color', 'white')
        kwargs.setdefault('font_size', 10)
        nx.draw(self.g, pos=pos, labels=node_labels,
                edgelist=list(path_edges), edge_color=list(path_edges.values()), **kwargs)
        plt.show()
