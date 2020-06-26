# coding: utf-8

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
        for node, path_scores in center_path_scores.items():
            for path, score in path_scores.items():
                for i in range(len(path) - 1):
                    edge = path[i:i+2]
                    edges[edge] = edges.get(edge, 0) + score
        return edges

    def draw(self):
        node_colors = ['red' if n in self.center_nodes else '#1f78b4' for n in self.g.nodes()]
        node_sizes = [800 if n in self.center_nodes else 500 for n in self.g.nodes()]
        edge_colors = [_[-1] for _ in self.g.edges(data='weight')]
        node_labels = {n: n for n in self.g.nodes}
        plt.figure(figsize=(10, 7))
        nx.draw(self.g, pos=nx.spring_layout(self.g),
                labels=node_labels, node_color=node_colors, node_size=node_sizes, font_color='white', font_size=10,
                edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'))
        plt.show()


class Lattice:
    def __init__(self, num_nodes, lattice: List[AttentionTensor], all_path_scores: Dict[int, PathScores]):
        self.g = self._build(num_nodes, lattice)
        self.num_nodes = num_nodes
        self.num_layers = len(lattice)
    
    def _get_node(self, layer, i):
        return f'{layer}_{i}'

    def _build(self, num_nodes, lattice: List[AttentionTensor]):
        g = nx.Graph()
        # first (input) layer
        nodes = [self._get_node(0, i) for i in range(num_nodes)]
        g.add_nodes_from(nodes, bipartite=0)
        # others
        for layer_idx, (edges, weights) in enumerate(lattice, 1):
            nodes = [self._get_node(layer_idx, i) for i in range(num_nodes)]
            g.add_nodes_from(nodes, bipartite=layer_idx)
            for (src, tgt), w in zip(edges, weights):
                src = self._get_node(layer_idx - 1, src)
                tgt = self._get_node(layer_idx, tgt)
                g.add_edge(src, tgt, weight=w)
        return g

    def draw(self):
        pos = {}
        # set positions for nodes, (node) top to bottom and (layer) left to right
        for layer_idx in range(self.num_layers + 1):
            pos.update((self._get_node(layer_idx, i), (layer_idx, self.num_nodes - i)) for i in range(self.num_nodes))
        # set labels and colors
        node_labels = {n: n.split('_')[1] for n in self.g.nodes}
        edge_colors = [_[-1] for _ in self.g.edges(data='weight')]
        nx.draw(self.g, pos=pos,
                labels=node_labels, node_size=500, font_color='white',
                edge_color=edge_colors, edge_cmap=plt.get_cmap('Greys'))
