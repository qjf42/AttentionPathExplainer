### What is it
**`AttentionPathExplainer(APE)`** is a tool to help understand how GNN model make its prediction leveraging the attention mechanism. 

### How it works
[`MessagePassing`](https://pytorch-geometric.readthedocs.io/en/latest/notes/create_gnn.html) is a common GNN framework expressed as a neighborhood aggregation scheme (the information of each node is aggregated and updated by its neighbors of previous layer). Thus we consider model as a multi-layer structure whose all layer is of size $|V|$ (or a lattice of size $|V| \times (|Layer| + 1)$).

1. For instance, applying a 3-layer GNN model to a graph with 7 nodes (left), its message flow can be modelled as a lattice (right).

|||
|-|-|
| ![raw_graph](https://i.loli.net/2020/06/26/4H7fqLrj8nChmK9.png) | ![lattice](https://i.loli.net/2020/06/26/M53Ej6RBDifdlro.png) |

2. If the model layer aggregates information based on attentions, using attention weights as edge weights, the lattice might look like this.

![attn](https://i.loli.net/2020/06/26/QEkBC6fHc2FAdNI.png)

3. Then we use [`viterbi algorithm`](https://en.wikipedia.org/wiki/Viterbi_algorithm) to get <font color="red">top k paths</font> of total weights(products of edge weights on the path). The time complexity is $O(|Layer| \times |E|)$, and space complexity is $O(k |V|)$.

#### Further study
1. Since [Transformer can be thought as a special case of GNN](https://graphdeeplearning.github.io/post/transformers-are-gnns/), so Transformer-based model (or any multi-layer attention model) such as BERT can also take advantage of it.
2. Besides attention marchanism, other components like dense/conv/residual layers may also affect the information flow. As deep learning models are difficult to truly understand, *APE* is not guaranteed to make an reasonable explanation.
3. Attention mechanism itself is also arguable. [Attention is not Explanation](https://arxiv.org/abs/1902.10186) and [Attention is not not Explanation](https://arxiv.org/abs/1908.04626) posed quite different arguments about the usefulness, and [Synthesizer](https://arxiv.org/abs/2005.00743) found that even random attention is not useless.

### Usage
1. Prerequisites
    - Since *APE* is independent of model and learning framework, so only [NetworkX](https://networkx.github.io/documentation/stable/) is necessary to draw explanatory graphs.
2. Prepare attention weights
   - Modify your model so that it returns **a list of AttentionTensors**
   - **AttentionTensors** is a tuple of (indices, attn_weights), indices is a **numpy array** with shape ($|E|$, 2), attn_weights ($|E|$,), representing aggregating flows of each layer
   - Multi-head attentions should be pooled or selected first
3. Node Classification
    ``` python
    ex = explainer.NodeClassificationExplainer()
    # run viterbi algorithm, only need to be called once
    ex.fit(attention_tensors, topk=16)
    # draw subgraph of L(#layer)-hop neighbors of node 1, with edge score as weight and returns path scores ({Path: score}), Path is a tuple of node ids
    path_scores = ex.explain([1], visualize=True)
    ```
   - e.g
  
    ![node](https://i.loli.net/2020/06/26/nHiWF7ZzVYo9mXr.png)

4. Link Predition
    ``` python
    ex = explainer.LinkPredictionExplainer()
    # run viterbi algorithm, only need to be called once
    ex.fit(attention_tensors, topk=16)
    # draw subgraph of L(#layer)-hop neighbors of node pair (1,2), with edge score as weight and returns path scores ({Path: score}), Path is a tuple of node ids
    path_scores = ex.explain((1, 2), visualize=True)
    ```
    - e.g
  
    ![link](https://i.loli.net/2020/06/26/Cp684jx1r9oPXvV.png)

5. Transformer-based (TODO)
    ``` python
    ex = explainer.TransformerExplainer(512)
    ex.explain(query, attention_tensors, topk=4, visualize=True)
    ```

### Examples
  - Notebooks in `examples` give simple experiments using [pytorch_geometric](https://github.com/rusty1s/pytorch_geometric) as GNN framework

### TODO
- [ ] Prettify graph visualization
- [ ] Edge and path score renormalization in subgraph
- [ ] Transformer-based model
- [ ] Node labels (currently only shows ids)
