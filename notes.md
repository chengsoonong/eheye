# Nodes and edges with types

## Problem setup

Assume we have three tables:
- **Nodes**: representing the features of each node, along with its node type
- **Edges**: representing the features of each edge, along with its edge type
- **Links**: a set of triples: (node1, edge, node2) where we use the index/key of the node/edge.

## Data representation

Separate internal memory representation decision from on-disk representation decision.
To start, let's make both the same, e.g. using SQLite on disk with pandas tables internally.

## k-SVD

Use only information from **Links**.
Assuming that there are only two types of nodes, and one type of edge (with a scalar feature),
and that the graph is bipartite, we can represent **Links** as a matrix.
Use one type of node to index the rows, and the second type to index the columns.
For every edge in **Links** insert the value of the scalar feature, otherwise
the entry in the matrix is zero.

Apply k-SVD, for example using [TruncatedSVD](http://scikit-learn.org/stable/modules/generated/sklearn.decomposition.TruncatedSVD.html).
