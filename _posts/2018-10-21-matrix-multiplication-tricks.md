---
layout: post
author: Lukas Galke
published: false
---

# From For-Loops to Matrix Multiplication

## Introduction 

For-loops are easy to write, but expensive to use in the context of vectors, matrices, and tensors.

MatMul: $$\boldsymbol{A} \cdot \boldsymbol{B} = \boldsymbol{C}$$

is defined as: $$c_{ij} = \sum_{k=1}^m a_{ik} \cdot b_{kj}$$


$$ \cdot : (M,N) * (N, K) = (M,K) $$

*Inner product:* $$v^T \cdot x$$

*Outer product:* $$ v \cdot x.T $$

The element of the resulting matrix $$c_{ij}$$ pairwise multiplied entries of the $$i$$th row of $$\boldsymbol{A}$$ and the $$j$$ column of $$\boldsymbol{B}$$.





## Related 

Blog post on geometric intuition of matrix multiplication.

TODO: find cite


## Computing the co-occurrence matrix of items for recommendations

Assume $X$ of shape $$(n_\text{users}, n_\text{items})$$ in which entry $$X_{u_,i}$$ is 1 iff user $$u$$ likes item $$i$$ and 0 otherwise.

Question: Which items co-occur with other items?

$$ M = X.T \cdot X $$

Then $$M_{a,b}$$ holds the number of times item $$a$$ co-occurred with item
$$b$$ (in the same item set of a user). The diagonal is, however, filled with
the *squares* of the raw-counts of an item.


## Embedding a term-document matrix

Now $$X$$ is a term-document matrix of shape $$(n_\text{documents}, n_\text{words})$$,
and we have an embeddind $$E$$ of shape $$(n_\text{words}, n_\text{embedding_dim}$$.

Then,

$$ M = X \cdot E $$

yields a matrix $$M$$ representing the documents as bags-of-embedded words
(shape: $$(n_\text{documents}, n_\text{embedding_dim})$$.

- When $$X$$ holds l1-normalized counts, the documents' vectors in $$M$$ are the
  *mean* of aggregated word vectorsj.
- When $$X$$ holds absolute word counts, the document's vectors in $$M$$ are the
  *sum* of aggregated word vectors.
- When $$X$$ holds TF-IDF l2-normalized counts, the document's vectors in $$M$$ are
  an tf-idf-weighted aggregation of word vectors.

## Computing class prototypes

Assume we have a document representation $$X$$ of shape $$(n_\text{documents},
n_\text{features}$$ along with a label-indicator matrix $$Y$$ of shape
$$(n_\text{documents}, n_\text{classes}$$, such that $$Y_{ij}$$ is 1, iff document
$$i$$ is assigned to class $$j$$.

Then, 

$$ M = Y^T \cdot E $$

yields a matrix $M$, in which all document representations are aggregated into
a class prototype.  Once again, depending on l1-normalization of $Y^T$, the
result is either the mean or the sum of the representation vectors.  Note that
this holds true not only for multi-class classification ($$Y$$ is one-hot encoded)
but also for multi-label classification ($$Y$$ is label-indicator matrix).


## Applying adjacency matrices 


## Piping a batch through neural network parameters

Typical frameworks implement the main operation as 

$$ X * W^T + b $$

## Conclusion

When you have a certain operation in mind, always consider the input and output shapes.
If the shapes match, maybe matrix multiplication does exactly what you had in mind.



