---
layout: post
author: Lukas Galke
published: false
---

### Background: Pruning

The key idea of pruning is to remove connections within a neural net without
harming its accuracy.  The first pruning techniques date back to 1992.  The
motivation for pruning is to reduce the model size and the energy consumption.
To actually do pruning several techniques have been proposed (CITE CITE CITE).
Magnitude pruning, for instance, prunes away those weights that have the lowest
magnitude, and therefore, the lowest effect on the network output.


### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks {% cite 2018arXiv180303635F --file references %}

> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural
> network contains a subnetwork that is initialized such that—when trained in
> isolation—it can match the test accuracy of the original network after
> training for at most the same number of iterations.
> {% cite 2018arXiv180303635F --file references %}


* Lottery Ticket Conjecture: SGD seeks out and trains a
  well-initialized subnetwork
* By this logic, overparameterized networks are easier to train
  because they have more combinations of subnetworks that are potential winning tickets.

### Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP {% cite 2019arxiv190602768y --file references %}

* ...
* ...

### One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers {% cite 2019arxiv190602773m --file references %}

### Sparse Transfer Learning via Winning Lottery Tickets {% cite 2019arXiv190507785M --file references %}

### Learning Sparse Networks Using Targeted Dropout {% cite 2019arXiv190513678G --file references %}

### Stabilizing the Lottery Ticket Hypothesis {% cite 2019arXiv190301611F --file references %}

* Iterative magnitude pruning at initialization is problematic for deeper networks.
* The authors propose to not prune at initialization but after few training epochs.

### References

{% bibliography --cited --file references -T bib %}
