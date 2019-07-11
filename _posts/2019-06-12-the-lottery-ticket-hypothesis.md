---
layout: post
author: Lukas Galke
published: true
---

## Outline

* Background: Pruning
* The Lottery ticket hypothesis
* Transfer Learning with winning tickets

### Background: Pruning

The key idea of pruning is to remove connections within a neural net without
harming its accuracy.  The first pruning techniques date back to 1992.  The
motivation for pruning is to reduce the model size and the energy consumption.
To actually do pruning several techniques have been proposed (CITE CITE CITE).
Magnitude pruning, for instance, prunes away those weights that have the lowest
magnitude, and therefore, the lowest effect on the network output[^imp].


### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks[^lth]

> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural
> network contains a subnetwork that is initialized such that—when trained in
> isolation—it can match the test accuracy of the original network after
> training for at most the same number of iterations.


* Before: the common experience was that pruned architectures are harder to train from scratch
* With iterative pruning, smaller winning tickets can be identifier than with
  one-shot pruning. But it is more expensive as it requires retraining.
* uses learning rate warmup for deeper models
* uses layer-wise pruning for LeNet, Conv-2/4/6, but global pruning for Resnet-18 and VGG-19
* Interaction with dropout: dropout may prime a network to be pruned and could make winning tickets easier to find
* compare against random tickets, retain mask but init randomly
* winning tickets generalize better than random tickets
* winning tickets' initialization is important
* winning tickets' structure is important
* Conjecture: SGD seeks out and trains a
  well-initialized subnetwork; overparameterized networks are easier to train
  because they have more combinations of subnetworks that are potential winning tickets.
* Using pruning techniques remains future work

### Stabilizing the Lottery Ticket Hypothesis[^lth-at-scale]

* Iterative magnitude pruning at initialization is problematic for deeper networks.
* Late resetting: The authors propose to not prune at initialization but after few training epochs.
* Learning rate warm-up is not necessary with late resetting


## Are winning tickets transferable across datasets?

### Sparse Transfer Learning via Winning Lottery Tickets[^trf1]

 similar to the paper described below

### One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers[^trf2]

* Analyze transfer within the image domain, on different image classification
  tasks (MNIST, CIFAR-10[0], ImageNet, Places365)
* Uses global pruning in favor of layer-wise pruning
* Use random structure for random tickets (different to original LTH), gain of
  winning tickets comes from init and structure
* Winning tickets transfer across datasets and optimmizers
* Winning tickets hold inductive biases which improve training of sparsified
  models
* Larger datasets lead to better transferable winning tickets




## Are there winning tickets in other domains?

### Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP[^lth-nlp]

Is the lottery ticket phenomenon an artefact of supervised image
classification with feed-forward conv nets or does it generalize to other
domains? The results show that, both in RL and NLP, winning tickets outperform
random tickets.


## More pruning techniques

### Learning Sparse Networks Using Targeted Dropout[^tgt-drop]

* standard training does not necessarily enfcourage nets to be amenable to
  pruning
* introduces targeted dropout
* idea: dropout itself enforces sparsity tolerance during training. Target
  presumably unimportant units with dropout.



## References

[^imp]: Han, Song, et al. ["Learning both weights and connections for efficient neural network."](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) NeurIPS 2015.
[^lth]: Frankle, Jonathan, and Michael Carbin. ["The lottery ticket hypothesis: Finding sparse, trainable neural networks."](https://arxiv.org/abs/1803.03635) ICLR 2019.
[^lth-at-scale]: Frankle, Jonathan, et al. ["The Lottery Ticket Hypothesis at Scale."](https://arxiv.org/abs/1903.01611) arXiv preprint arXiv:1903.01611 (2019).
[^trf1]: Mehta, Rahul. ["Sparse Transfer Learning via Winning Lottery Tickets."](https://arxiv.org/abs/1905.07785) arXiv preprint arXiv:1905.07785 (2019).
[^trf2]: Morcos, Ari S., et al. ["One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers."](https://arxiv.org/abs/1906.02773) arXiv preprint arXiv:1906.02773 (2019).
[^lth-nlp]: Yu, Haonan, et al. ["Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP."](https://arxiv.org/abs/1906.02768) arXiv preprint arXiv:1906.02768 (2019).
[^tgt-drop]: Gomez, Aidan N., et al. ["Learning Sparse Networks Using Targeted Dropout."](https://arxiv.org/abs/1905.13678) arXiv preprint arXiv:1905.13678 (2019).
