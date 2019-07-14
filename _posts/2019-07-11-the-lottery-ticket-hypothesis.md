---
layout: post
author: Lukas Galke
published: true
---

*-- Work in Progress --*

Neural networks become larger and larger using up to billions of parameters.
Studies show that drastic over-parametrization indeed leads to improved generalization performance.
Researchers start to quantify the effort to train these large-scale models in
\$\$\$ on cloud computing platforms and also in carbon emissions.

After training, however, large parts of these large-scale models can be pruned away without harming the accuracy of the model[^imp].
The common experience until now was that the pruned networks cannot be trained from scratch.
Now, the lottery ticket hypothesis[^lth] (LTH) comes in, which states small sub-networks exist that -- when trained in isolation -- do achieve the same accuracy in the same training time as their large-scale counterparts.
Why is this important?
The LTH suggests that it is not necessary to train a full-model, if only we could identify winning tickets early during training.
This could save us wallets of \$\$\$ and tons of carbon emissions.

#### Outline

* Background: Pruning
* The initialization lottery, winning tickets, and how to find them
* Do winning tickets transfer?
* [TL;DR](#tldr)

### Background: Pruning

The key idea of pruning is to remove connections within a neural net without
harming its accuracy. Pruning techniques date back to 1992.  The
motivation for pruning is to reduce the model size, and thus, space requirements and the energy consumption.
There are several pruning techniques.
Magnitude pruning, for instance, prunes away those weights that have the lowest
magnitude, and therefore, the lowest effect on the network output[^imp].


### The Lottery Ticket Hypothesis: Finding Sparse, Trainable Neural Networks[^lth]

Over-parametrization plays a key role in deep learning. Despite the common
fear that over-parametrized models tend to overfit, research shows that
it, in fact, helps to generalize.

> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural
> network contains a subnetwork that is initialized such that—when trained in
> isolation—it can match the test accuracy of the original network after
> training for at most the same number of iterations.


Before the lottery ticket hypothesis (LTH) the common experience was that pruned architectures were harder to train from scratch.
The LTH, however, states that subsets of weights can be trained to match or even outperform the accuracy of the unpruned network, when the initializiation is retained. These subsets are called *winning tickets*. They author's compare the trained accuracy of winning tickets against randomly initialized weights with the same structure (*random tickets*) as a winning ticket. 
The authors have shown that winning tickets exists for LeNet and Conv-2/4/6, Resnet-18 and VGG-19. 

#### Procedure to identify winning tickets

until desired sparsity is reached:
1. train a full model
2. prune away a certain fraction of parameters
3. use masked weights and retrain with their original
   initialization
4. repeat from step 2 (iterative pruning)

* compare against random tickets, retain mask but init randomly
* winning tickets generalize better than random tickets
* winning tickets' initialization is important
* winning tickets' structure is important
* Conjecture: SGD seeks out and trains a
  well-initialized subnetwork; overparameterized networks are easier to train
  because they have more combinations of subnetworks that are potential winning tickets.
* Using pruning techniques remains future work



### how to find winning tickets?

Several tricks are necessary to find winning tickets via pruning.

#### iterative vs one-shot pruning

* With iterative pruning, smaller winning tickets can be identified than with
  one-shot pruning. But it is more expensive as it requires retraining.

#### global vs local pruning

During pruning, one can either prune to the desired fraction of weights at each
layer, or put the weights of all layers into one pool and prune globally.
In the original LTH paper[^lth], the authors use local pruning for LeNet and 
Conv-2/4/6, while they use global pruning for the deeper models: Resnet-18 and
VGG-19. The idea is that within deeper models, some layers' weights might be
more important to keep than others'[^trf2].

#### late resetting and learning rate warmup

  Learning rate warmup can help to find winning tickets for deeper models[^lth].
  In follow-up work, the authors have introduced a different technique to deal with deeper models: late resetting[^lth-at-scale].
  With late resetting, winning tickets start with weights very early in the training process (one and five epochs) of the original model.
  When late resetting is used, learning rate warm-up is not necessary anymore.


#### Winning tickets' initialization and structure matter

LTH[^lth] compares winning tickets against random tickets.
These random tickets share the same structure but are re-initialized at random.
The success of winning tickets does not only come from
the initialization but also from the structure itself[^trf2].
The empirical results from the original LTH paper compares against randomly
initialized tickets with the same structure. This is more challenging than
comparing against random tickets whose mask is also drawn at random.

### Deconstructing lottery tickets: Zeros, signs, and the supermask[^deconstruct]

* hypothesize that subnetworks work well when weights are close to their final
  values
* the only crucial element is the sign of the initialization
* sometimes, specific supermasks even work without further training

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




### Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP[^lth-nlp]

Is the lottery ticket phenomenon an artefact of supervised image
classification with feed-forward convolutional nets nets or does it generalize to other
domains? The results show that, both in RL and NLP, winning tickets outperform
random tickets.


#### Pruning and dropout

Dropout is a well-known regularization method that encourages sparsity tolerance during training by setting a random fraction of weights or hidden units to zero.
However, when pruning is applied after training, the fraction of pruned weights depend on a heuristic such as the magnitude of the weights.
Gomez et al [^tgt-drop] pursue the idea of improving the interaction of dropout and pruning.
The idea is that dropout could be targeted to units, which are likely to be pruned, i.e., those with low magnitude.
In their paper[^tgt-drop], the authors analyze not only the standard unit-dropout but also weight-dropout (aka DropConnect), which is even closer to the employed pruning techniques.

#### L1 and L2 Norm

An L1 penalty on the weights of a neural network encourages sparse weights.
Counterintuitively, it was shown that an L2 penalty leads to neural nets that
are more amenable to pruning than nets with an L1 penalty.

#### Identifying winning tickets early

To benefit from winning tickets at training time, it is not enough to know that a winning ticket exists.
The holy grail is how to identify winning tickets early in the training process.
Dettmers and Zettlemoyer[^fromscratch] do propose such an approach already.


## tl;dr

* Under the lottery ticket hypothesis, neural nets contain sub-networks, whose initialization and
  structure yields to better results than the original network.
* Highly over-parametrized nets generalize better, which might be explained by exponentially more possible sub-networks to form winning tickets. 
* The LTH merely claims that such sub-networks exist. To benefit from this
  knowledge, one needs to find the winning tickets already during training.
* What can we learn from the LTH about initialization?

## References

[^imp]: Han, Song, et al. ["Learning both weights and connections for efficient neural network."](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) NeurIPS 2015.
[^lth]: Frankle, Jonathan, and Michael Carbin. ["The lottery ticket hypothesis: Finding sparse, trainable neural networks."](https://arxiv.org/abs/1803.03635) ICLR 2019.
[^lth-at-scale]: Frankle, Jonathan, et al. ["The Lottery Ticket Hypothesis at Scale."](https://arxiv.org/abs/1903.01611) arXiv preprint arXiv:1903.01611 (2019).
[^deconstruct]: Zhou, Hattie, et al. ["Deconstructing lottery tickets: Zeros, signs, and the supermask."](https://arxiv.org/abs/1905.01067) arXiv preprint arXiv:1905.01067 (2019).
[^trf1]: Mehta, Rahul. ["Sparse Transfer Learning via Winning Lottery Tickets."](https://arxiv.org/abs/1905.07785) arXiv preprint arXiv:1905.07785 (2019).
[^trf2]: Morcos, Ari S., et al. ["One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers."](https://arxiv.org/abs/1906.02773) arXiv preprint arXiv:1906.02773 (2019).
[^lth-nlp]: Yu, Haonan, et al. ["Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP."](https://arxiv.org/abs/1906.02768) arXiv preprint arXiv:1906.02768 (2019).
[^tgt-drop]: Gomez, Aidan N., et al. ["Learning Sparse Networks Using Targeted Dropout."](https://arxiv.org/abs/1905.13678) arXiv preprint arXiv:1905.13678 (2019).
[^smallify]: Leclerc, Guillaume, et al. ["Smallify: Learning network size while training."](https://arxiv.org/abs/1806.03723) arXiv preprint arXiv:1806.03722 (2018). 
[^fromscratch]: T Dettmers, L Zettlemoyer. ["Sparse Networks from Scratch: Faster Training without Losing Performance"](https://arxiv.org/abs/1907.04840) arXiv preprint arXiv:1907.04840.

++ Morphnet?
