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
So far, the consensus was that it is still important to start with a large
model[^smallify].
Now, the lottery ticket hypothesis[^lth] (LTH) comes in, which states small sub-networks exist that -- when trained in isolation ** do achieve the same accuracy in the same training time as their large-scale counterparts.

#### Outline

* Background: Pruning
* The Lottery ticket hypothesis
* Transfer Learning with winning tickets

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

Before the lottery ticket hypothesis, the commo

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
3. use mask of retrained parameters and retrain with there original
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

#### iterative vs one-shot pruning

* With iterative pruning, smaller winning tickets can be identified than with
  one-shot pruning. But it is more expensive as it requires retraining.

#### global vs local pruning

During pruning, one can either prune to the desired fraction of weights at each
layer, or put the weights of all layers into one pool and prune globally.
In the original LTH paper[^lth], the authors use local pruning for LeNet and 
Conv-2/4/6, while they use global pruning for the deeper models: Resnet-18 and
VGG-19. The idea is that within deeper models, some layers' weights might be
more important to retain than others'[^trf2].

#### late resetting and learning rate warmup

  Learning rate warmup can help to find winning tickets for deeper models[^lth].
  In follow-up work, the authors have introduced a different technique to deal with deeper models: late resetting[^lth-at-scale].
  With late resetting, winning tickets start with weights very early in the training process (one and five epochs) of the original model.
  When late resetting is used, learning rate warm-up is not necessary anymore.


#### Both initialization and structure matter

LTH[^lth] compares winning tickets against random tickets.
These random tickets share the same structure but are re-initialized at random.
It can be argued, that the success of winning tickets does not only come from
the initialization but also from the structure itself.



#### Interaction with dropout

Dropout may prime a network to be pruned and could make winning tickets easier to find.
Gomez and Hinton[^tgt-drop] introduce an adaption of dropout that assigns higher drop
probabilities to low-magnitude weights to foster later pruning.


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


## Outlook

* The holy grail is how to identify winning tickets early in the
  training process.


## References

[^imp]: Han, Song, et al. ["Learning both weights and connections for efficient neural network."](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) NeurIPS 2015.
[^lth]: Frankle, Jonathan, and Michael Carbin. ["The lottery ticket hypothesis: Finding sparse, trainable neural networks."](https://arxiv.org/abs/1803.03635) ICLR 2019.
[^lth-at-scale]: Frankle, Jonathan, et al. ["The Lottery Ticket Hypothesis at Scale."](https://arxiv.org/abs/1903.01611) arXiv preprint arXiv:1903.01611 (2019).
[^trf1]: Mehta, Rahul. ["Sparse Transfer Learning via Winning Lottery Tickets."](https://arxiv.org/abs/1905.07785) arXiv preprint arXiv:1905.07785 (2019).
[^trf2]: Morcos, Ari S., et al. ["One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers."](https://arxiv.org/abs/1906.02773) arXiv preprint arXiv:1906.02773 (2019).
[^lth-nlp]: Yu, Haonan, et al. ["Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP."](https://arxiv.org/abs/1906.02768) arXiv preprint arXiv:1906.02768 (2019).
[^tgt-drop]: Gomez, Aidan N., et al. ["Learning Sparse Networks Using Targeted Dropout."](https://arxiv.org/abs/1905.13678) arXiv preprint arXiv:1905.13678 (2019).
[^smallify]: Leclerc, Guillaume, et al. ["Smallify: Learning network size while training."](https://arxiv.org/abs/1806.03723) arXiv preprint arXiv:1806.03723 (2018). 
