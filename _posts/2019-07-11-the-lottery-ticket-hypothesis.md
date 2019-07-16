---
layout: post
author: Lukas Galke
published: true
---

*-- Work in Progress --*

[TL;DR](#tldr)

<!-- overparametrization helps generalization !-->
The common rationale when training neural networks is that larger networks have
more capacity but are more prone to overfitting the training data.
Recent studies have shown that over-parametrization can, in fact, act as a regularizer and lead to improved generalization performance[^arora2018].
At the same time, neural networks become larger and larger using up to billions of parameters.
Researchers start to quantify the effort to train these large-scale models in
\$\$\$ on cloud computing platforms and in tons of carbon emissions.

After training, however, large parts of large-scale models can be pruned away without harming the accuracy of the model.
Pruning techniques date back to 1992 (CITE LECUN).
The motivation for pruning is to reduce the model size, and thus, space requirements and the energy consumption.
One pruning technique is *magnitude pruning*, which prunes those weights that have the lowest magnitude, and therefore, the lowest effect on the network output[^imp].

Before the lottery ticket hypothesis (LTH) the common experience was that pruned architectures were harder to train from scratch.
Now, the LTH states that subnetworks can be trained to match or even outperform the accuracy of the unpruned network.
These subnetworks are called *winning tickets*.
The author's compare the trained accuracy of winning tickets against randomly initialized weights with the same structure (*random tickets*). 
Why is this important?
The LTH suggests that it is not necessary to train a full-model, if only we could identify winning tickets early during training.
If this was possible, it could save us wallets of \$\$\$ and tons of carbon emissions. 

> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural
> network contains a subnetwork that is initialized such that—when trained in
> isolation—it can match the test accuracy of the original network after
> training for at most the same number of iterations.[^lth]

#### a minimal example: sum of two inputs
 
To get an intution on the LTH, let's consider the simple task of computing the sum of two inputs $$y = x_0 + x_1 $$.
We want to approximate the ground truth $$y$$ with a two-layer, linear neural net with $$n$$ hidden units and no bias.

<!-- $$\hat{y} = f(x) = \boldsymbol{W}^{(2)} \left( \boldsymbol{W}^{(1)} \boldsymbol{x} + \boldsymbol{b}^{(1)} \right) + \boldsymbol{b}^{(2)}$$ !-->

$$f(x) = \sum_i^n w^{(2)}_i (w^{(1)}_{i,1} x_1 + w^{(1)}_{i,2} x_2)$$

<center>
<img src="/assets/img/LTH.png" alt="exemplary neural net with a winning ticket" width="80%"/>
</center>


For humans, a winning ticket for the sum of two inputs is easy to determine.
Such a winning ticket would be $$w^{(1)}_{i,1} = w^{(1)}_{i,2} = w^{(2)}_{i} = 1$$ for some $$i$$ with all remaining weights being zero.
This winning ticket would even generalize out of the training data domain, as it
actually does compute the real sum of its two inputs.

No matter how large we chose the hidden layer size $$n$$, our winning ticket will always consist of three nonzero weights.
Thus, we can prune $$\frac{n-3}{n}$$ of the weights without harming accuracy.
When we start training with a mask consisting of only those three nonzero-parameters, the network eventually learns the correct weights.

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


#### iterative vs one-shot pruning

With iterative pruning, smaller winning tickets can be identified than with one-shot pruning.
This is in-line with the results of the paper on iterative magnitude pruning[^imp].
But it is more expensive because it requires iterative training.

#### global vs local pruning

During pruning, one can either prune to the desired fraction of weights at each
layer, or put the weights of all layers into one pool and prune globally.
In the original LTH paper[^lth], the authors use local pruning for LeNet and 
Conv-2/4/6, while they use global pruning for the deeper models: Resnet-18 and
VGG-19. The idea is that within deeper models, some layers' weights might be
more important to keep[^trf2].

#### late resetting vs learning rate warm-up

  Learning rate warmup can help to find winning tickets for deeper models[^lth].
  In follow-up work, the authors have introduced a different technique to deal with deeper models: late resetting[^lth-at-scale].
  With late resetting, winning tickets are initialized with weights early in the training process (about one and five epochs) of the original model.
  When late resetting is used, learning rate warm-up is not necessary anymore.

#### winning tickets' initialization and structure matter

LTH[^lth] compares winning tickets against random tickets.
These random tickets share the same structure but are re-initialized at random.
The success of winning tickets does not only come from
the initialization but also from the structure itself[^trf2].
The empirical results from the original LTH paper compares against randomly
initialized tickets with the same structure. This is more challenging than
comparing against random tickets whose mask is also drawn at random.

#### there are winning tickets outside of the image domain

Is the lottery ticket phenomenon an artefact of supervised image
classification with feed-forward convolutional nets nets or does it generalize to other
domains? Yu et al. could show that winning tickets also exist in RL and NLP
architectures[^lth-nlp].

#### winning tickets are transferable across tasks

Several works have analyzed whether winning tickets are transferable across tasks within the image domain[^trf1][^trf2].
Both works suggest that winning tickets are transferable across tasks.
However, each makes use of a particular relaxation compared to the original LTH.
Mehta[^trf1] relaxes late resetting to using the best weights anywhere in the training process on the source task.
Morcos et al.[^trf2] compare against random
tickets that are not only randomly initialized but also have a random structure,
which is less challenging.
The authors argue that the inductive biases of winning tickets consists of both the
initialization and the structure. They further observe that larger
datasets lead to better transferable winning tickets.

#### How good are random tickets with the same mask as winning tickets


#### how do winning tickets look like[^deconstruct]

* hypothesize that subnetworks work well when weights are close to their final
  values
* the only crucial element is the sign of the initialization
* sometimes, specific supermasks even work without further training


#### Pruning and dropout

Dropout is a well-known regularization method that encourages sparsity tolerance during training by setting a random fraction of weights or hidden units to zero.
However, when pruning is applied after training, the fraction of pruned weights depend on a heuristic such as the magnitude of the weights.
Gomez et al [^tgt-drop] pursue the idea of improving the interaction of dropout and pruning.
The idea is that dropout could be targeted to units, which are likely to be pruned, i.e., those with low magnitude.
In their paper[^tgt-drop], the authors analyze not only the standard unit-dropout but also weight-dropout (aka DropConnect), which is even closer to the employed pruning techniques.

#### l1 and l2 regularization terms

An L1 penalty on the weights of a neural network encourages sparse weights.
Counterintuitively, an L2 penalty leads to neural nets that
are more amenable to pruning than nets trained with an L1 penalty.

TODO: which paper was it?

#### identifying winning tickets early

To benefit from winning tickets at training time, it is not enough to know that a winning ticket exists.
The holy grail is how to identify winning tickets early in the training process.
Dettmers and Zettlemoyer[^fromscratch] do propose such an approach already.


#### tl;dr

* Under the lottery ticket hypothesis, neural nets contain sub-networks, whose initialization and
  structure yields to better results than the original network.
* Highly over-parametrized nets generalize better, which might be explained by exponentially more possible sub-networks to form winning tickets. 
* The LTH merely claims that such sub-networks exist. To benefit from this
  knowledge, one needs to find the winning tickets already during training.
* What can we learn from the LTH about initialization?

#### References

[^imp]: Han, Song, et al. ["Learning both weights and connections for efficient neural network."](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) NeurIPS 2015.
[^lth]: Frankle, Jonathan, and Michael Carbin. ["The lottery ticket hypothesis: Finding sparse, trainable neural networks."](https://arxiv.org/abs/1803.03635) ICLR 2019.
[^lth-at-scale]: Frankle, Jonathan, et al. ["The Lottery Ticket Hypothesis at Scale."](https://arxiv.org/abs/1903.01611) arXiv preprint arXiv:1903.01611 (2019).
[^deconstruct]: Zhou, Hattie, Janice Lan, Rosanne Liu, and Jason Yosinski. ["Deconstructing lottery tickets: Zeros, signs, and the supermask."](https://arxiv.org/abs/1905.01067) arXiv preprint arXiv:1905.01067 (2019).
[^trf1]: Mehta, Rahul. ["Sparse Transfer Learning via Winning Lottery Tickets."](https://arxiv.org/abs/1905.07785) arXiv preprint arXiv:1905.07785 (2019).
[^trf2]: Morcos, Ari S., et al. ["One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers."](https://arxiv.org/abs/1906.02773) arXiv preprint arXiv:1906.02773 (2019).
[^lth-nlp]: Yu, Haonan, et al. ["Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP."](https://arxiv.org/abs/1906.02768) arXiv preprint arXiv:1906.02768 (2019).
[^tgt-drop]: Gomez, Aidan N., et al. ["Learning Sparse Networks Using Targeted Dropout."](https://arxiv.org/abs/1905.13678) arXiv preprint arXiv:1905.13678 (2019).
[^smallify]: Leclerc, Guillaume, et al. ["Smallify: Learning network size while training."](https://arxiv.org/abs/1806.03723) arXiv preprint arXiv:1806.03722 (2018). 
[^fromscratch]: T Dettmers, L Zettlemoyer. ["Sparse Networks from Scratch: Faster Training without Losing Performance"](https://arxiv.org/abs/1907.04840) arXiv preprint arXiv:1907.04840.
[^arora2018]: Arora, Sanjeev, Nadav Cohen, and Elad Hazan. ["On the optimization of deep networks: Implicit acceleration by overparameterization."](https://arxiv.org/abs/1802.06509) ICML 2018.
[^morphnet]: Gordon, Ariel, Elad Eban, Ofir Nachum, Bo Chen, Hao Wu, Tien-Ju Yang, and Edward Choi. ["Morphnet: Fast & simple resource-constrained structure learning of deep networks."](http://openaccess.thecvf.com/content_cvpr_2018/html/Gordon_MorphNet_Fast__CVPR_2018_paper.html) In Proceedings of the IEEE Conference on Computer Vision and Pattern Recognition, pp. 1586-1595. 2018.

