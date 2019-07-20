---
layout: post
author: Lukas Galke
published: true

---

#### introduction

Neural networks become larger and larger and use up to billions of parameters.
Researchers start to quantify the effort to train these large-scale models in
\$\$\$ amounts on cloud computing platforms and in even in tons of carbon emissions.
The common understanding used to be that overparametrized networks have
more capacity but are also more prone to overfit the training data.
Recent studies have shown that overparametrization can, in fact, act as a regularizer and lead to improved generalization performance[^arora2018].

After training, however, large parts of such large-scale models can be pruned away without harming the accuracy of the model.
Pruning techniques date back to 1990[^braindmg] with LeCun et al.'s paper on optimal brain damage.
The motivation for pruning is to reduce the model size and thus, memory requirement, inference time, and energy consumption.
One pruning technique is *magnitude pruning*, which prunes those weights that have the lowest magnitude, and therefore, the lowest effect on the network output[^mp].

Before the lottery ticket hypothesis[^lth] (LTH) the common experience was that pruned architectures were harder to train from scratch.
Now, the LTH states that certain subnetworks can be trained to match or even outperform the accuracy of the original, unpruned network.
The key idea is iteratively train a network and prune its parameters until only a small fraction of parameters are remaining.
In each iteration, the surviving weights are reset to their initialization.
The resulting subnetwork can then be trained in comparable time to match the accuracy that the original network would achieve.
We call these subnetworks are referred to as *winning tickets*.

> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural
> network contains a subnetwork that is initialized such that—when trained in
> isolation—it can match the test accuracy of the original network after
> training for at most the same number of iterations.[^lth]

Their experiments show that they can find subnetworks that are only 10-20\%
of the size of their dense counterparts. On lower sparsity levels, these
subnetworks could achieve even higher test accuracies than the original networks.

Why is this important?
The LTH suggests that it is not necessary to train a full-model, if we could only identify winning tickets early in the training process.
If this was possible, it could save us wallets of \$\$\$ and tons of carbon emissions. 


#### a toy example: sum of two inputs
 
To get an intuition on the LTH, let's consider the simple task of computing the sum of two inputs $$y = x_0 + x_1 $$.
We want to approximate the ground truth $$y$$ with a two-layer, linear neural net with $$n$$ hidden units and no bias.

$$f(x) = \sum_i^n w^{(2)}_i (w^{(1)}_{i,1} x_1 + w^{(1)}_{i,2} x_2)$$

<center>
<img src="/assets/img/LTH.png" alt="exemplary neural net with a winning ticket" width="80%"/>
</center>

For humans, a winning ticket for the sum of two inputs is easy to determine.
Such a winning ticket would be $$w^{(1)}_{i,1} = w^{(1)}_{i,2} = w^{(2)}_{i} = 1$$ for some $$i$$ with all remaining weights being zero.
This winning ticket would even generalize out of the training data domain, as it
actually does compute the real sum of its two inputs.

No matter how large we chose the hidden layer size $$n$$, our winning ticket will always consist of three nonzero weights.
Thus, we can prune all but those three weights without harming accuracy.
An alternative would be that the input values are passed through the first layer and summed up in the second layer, which would need four weights in total.
When we start training with a mask for those three (or four) nonzero parameters, the network will eventually learn the correct weights.

#### how to identify winning tickets

To show that winning tickets exist, Frankle and Carbin[^lth] employ the following procedure, which they label *iterative magnitude pruning*:

1. Initialize model with parameters $$\theta_0$$ along with a mask $$m$$ set to
   all ones
2. Train the (masked) model for $$j$$ iterations 
3. Prune the lowest magnitude parameters and update $$m$$ accordingly.
4. Reset $$\theta[m]$$ to their values in $$\theta_0$$ and fix all other
   parameters to zero.
5. Repeat from step 2 unless stopping criterion on sparsity or
   validation accuracy is met 

The result is a subnetwork (given by mask m) along with its initialization, which may perform one more training pass.
In their study[^lth], the authors compare the accuracy of these winning tickets against the whole model and against random tickets.
Random tickets share the same structure but are re-initialized randomly before
the final training pass.
The main result is that the winning tickets consistently lead to higher scores
than random tickets, and can also match or even outperform the full model.

The authors conjecture that the optimizer focuses on training the weights of a well-initialized sub-network.
The number of possible subnetworks grows exponentially with the number of
parameters. This may be an explanation why highly overparametrized networks
generalize better[^arora2018].

#### iterative vs one-shot pruning

The authors compare one-shot pruning, i.e. do only one pass of training, against
the iterative pruning procedure described above.
With iterative pruning, smaller winning tickets can be identified than with one-shot pruning.
This is in-line with the results of the paper on iterative magnitude pruning[^mp],
but it is more expensive because it requires iterative training.

#### global vs local pruning

During pruning, one can either prune to the desired fraction of weights at each
layer, or put the weights of all layers into one pool and prune globally.
In the LTH paper[^lth], the authors use local pruning for LeNet and 
Conv-2/4/6, while they use global pruning for the deeper models: Resnet-18 and
VGG-19. The idea is that within deeper models, the weights of some layers might be
more important to keep[^trf2]. On the other side, you need to be careful with
global pruning because it can, in principle, prune a whole layer which renders
your model untrainable.

#### late resetting vs learning rate warm-up

Learning rate warmup can help to find winning tickets for deeper models[^lth].
In follow-up work, the authors have introduced a different technique to deal with deeper models: late resetting[^lth-at-scale].
With late resetting, the weights are not resetted to their values before the first training iteration but to some values very early in the training process (after about one to five iterations).
When late resetting is used, learning rate warm-up is not necessary anymore.
Late resetting is specifically important to find winning tickets for deeper
models.

#### winning tickets' initialization and structure matter

**TODO: prune this paragraph, maybe include in "how do winning tikets look
like"**

In the LTH[^lth] paper, winning tickets are evaluated against random tickets.
These random tickets share the same structure but are re-initialized at random.
The inductive bias of winning tickets comes from both 
the initialization and the structure[^trf2].
The empirical results from the original LTH paper compares against randomly
initialized tickets with the same structure. This is more challenging than
comparing against random tickets whose mask is also drawn at random.

#### winning tickets outside of the image domain

Is the lottery ticket phenomenon an artefact of supervised image classification with feed-forward convolutional nets nets or does it generalize to other domains?
Yu et al.
[^lth-nlp] could show that winning tickets also exist in reinforcement learning and natural language processing architectures.
Their experiments include classic control problems, Atari games, LSTMs, and Transformers.
They could find winning tickets in all settings, which suggests, that the LTH phenomenon is not restricted to supervised image classification but might be a general feature of deep neural nets.

#### are winning tickets transferable across tasks?

Several works have analyzed whether winning tickets are transferable across tasks within the image domain[^trf1][^trf2].
Both works suggest that winning tickets are transferable across tasks.
However, each makes use of a particular relaxation.
Mehta[^trf1] relaxes late resetting to using the best weights anywhere in the training process on the source task.
His explanation of this decision is that the purpose of transfer learning is to save training effort on the target task.

Morcos et al.[^trf2] compare against random
tickets that are not only randomly initialized but are also randomly permuted.
The authors argue that the inductive biases of winning tickets consists of both the
initialization and the structure. They further observe that larger
datasets lead to better transferable winning tickets.

#### how do winning tickets look like

Zhou et al.[^deconstruct] have conducted a closer investigation on winning
tickets. They show that a crucial element of the initialization is the sign of
the weight. Furthermore, the authors develop the notion of supermasks that lead to good
accuracy even without further training. They claim that sparse
subnetworks work particularly well, when inititializations are close to their final form.

#### pruning and dropout

Dropout is a well-known regularization method that encourages sparsity tolerance during training by setting a random fraction of weights or hidden units to zero.
However, when pruning is applied after training, the fraction of pruned weights depend on a heuristic such as the magnitude of the weights.
Gomez et al.[^tgt-drop] pursue the idea of improving the interaction of dropout and pruning.
The idea is that dropout could be targeted to units, which are likely to be pruned, i.e., those with low magnitude.
In the paper, the authors analyze not only the standard unit-dropout but also weight-dropout (aka DropConnect), which is even closer to the employed pruning techniques.


#### pruning on-the-go

The holy grail of winning tickets is to identify them as early as possible in the training process.
Dettmers and Zettlemoyer[^fromscratch] propose a technique to identify winning tickets without the need for expensive retraining.
They exploit the momentum of the gradients to determine how fast weights change during training and prune those that do not change much.
Furthermore, the values of pruned weights are redistributed dynamically.
The results show that this so-called sparse momentum technique outperforms their baselines for sparse learning.

#### limitations

There are also studies that challenge the LTH:
Gale et al.[^ch1] conduct a large-scale comparison of sparse neural nets on
machine translation with transfomers and image classification with ResNet-50.
They confirm that naive magnitude pruning[^mp] is the best pruning technique among the compared ones. However, they report that the LTH approach fails to find winning tickets for these architectures.
Liu et al[^ch2] show that -- with a carefully selected learning rate -- random tickets can perform as well as winning tickets.
Both works, however, did not yet use late resetting[^lth-at-scale],
which helps to find winning tickets especially in deep architectures.
Another limitation is that the LTH only claims that winning tickets exist, but does not give you winning tickets right-away.
To find a winning ticket, you also need to start with a large, dense model in the first place.


#### experiments on the sum-of-two-inputs example

Let's implement the sum-of-two-inputs example from [above](#a-toy-example-sum-of-two-inputs).
We use local pruning such that the second layer does not get pruned away
completely.
We begin with 200 hidden units and train for 10 epochs for each pruning round.
We iteratively prune 25% of the weights (by magnitude) until only 2 weights are left in each layer.
In each round, we use late resetting to the weights after the first training iteration.
We expect that the winning ticket will pass the two inputs through the first
layer and sum them up in the second layer.
We train on the interval [-1,1) and test on the interval [1,2).
We end up with the following results for the root mean squared error.

```
Pruning 0.25 weights => 4 weights still active ~= 0.67%
Stopping criterion met.
This is your winning ticket:
        Layer 0
        [60,0]: 0.9429353475570679
        [60,1]: 0.9429353475570679
        Layer 1
        [0,60]: 1.06050705909729
        [0,101]: -0.4135688245296478
Winning ticket RMSE: 6.870159915858438e-05
This is a random ticket:
        Layer 0
        [60,0]: -0.8950394988059998
        [60,1]: -0.8950406908988953
        Layer 1
        [0,60]: -1.1172559261322021
        [0,101]: 0.008170465007424355
Reinit Random ticket RSME: 7.364383157981381e-05
This is a permuted random ticket:
        Layer 0
        [2,0]: 0.15243083238601685
        [2,1]: -0.40985623002052307
        Layer 1
        [0,60]: -0.002661715727299452
        [0,101]: 0.03881140798330307
Permute+Reinit Random ticket RSME: 6.60740392345907
Full-model RSME: 7.780414107555133e-06
```

We observe that iterative magnitude pruning yields a winning ticket that
corresponds to our human intuition (or at least, it comes close).
Please note that the [0,101] weight on the second layer irrelevant as it will only ever receive zero inputs.
The winning ticket's accuracy with 4 weights is on par with
the accuracy of the full model with 600 weights. The randomly reinitialized
ticket also succeeds to learn good weights (the inverted signs cancel each other out).
In contrast, the locally permuted ticket has a dead end and cannot learn anything.

What can we learn from implementing the "sum-of-two-inputs" toy example?

1. We can verify our previous thought-experiment that it is possible to find a winning ticket with only four weights
   that leads comparable error as the full 600 parameter model.
2. For this simple task, the initialization of winning tickets might be less
   important than it is in other tasks.
3. We see that a comparison with randomly permuted tickets is dangerous. The
   random permutation has lead to a "dead end", which may render the model
   untrainable.

#### conclusion

The lottery ticket hypothesis states that dense neural networks contain sparse subnetworks that can be trained in isolation to match the performance of the dense net.
This phenomenon offers a novel interpretation of overparametrization, which leads to exponentially more draws from the lottery.
To benefit from their existence, one needs to find methods to identify winning tickets early and without training the full model at all.
Some approaches already tackle this, while others focus on training methods that make neural networks more amenable to later pruning.
If we could identify winning tickets early or transfer them to other domains, we would save substantial amounts of training effort.
Winning tickets sometimes even outperform the original networks, which might have implications for our understanding of and the design of architectures and their initializations.
We can further confirm that iterative magnitude pruning succeeds to finds
winning tickets that correspond to human wisdom for a simple task.


#### bonus material

* [A clean and generic pytorch implementation of magnitude pruning](https://github.com/torch-pruning) featuring our beloved toy task to tinker around with.

#### references

[^mp]: Han, Song, et al. ["Learning both weights and connections for efficient neural network."](https://papers.nips.cc/paper/5784-learning-both-weights-and-connections-for-efficient-neural-network.pdf) NeurIPS 2015.
[^lth]: Frankle, Jonathan, and Michael Carbin. ["The lottery ticket hypothesis: Finding sparse, trainable neural networks."](https://arxiv.org/abs/1803.03635) ICLR 2019.
[^lth-at-scale]: Frankle, Jonathan, et al. ["The Lottery Ticket Hypothesis at Scale."](https://arxiv.org/abs/1903.01611) arXiv preprint arXiv:1903.01611 (2019).
[^deconstruct]: Zhou, Hattie, Janice Lan, Rosanne Liu, and Jason Yosinski. ["Deconstructing lottery tickets: Zeros, signs, and the supermask."](https://arxiv.org/abs/1905.01067) arXiv preprint arXiv:1905.01067 (2019).
[^trf1]: Mehta, Rahul. ["Sparse Transfer Learning via Winning Lottery Tickets."](https://arxiv.org/abs/1905.07785) arXiv preprint arXiv:1905.07785 (2019).
[^trf2]: Morcos, Ari S., et al. ["One ticket to win them all: generalizing lottery ticket initializations across datasets and optimizers."](https://arxiv.org/abs/1906.02773) arXiv preprint arXiv:1906.02773 (2019).
[^lth-nlp]: Yu, Haonan, et al. ["Playing the lottery with rewards and multiple languages: lottery tickets in RL and NLP."](https://arxiv.org/abs/1906.02768) arXiv preprint arXiv:1906.02768 (2019).
[^tgt-drop]: Gomez, Aidan N., et al. ["Learning Sparse Networks Using Targeted Dropout."](https://arxiv.org/abs/1905.13678) arXiv preprint arXiv:1905.13678 (2019).
[^fromscratch]: T Dettmers, L Zettlemoyer. ["Sparse Networks from Scratch: Faster Training without Losing Performance"](https://arxiv.org/abs/1907.04840) arXiv preprint arXiv:1907.04840.
[^arora2018]: Arora, Sanjeev, Nadav Cohen, and Elad Hazan. ["On the optimization of deep networks: Implicit acceleration by overparameterization."](https://arxiv.org/abs/1802.06509) ICML 2018.
[^braindmg]: LeCun, Yann, John S. Denker, and Sara A. Solla. ["Optimal brain damage."](http://papers.nips.cc/paper/250-optimal-brain-damage.pdf) In Advances in neural information processing systems, pp. 598-605. 1990.
[^ch1]: Gale, Trevor, Erich Elsen, and Sara Hooker. ["The state of sparsity in deep neural networks."](https://arxiv.org/abs/1902.09574) arXiv preprint arXiv:1902.09574 (2019).
[^ch2]: Liu, Zhuang, Mingjie Sun, Tinghui Zhou, Gao Huang, and Trevor Darrell. ["Rethinking the value of network pruning."](https://arxiv.org/abs/1810.05270) arXiv preprint arXiv:1810.05270 (2018).

