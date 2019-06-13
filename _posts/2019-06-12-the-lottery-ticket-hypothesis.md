---
layout: post
author: Lukas Galke
published: true
---




> **The Lottery Ticket Hypothesis.** A randomly-initialized, dense neural network contains a subnetwork that is initialized such that—when trained in isolation—it can match the test accuracy of the original network after training for at most the same number of iterations.
> {% cite 2018arXiv180303635F --file references %}



* Lottery Ticket Conjecture: SGD seeks out and trains a well-initialized subnetwork
* By this logic, overparameterized networks are easier to train because
they have more combinations of subnetworks that are potential winning tickets.


## References

{% bibliography --cited --file references %}
