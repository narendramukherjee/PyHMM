# PyHMM
MAP or MLE estimation of HMM paramters using EM. Right now, following is supported:
1. Categorical emissions, with Beta/Dirichlet priors on the parameters through pseudocounts.
2. Independent Bernoulli emissions, with Beta priors on the parameters through pseudocounts.

Variational inference of HMM parameters with VB-EM. Following distributions are supported:
1. Categorical emissions
2. Independent Bernoulli emissions

Check the examples notebooks for examples of MAP/MLE and variational inference.
