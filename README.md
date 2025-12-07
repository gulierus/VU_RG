# Project Transformer

## TODO: 7.12.
- Finish the experiments with GP with fixed hyperparameter
  - Check with a few data points
  - Does prediction at points far away from the measurement converge to the prior?
  - Is the variance at points x' underestimated?
- Analyze relations mentioned in [link](background/PFN-GP.md)
  - Try visualizing the corresponding quantities, such as the weights of $y$

- Try to train a more advanced PFN for a GP with distribution of hyperparameters
  - check how the PFN works with context data generated from GP with a fixed HP
  - analyze if the attention learned the correct kernel 


## TODO: 4.11.
- rewrite hand notes into latex notes with explanation of examples from the *GP_test_zone.ipynb file*.

## Selfstudy:
- done TODO from 21.10. (*GP_test_zone.ipynb file*).
- made notes about hyperparaametres estimations, Type II ML e.t.c.
- studied "Transformers Can Do Bayesian Inference".

## TODO: 21.10.
- hyperparameter selection/posterior
- select N (e.g. N=8) points from a step function and evaluate GPs of many hyperparameters
- compute marginal likelihood (2.30) on a grid of (l,sigma) and plot their posterior
- run black-box optimization to find maximum marginal likelihood of the hyperparameters
- perform a sensitivity study on the increasing number of observations

## TODO: 6.10.
- try GP on toy problem., e.g. a step function


