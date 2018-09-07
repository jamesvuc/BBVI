# BBVI
A collection of Black Box Variational Inference algorithms implemented in an object-oriented Python framework using Autograd. 

`bbvi.py` provides a class `BaseBBVIModel` which is a base class for a general Bayesian inference problem to be solved by variational methods. Specifically, if we have a posterior p(z|x), `BaseBBVIModel` provides the framework and machinery to approximate p(z|x) by a distribution q(z|lambda) where lambda is the variational parameter(s) that determines the variational family. This is based on some excellent models found at https://github.com/HIPS/autograd/blob/master/examples/.

## How to use ##
1. Derive a model class from `BaseBBVIModel` which requires that you implement the log-posterior, the log-variational approximation, a sampler of the variational approximation, and a parameter-handler. 
2. Chose an ELBO gradient estimator (more on this below).
3. Call the method `run_VI` with the appropriate initial parameters.

## Available ELBO Estimators ##
This framework is designed to facilitate rapid experimentation with different BBVI methods. We have provided three ELBO gradient estimators, but it is a simple matter to add others afterward. The estimators broadly fall into two categories: "stochastic search" and "reparameterization":
### Stochastic Search ###
These follow the original approach of Ranganath et al. (https://arxiv.org/abs/1401.0118) and uses the gradient of the variational distribution. 
* `how='stochsearch'`: The original method from Ranganath et al. (2013) (https://arxiv.org/abs/1401.0118).
### Reparameterization
These follow the approach taken in Kingma et al. (https://arxiv.org/abs/1506.02557) by parameterizing the samples then using backprop.
* `how='reparam'`: The original method from Kingma et al. (2015) (https://arxiv.org/abs/1506.02557) and following the implementation from http://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf.
* `how='noscore'`: A clever and simple variant of reparameterization that omits the score function from the derivative estimate from Roder et al. (2017) https://papers.nips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference.pdf

We should also note that various enhancements to these models (Rao-Blackwellization, control variates) were considered but omitted because neither is easily applied to a black-box model. These models could be specialized to include these enhancements on a case-by-case basis.

## Examples ##
Three examples are provided, `BBVI_test1.py` and `BBVI_test2.py` deal with using a mutlivariate Gaussian to approximate a highly non-normal distribution. `BBVI_test3.py` shows how to do inference on the weights of a MLP. It is worthwhile noting that the variational approximation and much of the code is essentially the same in each example, and only the details of the specific problem need to be adjusted.
