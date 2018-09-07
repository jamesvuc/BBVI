from __future__ import absolute_import
from __future__ import print_function
import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

import sys
sys.path.append('..')
from bbvi import BaseBBVIModel

"""
=================
ANN functions here
==================
"""
def make_nn_funs(layer_sizes, L2_reg, noise_variance, nonlinearity=np.tanh):
    """These functions implement a standard multi-layer perceptron,
    vectorized over both training examples and weight samples."""
    shapes = list(zip(layer_sizes[:-1], layer_sizes[1:]))
    num_weights = sum((m+1)*n for m, n in shapes)

    def unpack_layers(weights):
        num_weight_sets = len(weights)
        for m, n in shapes:
            yield weights[:, :m*n]     .reshape((num_weight_sets, m, n)),\
                  weights[:, m*n:m*n+n].reshape((num_weight_sets, 1, n))
            weights = weights[:, (m+1)*n:]

    def predictions(weights, inputs):
        """weights is shape (num_weight_samples x num_weights)
           inputs  is shape (num_datapoints x D)"""
        inputs = np.expand_dims(inputs, 0)
        for W, b in unpack_layers(weights):
            outputs = np.einsum('mnd,mdo->mno', inputs, W) + b
            inputs = nonlinearity(outputs)
        return outputs

    def logprob(weights, inputs, targets):
        log_prior = -L2_reg * np.sum(weights**2, axis=1)
        preds = predictions(weights, inputs)
        log_lik = -np.sum((preds - targets)**2, axis=1)[:, 0] / noise_variance
        return log_prior + log_lik

    return num_weights, predictions, logprob


def build_toy_dataset(n_data=40, noise_std=0.1):
    D = 1
    rs = npr.RandomState(0)
    inputs  = np.concatenate([np.linspace(0, 2, num=n_data/2),
                              np.linspace(6, 8, num=n_data/2)])
    targets = np.cos(inputs) + rs.randn(n_data) * noise_std
    inputs = (inputs - 4.0) / 4.0
    inputs  = inputs.reshape((len(inputs), D))
    targets = targets.reshape((len(targets), D))
    return inputs, targets


if __name__ == '__main__':

    """Build the MLP and dataset"""
    # Specify inference problem by its unnormalized log-posterior.
    rbf = lambda x: np.exp(-x**2)#deep basis function model
    relu = lambda x: np.maximum(x, 0.)
    num_weights, predictions, logprob = \
        make_nn_funs(layer_sizes=[1, 20, 20, 1], L2_reg=0.1,
                     noise_variance=0.01, nonlinearity=rbf)

    inputs, targets = build_toy_dataset()

    # suitable for stochastic gradient approximation for large data here
    log_posterior = lambda weights, t: logprob(weights, inputs, targets)


    """Construct the Bayesian inference problem here"""
    class NN_VI(BaseBBVIModel):
        def __init__(self):
            # pass
            self.fig, self.ax=plt.subplots(1)
            plt.show(block=False)

            self.elbo_hist=[]

            BaseBBVIModel.__init__(self)

        def unpack_params(self, params):
            #mu log_sigma=params
            return params[:, 0], params[:, 1]

        def log_var_approx(self, z, params):
            #we will again approximate p(W|data) by N(W|mu, sigma)
            # where sigma is restricted to be diagonal.
            mu, log_sigma=self.unpack_params(params)
            sigma=np.diag(np.exp(2*log_sigma))+1e-6
            return mvn.logpdf(z, mu, sigma)

        def sample_var_approx(self, params, n_samples=2000):
            mu, log_sigma=self.unpack_params(params)
            return npr.randn(n_samples, mu.shape[0])*np.exp(log_sigma)+mu
            
        # specify the distribution to be approximated
        def log_prob(self, z):
           return log_posterior(z, 0)

        def callback(self, *args):
            self.elbo_hist.append(self._estimate_ELBO(args[0],0))
            if args[1]%5==0:
                # print(args[1])

                params, t, g = args
                # print("Iteration {} lower bound {}".format(t, -objective(params, t)))
                print("Iteration {}".format(t))

                # Sample functions from posterior.
                rs = npr.RandomState(0)
                mean, log_std = self.unpack_params(params)
                #rs = npr.RandomState(0)
                sample_weights = rs.randn(10, num_weights) * np.exp(log_std) + mean
                plot_inputs = np.linspace(-8, 8, num=400)
                outputs = predictions(sample_weights, np.expand_dims(plot_inputs, 1))

                # Plot data and functions.
                plt.cla()
                self.ax.plot(inputs.ravel(), targets.ravel(), 'bx')
                self.ax.plot(plot_inputs, outputs[:, :, 0].T)
                self.ax.set_ylim([-2, 3])
                plt.draw()
                plt.pause(1.0/60.0)

     # Initialize variational parameters
    rs = npr.RandomState(0)
    init_mean    = rs.randn(num_weights)
    init_log_std = -5 * np.ones(num_weights)
    # init_var_params = np.concatenate([init_mean, init_log_std])
    init_var_params = np.vstack([init_mean, init_log_std]).T

    mod=NN_VI()

    if False:
    # if True:
        #this is too noisy to be effective
        var_params=mod.run_VI(init_var_params,
            step_size=0.1,
            num_iters=500,
            num_samples=20,
            # num_samples=20,
            how='stochsearch'
        )
    else:
        #this works well
        var_params=mod.run_VI(init_var_params,
            # step_size=0.01,
            step_size=0.05,
            # num_iters=500,
            num_iters=200,
            num_samples=20,
            # how='reparam'
            how='noscore'
        )

    print('elbo=',mod._estimate_ELBO(var_params, 0))

    f,ax=plt.subplots(1)
    ax.plot(mod.elbo_hist)
    plt.show()

    