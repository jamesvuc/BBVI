import matplotlib.pyplot as plt
from abc import ABCMeta, abstractmethod

from profilehooks import profile
import datetime as dt

import autograd.numpy as np
import autograd.numpy.random as npr

from autograd import grad
from autograd.misc.optimizers import adam
from autograd.core import getval


# ======Model====
class BaseBBVIModel(metaclass=ABCMeta):
	"""
	An abstract base class providing the structure for a general Bayesian
	inference problem to be solved using black box variational inference. 
	We provide a number of ELBO graient approximations, with ease of experimentation
	being a primary goal.

	To use this framework, one must derive their own class (i.e. model), and implement 
	the user-specified mehtods indicated below.

	The mechanics follow those of 
	https://github.com/HIPS/autograd/blob/master/examples/black_box_svi.py
	"""
	def __init__(self):

		self._init_var_params=None
		self._var_params=None

		self.N_SAMPLES=None
	"""
	=======User-specified methods=====
	These methods must be implemented when the model is derived from this base class.
	The user-specified signatures should match those below,.
	"""
	# Variational approx
	@abstractmethod
	def unpack_params(self, params):
		"""
		Unpacks the numpy array 'params' and returns a tuple of the parameters
		for use in log_var_approx and sample_var_approx.
		"""
		pass

	@abstractmethod
	def log_var_approx(self, z, params):
		"""
		Computes the log variational approximation of z to the posterior log_prob
		using variational parameters params. Should be vectorized over z.
		"""
		pass

	@abstractmethod
	def sample_var_approx(self, params, n_samples=1000):
		"""
		Returns samples from the variational approximation with parameters params.
		"""
		pass

	# Joint Distribution
	@abstractmethod
	def log_prob(self, z):
		"""
		Computes the log-posterior of latent variables z. 
		"""
		pass

	def callback(self, *args):
		"""
		Optional method called once per optimization step.
		"""
		pass

	"""
	=======-Generic VI methods=======
	"""

	"""------Stochastic Search-------"""
	def _objfunc(self, params, t):
		"""
		Implements an unadjusted stochastic-search BBVI gradient estimate according
		to https://arxiv.org/abs/1401.0118.
		"""
		samps=self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

		return np.mean(self.log_var_approx(samps, params)*(self.log_prob(samps)-self.log_var_approx(samps, getval(params))))

	def _objfuncCV(self, params, t):
		"""
		Experimental: Implements a version of above with an estimated control variate.
		"""
		samps=self.sample_var_approx(getval(params), n_samples=self.N_SAMPLES)

		a_hat=np.mean(self.log_prob(samps)-self.log_var_approx(samps, getval(params)))

		return np.mean(self.log_var_approx(samps, params)*(self.log_prob(samps)-self.log_var_approx(samps, getval(params))-a_hat))

	"""-----Reparameterization Trick--------"""

	def _estimate_ELBO(self, params, t):
		"""
		Implements the ELBO estimate from http://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf
		which in turn implements the reparamerization trick from https://arxiv.org/abs/1506.02557
		"""
		samps=self.sample_var_approx(params, n_samples=self.N_SAMPLES)

		# estimates -E[log p(z)-log q(z)]
		return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, params), axis=0)#this one appears to be correct

	def _estimate_ELBO_noscore(self, params, t):
		"""
		Implements the ELBO estimate from 
		https://papers.nips.cc/paper/7268-sticking-the-landing-simple-lower-variance-gradient-estimators-for-variational-inference.pdf
		which can reduce variance in certain cases.
		"""
		samps=self.sample_var_approx(params, n_samples=self.N_SAMPLES)

		#eliminates the score function
		return -np.mean(self.log_prob(samps) - self.log_var_approx(samps, getval(params)), axis=0)#this one appears to be correct

	"""-----Optimization------"""
	def run_VI(self, init_params, num_samples=50, step_size=0.01, num_iters=2000, how='stochsearch'):
		hows=['stochsearch', 'reparam', 'noscore']
		if how not in hows:
			raise KeyError('Allowable VI methods are', hows)

		self.N_SAMPLES=num_samples

		#select the gradient type
		if how == 'stochsearch':	
			#not CV
			_tmp_gradient=grad(self._objfunc)
			#CV
			# _tmp_gradient=grad(self._objfuncCV)

		elif how == 'reparam':
			_tmp_gradient=grad(self._estimate_ELBO)
		
		elif how == 'noscore':
			_tmp_gradient=grad(self._estimate_ELBO_noscore)

		else:
			raise Exception("Allowable ELBO estimates are",hows)

		#set the initial parameters
		self._init_var_params=init_params

		#start the clock
		s=dt.datetime.now()

		#run the VI
		self._var_params=adam(_tmp_gradient, self._init_var_params,
			step_size=step_size,
			num_iters=num_iters,
			callback=self.callback
		)

		#finished
		print('done in:',dt.datetime.now()-s)
		
		return self._var_params

if __name__=='__main__':
	from JV_BBVI_test import run_BBVI_test1

	run_BBVI_test()


