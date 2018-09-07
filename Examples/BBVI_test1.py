import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

import sys
sys.path.append('..')
from bbvi import BaseBBVIModel

"""
This implements the example in "Black-Box Stochastic Variational Inference
in Five Lines of Python" By D. Duvenaud and R. Adams.
"""

class TestModel1(BaseBBVIModel):
	def __init__(self, D=2):
		self.dim=D
		plt.show(block=False)
		self.fig, self.ax=plt.subplots(2)
		self.elbo_hist=[]

		BaseBBVIModel.__init__(self)

	# specify the variational approximator
	def unpack_params(self, params):
		# print('params shape',params.shape)
		return params[:, 0], params[:, 1]

	def log_var_approx(self, z, params):
		mu, log_sigma=self.unpack_params(params)
		sigma=np.diag(np.exp(2*log_sigma))+1e-6
		return mvn.logpdf(z, mu, sigma)

	def sample_var_approx(self, params, n_samples=2000):
		mu, log_sigma=self.unpack_params(params)
		return npr.randn(n_samples, mu.shape[0])*np.exp(log_sigma)+mu

	# specify the distribution to be approximated
	def log_prob(self, z):
		mu, log_sigma = z[:, 0], z[:, 1]#this is a vectorized extraction of mu,sigma
		sigma_density = norm.logpdf(log_sigma, 0, 1.35)
		mu_density = norm.logpdf(mu, 0, np.exp(log_sigma))

		return sigma_density + mu_density

	def plot_isocontours(self, ax, func, xlimits=[-2, 2], ylimits=[-4, 2], numticks=101):
		x = np.linspace(*xlimits, num=numticks)
		y = np.linspace(*ylimits, num=numticks)
		X, Y = np.meshgrid(x, y)
		zs = func(np.concatenate([np.atleast_2d(X.ravel()), np.atleast_2d(Y.ravel())]).T)
		Z = zs.reshape(X.shape) 
		# plt.contour(X, Y, Z)
		ax.contour(X, Y, Z)
		ax.set_yticks([])
		ax.set_xticks([])


	def callback(self, *args):
		self.elbo_hist.append(self._estimate_ELBO(args[0], 0))
		if args[1]%50==0:
			print(args[1])
			curr_params=args[0]
			for a in self.ax:
				a.cla()
			self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_prob(z)))
			self.plot_isocontours(self.ax[0], lambda z:np.exp(self.log_var_approx(z, curr_params)))
			self.ax[1].plot(self.elbo_hist)
			self.ax[1].set_title('elbo estimate='+str(round(self.elbo_hist[-1],4)))
			plt.pause(1.0/30.0)

			plt.draw()


def run_BBVI_test():
	init_params=np.hstack([np.ones((2,1))*0, np.ones((2,1))*0])

	mod=TestModel1()

	var_params=mod.run_VI(init_params,
		step_size=0.0001,
		num_iters=5000,
		num_samples=1000,
		how='stochsearch'
	)

	# var_params=mod.run_VI(init_params,
	# 	step_size=0.01,
	# 	num_iters=1500,
	# 	num_samples=500,
	# 	# how='reparam'
	# 	# how='noscore'
	# )

	print('init params=')
	print(init_params)
	print('final params=')
	print(var_params)
	print('elbo=',mod._estimate_ELBO(var_params, 0))

if __name__=='__main__':

	run_BBVI_test()