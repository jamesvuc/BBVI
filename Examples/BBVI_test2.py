import matplotlib.pyplot as plt

import autograd.numpy as np
import autograd.numpy.random as npr
import autograd.scipy.stats.multivariate_normal as mvn
import autograd.scipy.stats.norm as norm

import sys
sys.path.append('..')
from bbvi import BaseBBVIModel

"""
This implements the example in http://www.cs.toronto.edu/~duvenaud/papers/blackbox.pdf
except using a full-size covariance matrix (rather than diagonal).
"""


class ModelTest2(BaseBBVIModel):
	def __init__(self, D=2):
		self.dim=D
		self.fig, self.ax=plt.subplots(1)
		plt.show(block=False)

		BaseBBVIModel.__init__(self)

	# specify the variational approximator
	def unpack_params(self, params):
		return params[:, 0], params[:, 1:]

	def log_var_approx(self, z, params):
		mu, root_sigma=self.unpack_params(params)
		sigma=0.5*np.dot(root_sigma, root_sigma.T)
		return mvn.logpdf(z, mu.flatten(), sigma)

	def sample_var_approx(self, params, n_samples=2000):
		mu, root_sigma=self.unpack_params(params)
		sigma=0.5*np.dot(root_sigma, root_sigma.T)
		return np.dot(npr.randn(n_samples, mu.shape[0]),np.sqrt(0.5)*root_sigma)+mu

	# specify the distribution to be approximated
	def log_prob(self, z):
		# the density we are approximating is two-dimensional
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
		plt.contour(X, Y, Z)
		ax.set_yticks([])
		ax.set_xticks([])


	def callback(self, *args):
		if args[1]%50==0:
			print(args[1])
			curr_params=args[0]
			plt.cla()
			self.plot_isocontours(self.ax, lambda z:np.exp(self.log_prob(z)))
			self.plot_isocontours(self.ax, lambda z:np.exp(self.log_var_approx(z, curr_params)))

			plt.pause(1.0/30.0)

			plt.draw()


def run_BBVI_test():
	init_params=np.hstack([np.ones((2,1))*0, np.eye(2)*-1])

	mod=ModelTest2()
	# var_params=mod.run_VI(init_params,
	# 	step_size=0.0001,
	# 	num_iters=4000,
	# 	num_samples=1000,
	# 	# how='stochsearch'
	# )

	var_params=mod.run_VI(init_params,
		step_size=0.005,
		num_iters=1500,
		num_samples=2000,
		# how='reparam'
		how='noscore'
	)


	print('init params=')
	print(init_params)
	print('final params=')
	print(var_params)


if __name__=='__main__':
	run_BBVI_test()