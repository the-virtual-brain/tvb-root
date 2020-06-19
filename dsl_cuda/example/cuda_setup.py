from tvb.simulator.lab import *
import numpy as np
import numpy.random as rgn
import math
import time
import logging
import itertools
import argparse
import os, sys

sys.path.insert(0, os.path.join(os.getcwd(), os.pardir))
import LEMS2CUDA

rgn.seed(79)

class TVB_test:

	def __init__(self):
		self.args = self.parse_args()
		self.sim_length = self.args.n_time # 400
		self.g = np.array([1.0])
		self.s = np.array([1.0])
		self.dt = 0.1
		self.period = 10.0
		self.omega = 60.0 * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(self.s, self.g, self.dt)
		self.integrator = integrators.EulerDeterministic(dt=self.dt)
		self.weights = self.SC = self.connectivity.weights
		self.lengths = self.connectivity.tract_lengths
		self.n_nodes = self.weights.shape[0]
		self.tavg_period = 10.0
		self.nstep = self.args.n_time  # 4s
		self.n_inner_steps = int(self.tavg_period / self.dt)
		self.nc = self.args.n_coupling
		self.ns = self.args.n_speed
		self.couplings, self.speeds = self.setup_params(self.nc, self.ns)
		self.params = self.expand_params(self.couplings, self.speeds)
		self.n_work_items, self.n_params = self.params.shape
		self.min_speed = self.speeds.min()
		self.buf_len_ = ((self.lengths / self.min_speed / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of 2
		self.states = 1

	def tvb_connectivity(self, speed, global_coupling, dt=0.1):
		white_matter = connectivity.Connectivity.from_file(source_file="paupau.zip")
		white_matter.configure()
		white_matter.speed = np.array([speed])
		white_matter_coupling = coupling.Linear(a=global_coupling)
		return white_matter, white_matter_coupling

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')
		parser.add_argument('-c', '--n_coupling', help='num grid points for coupling parameter', default=32, type=int)
		parser.add_argument('-s', '--n_speed', help='num grid points for speed parameter', default=32, type=int)
		parser.add_argument('-t', '--test', help='check results', action='store_true')
		parser.add_argument('-n', '--n_time', help='number of time steps to do (default 400)', type=int, default=400)
		parser.add_argument('-v', '--verbose', help='increase logging verbosity', action='store_true', default='-v')
		# parser.add_argument('-p', '--no_progress_bar', help='suppress progress bar', action='store_false')
		# parser.add_argument('--caching',
		# 					choices=['none', 'shared', 'shared_sync', 'shuffle'],
		# 					help="caching strategy for j_node loop (default shuffle)",
		# 					default='none'
		# 					)
		parser.add_argument('--node_threads', default=1, type=int)
		parser.add_argument('--model',
							choices=['Rwongwang', 'Kuramoto', 'Epileptor', 'Oscillator', \
									 'Oscillatorref', 'Kuramotoref', 'Rwongwangref'],
							help="neural mass model to be used during the simulation",
							default='Oscillator'
							)
		parser.add_argument('--lineinfo', default=True, action='store_true')

		parser.add_argument('--filename', default="kuramoto_network.c", type=str,
							help="Filename to use as GPU kernel definition")

		parser.add_argument('-b', '--bench', default="regular", type=str, help="What to bench: regular, numba, cuda")

		args = parser.parse_args()
		return args

	def expand_params(self, couplings, speeds):  # {{{
		# the params array is transformed into a 2d array
		# by first creating tuples of (speed, coup) and arrayfying then
		# pycuda (check) threats them as flattenened arrays but numba needs 2d indexing
		params = itertools.product(speeds, couplings)
		params = np.array([vals for vals in params], np.float32)
		return params  # }}}

	def setup_params(self, nc, ns):  # {{{
		# the correctness checks at the end of the simulation
		# are matched to these parameter values, for the moment
		couplings = np.logspace(1.6, 3.0, nc)
		speeds = np.logspace(0.0, 2.0, ns)
		return couplings, speeds  # }}}

	# Todo: check if this function work. derr_speed > 500 and derr_coupl < -1500 evaluate to false for pyCuda runs
	def check_results(self, n_nodes, n_work_items, tavg, weights, speeds, couplings, logger, args):
		r, c = np.triu_indices(n_nodes, 1)
		win_size = args.n_time  # 4s? orig 200 # 2s?
		win_tavg = tavg.reshape((-1, win_size) + tavg.shape[1:])
		err = np.zeros((len(win_tavg), n_work_items))
		logger.info('err.shape %s', err.shape)
		# TODO do cov/corr in kernel
		for i, tavg_ in enumerate(win_tavg):
			for j in range(n_work_items):
				fc = np.corrcoef(tavg_[:, :, j].T)
				# err[i, j] = ((fc[r, c] - weights[r, c])**2).sum()   weights is 1 dim array
				# logger.info('fc[r, c].shape %s, weights[r].shape %s', fc[r, c].shape, weights[r].shape)
				err[i, j] = ((fc[r, c] - weights[r, c]) ** 2).sum()
		# look at 2nd 2s window (converges quickly)
		err_ = err[-1].reshape((speeds.size, couplings.size))
		# change on fc-sc metric wrt. speed & coupling strength
		derr_speed = np.diff(err_.mean(axis=1)).sum()
		derr_coupl = np.diff(err_.mean(axis=0)).sum()
		logger.info('derr_speed=%f, derr_coupl=%f', derr_speed, derr_coupl)
		# if args.dataset == 'hcp':
		assert derr_speed > 350.0
		assert derr_coupl < -500.0
		# if args.dataset == 'sep':
		# 	assert derr_speed > 5e4
		# 	assert derr_coupl > 1e4

		logger.info('result OK')

	def start_cuda(self, logger):
		logger.info('start Cuda run')
		from cuda_run import CudaRun
		cudarun = CudaRun()
		tavg_data = cudarun.run_simulation(self.weights, self.lengths, self.params, self.speeds, logger,
										   self.args, self.n_nodes, self.n_work_items, self.n_params, self.nstep,
										   self.n_inner_steps, self.buf_len, self.states, self.dt, self.min_speed)

		# Todo: fix this for cuda
		# self.check_results(self.n_nodes, self.n_work_items, tavg_data, self.weights, self.speeds, self.couplings, logger, self.args)

	def set_CUDAmodel_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))), os.pardir,'CUDAmodels',
								 self.args.model.lower() + '.c')

	def set_states(self):
		if 'kuramoto' in self.args.model.lower():
			self.states = 1
		elif 'oscillator' in self.args.model.lower():
			self.states = 2
		elif 'wongwang' in self.args.model.lower():
			self.states = 2
		elif 'montbrio' in self.args.model.lower():
			self.states = 2
		elif 'epileptor' in self.args.model.lower():
			self.states = 6

	def startsim(self):

		tic = time.time()
		logging.basicConfig(level=logging.DEBUG if self.args.verbose else logging.INFO)
		logger = logging.getLogger('[TVB_CUDA]')

		logger.info('dt %f', self.dt)
		logger.info('nstep %d', self.nstep)
		logger.info('n_inner_steps %f', self.n_inner_steps)
		if self.args.test and self.args.n_time % 200:
			logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing')  # }}}

		# setup data
		logger.info('weights.shape %s', self.weights.shape)
		logger.info('lengths.shape %s', self.lengths.shape)
		logger.info('n_nodes %d', self.n_nodes)

		# couplings and speeds are not derived from the regular TVB connection setup routine.
		# these parameters are swooped every GPU spawn
		logger.info('single connectome, %d x %d parameter space', self.ns, self.nc)
		logger.info('%d total num threads', self.ns * self.nc)
		logger.info('min_speed %f', self.min_speed)
		logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)

		self.set_CUDAmodel_dir()

		self.set_states()

		logger.info('number of states %d', self.states)
		logger.info('filename %s', self.args.filename)
		logger.info('model %s', self.args.model)

		tac = time.time()
		logger.info("Setup in: {}".format(tac - tic))

		self.start_cuda(logger)

		toc = time.time()
		elapsed = toc - tic
		logger.info('Finished python simulation successfully in: %0.3f', elapsed)
		logger.info('%0.3f M step/s', 1e-6 * self.nstep * self.n_inner_steps * self.n_work_items / elapsed)
		logger.info('finished')


if __name__ == '__main__':

	example = TVB_test()

	# start templating the model specified on cli
	LEMS2CUDA.cuda_templating(example.args.model, '../../dsl_cuda/XMLmodels/')

	# start simulation with templated model
	example.startsim()
