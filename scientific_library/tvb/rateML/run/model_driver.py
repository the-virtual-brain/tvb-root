from __future__ import print_function

import logging
import itertools
import argparse
import pickle

from tvb.simulator.lab import *

import os.path
import numpy as np
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray

import time
import tqdm

here = os.path.dirname(os.path.abspath(__file__))


class Driver_Setup:

	def __init__(self):
		self.args = self.parse_args()
		self.dt = 0.1
		self.connectivity = self.tvb_connectivity(self.args.tvbn)
		self.weights = self.connectivity.weights
		self.lengths = self.connectivity.tract_lengths
		self.tavg_period = 10.0
		self.n_inner_steps = int(self.tavg_period / self.dt)
		self.params, self.speeds, self.couplings = self.setup_params(self.args.n_coupling, self.args.n_speed)
		self.n_work_items, self.n_params = self.params.shape
		speedsmin = self.speeds.min()
		if self.speeds.min() <= 0.0:
			speedsmin = 0.1
		self.buf_len_ = ((self.lengths / speedsmin / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of
		self.states = 2
		self.exposures = 2

	def tvb_connectivity(self, tvbnodes):
		white_matter = connectivity.Connectivity.from_file(source_file="connectivity_"+str(tvbnodes)+".zip")
		white_matter.configure()
		return white_matter

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
							help="neural mass model to be used during the simulation",
							default='Oscillator'
							)
		parser.add_argument('--lineinfo', default=True, action='store_true')

		parser.add_argument('--filename', default="kuramoto_network.c", type=str,
							help="Filename to use as GPU kernel definition")

		parser.add_argument('-bx', '--blockszx', default="32", type=int, help="Enter block size x")
		parser.add_argument('-by', '--blockszy', default="32", type=int, help="Enter block size y")

		parser.add_argument('-val', '--validate', default=False, help="Enable validation to refmodels")

		parser.add_argument('--tvbn', default="68", type=int, help="Number of tvb nodes")

		args = parser.parse_args()
		return args


	def setup_params(self, n0, n1):  # {{{
		# the correctness checks at the end of the simulation
		# are matched to these parameter values, for the moment
		sweeparam0 = np.linspace(0.0, 2.0, n0)
		sweeparam1 = np.linspace(1.6, 3.0, n1)
		sweeparam1 = np.array([.0042] * n1)
		sweeparam0 = np.array([4.0] * n0)
		params = itertools.product(sweeparam0, sweeparam1)
		params = np.array([vals for vals in params], np.float32)
		return params, sweeparam0, sweeparam1  # }}}


class Driver_Execute(Driver_Setup):

	def __init__(self, ds):
		self.args = ds.args
		self.set_CUDAmodel_dir()
		self.weights, self.lengths, self.params = ds.weights, ds.lengths, ds.params
		self.buf_len, self.states, self.n_work_items = ds.buf_len, ds.states, ds.n_work_items
		self.n_inner_steps, self.n_params, self.dt = ds.n_inner_steps, ds.n_params, ds.dt
		self.exposures = ds.exposures

	def set_CUDAmodel_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "generatedModels", self.args.model.lower() + '.c')

	def compare_with_ref(self, logger, tavg0):
		self.args.model = self.args.model + 'ref'
		self.set_CUDAmodel_dir()
		tavg1 = self.start_cuda(logger)

		# compare output to check if same as template
		comparison = (tavg0.ravel() == tavg1.ravel())
		#logger.info('Templated version is similar to original %d:', comparison.all())
		#logger.info('Correlation coefficient: %f', corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1])

	def make_kernel(self, source_file, warp_size, block_dim_x, args, lineinfo=False, nh='nh'):

		try:
			with open(source_file, 'r') as fd:
				source = fd.read()
				source = source.replace('M_PI_F', '%ff' % (np.pi, ))
				opts = ['--ptxas-options=-v', '-maxrregcount=32', '-lineinfo']
				if lineinfo:
					opts.append('-lineinfo')
				opts.append('-DWARP_SIZE=%d' % (warp_size, ))
				opts.append('-DBLOCK_DIM_X=%d' % (block_dim_x, ))
				opts.append('-DNH=%s' % (nh, ))

				idirs = [here]
				# logger.info('nvcc options %r', opts)

				try:
					network_module = SourceModule(
							source, options=opts, include_dirs=idirs,
							no_extern_c=True,
							keep=False,)
				except drv.CompileError as e:
					logger.error('Compilation failure \n %s', e)
					exit(1)

				# generic func signature creation
				mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjfPfS_S_S_S_')

				step_fn = network_module.get_function(mod_func)

		except FileNotFoundError as e:
			logger.error('%s.\n  Generated model filename should match model on cmdline', e)

		return step_fn #}}}

	def cf(self, array):#{{{
		# coerce possibly mixed-stride, double precision array to C-order single precision
		return array.astype(dtype='f', order='C', copy=True)#}}}

	def nbytes(self, data):#{{{
		# count total bytes used in all data arrays
		nbytes = 0
		for name, array in data.items():
			nbytes += array.nbytes
		return nbytes#}}}

	def make_gpu_data(self, data):#{{{
		# put data onto gpu
		gpu_data = {}
		for name, array in data.items():
			try:
				gpu_data[name] = gpuarray.to_gpu(self.cf(array))
			except drv.MemoryError as e:
				logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large for this GPU',
							 e, self.args.n_coupling, self.args.n_speed)
				exit(1)
		return gpu_data#}}}

	def gpu_info(self):
		cmd = "nvidia-smi -q -d MEMORY,UTILIZATION"
		returned_value = os.system(cmd)  # returns the exit code in unix
		drv.mem_get_info()
		print('returned value:', returned_value)

	def run_simulation(self, logger):

		# setup data#{{{
		data = { 'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T }
		base_shape = self.n_work_items,
		for name, shape in dict(
			# tavg0=(self.args.tvbn, ),
			# tavg1=(self.args.tvbn, ),
			tavg0=(self.exposures, self.args.tvbn,),
			tavg1=(self.exposures, self.args.tvbn,),
			state=(self.buf_len, self.states * self.args.tvbn),
			).items():
			# memory error exception for compute device
			try:
				data[name] = np.zeros(shape + base_shape, 'f')
			except MemoryError as e:
				logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
							 'for this compute device',
							 e, self.args.n_coupling, self.args.n_speed)
				exit(1)

		gpu_data = self.make_gpu_data(data)#{{{

		logger.info('history shape %r', data['state'].shape)
		logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))#}}}

		# setup CUDA stuff#{{{
		step_fn = self.make_kernel(
			source_file=self.args.filename,
			warp_size=32,
			block_dim_x=self.args.n_coupling,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			args=self.args,
			lineinfo=self.args.lineinfo,
			nh=self.buf_len,
			# model=args.model,
			)#}}}

		# setup simulation#{{{
		tic = time.time()
		# logger.info('self.args.n_time %i', self.args.n_time)

		n_streams = 32
		streams = [drv.Stream() for i in range(n_streams)]
		events = [drv.Event() for i in range(n_streams)]
		tavg_unpinned = []

		try:
			tavg = drv.pagelocked_zeros((n_streams,) + data['tavg0'].shape, dtype=np.float32)
		except drv.MemoryError as e:
			logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large for this GPU',
						 e, self.args.n_coupling, self.args.n_speed)
			exit(1)

		# tavg = drv.pagelocked_zeros(data['tavg0'].shape, dtype=np.float32)
		# logger.info('data[tavg].shape %s', data['tavg'].shape)
		#}}}

		#  A ceiling devision because threads equal parameters. Less threads mean not all params are simulated
		gridx = int(np.ceil(self.args.n_coupling / self.args.blockszx))
		if (gridx==0):
			gridx=1
		gridy = int(np.ceil(self.args.n_speed / self.args.blockszy))
		if (gridy==0):
			gridy=1
		final_block_dim = self.args.blockszx, self.args.blockszy, 1
		final_grid_dim = gridx, gridy

		# print('final grid dim %r', final_grid_dim)
		# print('final block dim %r', final_block_dim)

		# logger.info('final block dim %r', final_block_dim)
		logger.info('final grid dim %r', final_grid_dim)
		# assert n_coupling_per_block * n_coupling_blocks == args.n_coupling #}}}

		# logger.info('gpu_data[lengts] %s', gpu_data['lengths'].shape)
		# logger.info('nnodes %r', args.tvbn)
		# logger.info('gpu_data[lengths] %r', gpu_data['lengths'])

		# run simulation#{{{
		# logger.info('submitting work')
		nstep = self.args.n_time

		try:
			for i in tqdm.trange(nstep, file=sys.stdout):

				try:
					event = events[i % n_streams]
					stream = streams[i % n_streams]

					if i > 0:
						stream.wait_for_event(events[(i - 1) % n_streams])

					step_fn(np.uintc(i * self.n_inner_steps), np.uintc(self.args.tvbn), np.uintc(self.buf_len),
							np.uintc(self.n_inner_steps), np.uintc(self.n_params), np.float32(self.dt),
							gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
							gpu_data['tavg%d' % (i%2,)],
							# gpu_data['tavg0'],
							block=final_block_dim, grid=final_grid_dim)

					event.record(streams[i % n_streams])
				except drv.LaunchError as e:
					logger.error('%s', e)
					exit(1)

				tavgk = 'tavg%d' % ((i + 1) % 2,)

				# async wrt. other streams & host, but not this stream.
				if i >= n_streams:
					stream.synchronize()
					tavg_unpinned.append(tavg[i % n_streams].copy())

				if i > 0:
					drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data[tavgk].ptr, stream=stream)
				# drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data['tavg0'].ptr, stream=stream)

				# tavg_unpinned.append(tavg.copy())
				# drv.memcpy_dtoh(
				# 	tavg,
				# 	gpu_data['tavg0'].ptr)

				# drv.memcpy_dtoh_async(tavg, gpu_data['tavg0'].ptr, stream=stream)

			# recover uncopied data from pinned buffer
			if nstep > n_streams:
				for i in range(nstep % n_streams, n_streams):
					stream.synchronize()
					tavg_unpinned.append(tavg[i].copy())

			for i in range(nstep % n_streams):
				stream.synchronize()
				tavg_unpinned.append(tavg[i].copy())

		except drv.LogicError as e:
			logger.error('%s. Check the number of states of the model or '
						 'GPU block shape settings blockdim.x/y %r.', e, final_block_dim)
			exit(1)
		except drv.RuntimeError as e:
			logger.error('%s', e)
			exit(1)


		# logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)
		print('kernel finished', file=sys.stderr)
		return tavg


if __name__ == '__main__':

	np.random.seed(79)

	tic = time.time()

	execute = Driver_Execute(Driver_Setup())

	logging.basicConfig(level=logging.DEBUG if execute.args.verbose else logging.INFO)
	logger = logging.getLogger('[TVB_CUDA]')

	# logger.info('dt %f', self.dt)
	# logger.info('nstep %d', self.args.n_time)
	# logger.info('n_inner_steps %f', self.n_inner_steps)
	# if self.args.test and self.args.n_time % 200:
	# 	logger.warning('rerun w/ a multiple of 200 time steps (-n 200, -n 400, etc) for testing')  # }}}
	#
	# # setup data
	# logger.info('weights.shape %s', self.weights.shape)
	# logger.info('lengths.shape %s', self.lengths.shape)
	# logger.info('n_nodes %d', self.args.tvbn)
	#
	# # couplings and speeds are not derived from the regular TVB connection setup routine.
	# # these parameters are swooped every GPU spawn
	# logger.info('single connectome, %d x %d parameter space', self.ns, self.nc)
	# logger.info('%d total num threads', self.ns * self.nc)
	# logger.info('min_speed %f', self.min_speed)
	# logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
	# logger.info('number of states %d', self.states)
	# logger.info('filename %s', self.args.filename)
	# logger.info('model %s', self.args.model)

	tac = time.time()
	# logger.info("Setup in: {}".format(tac - tic))

	tavg0 = execute.run_simulation(logger)
	toc = time.time()

	if (execute.args.validate==True):
		execute.compare_with_ref(logger, tavg0)

	toc = time.time()
	elapsed = toc - tic

	# write output to file
	tavg_file = open('tavg_data', 'wb')
	pickle.dump(tavg0, tavg_file)
	tavg_file.close()
	print('Finished CUDA simulation successfully in: {0:.3f}'.format(elapsed))
	print('in {0:.3f} M step/s'.format(1e-6 * execute.args.n_time * execute.n_inner_steps * execute.n_work_items / elapsed))
	# logger.info('finished')
