from __future__ import print_function

import logging
import itertools
import argparse
import pickle

from tvb.simulator.lab import *
from tvb.basic.logger.builder import get_logger

from tvb.rateML.run.regular_run import regularRun

import os.path
import numpy as np
try:
	import pycuda.autoinit
	import pycuda.driver as drv
	from pycuda.compiler import SourceModule
	import pycuda.gpuarray as gpuarray
except ImportError:
	logging.warning('pycuda not available, rateML driver not usable.')

import matplotlib.pyplot as plt

import time
import tqdm

here = os.path.dirname(os.path.abspath(__file__))

class Driver_Setup:

	def __init__(self):
		self.args = self.parse_args()

		self.logger = get_logger('tvb.rateML')
		self.logger.setLevel(level='INFO' if self.args.verbose else 'WARNING')

		self.checkargbounds()

		self.dt = self.args.delta_time
		self.connectivity = self.tvb_connectivity(self.args.n_regions)
		self.weights = self.connectivity.weights
		self.lengths = self.connectivity.tract_lengths
		self.tavg_period = 1.0
		self.n_inner_steps = int(self.tavg_period / self.dt)

		self.params = self.setup_params(
		self.args.n_sweep_arg0,
		self.args.n_sweep_arg1,
		)

		# bufferlength is based on the minimum of the first swept parameter (speed for many tvb models)
		self.n_work_items, self.n_params = self.params.shape
		self.buf_len_ = ((self.lengths / self.args.speeds_min / self.dt).astype('i').max() + 1)
		self.buf_len = 2 ** np.argwhere(2 ** np.r_[:30] > self.buf_len_)[0][0]  # use next power of

		self.states = self.args.states
		self.exposures = self.args.exposures

		if self.args.gpu_info:
			self.logger.setLevel(level='INFO')
			self.gpu_device_info()
			exit(1)

		self.logdata()

	def logdata(self):

		self.logger.info('dt %f', self.dt)
		self.logger.info('n_nodes %d', self.args.n_regions)
		self.logger.info('weights.shape %s', self.weights.shape)
		self.logger.info('lengths.shape %s', self.lengths.shape)
		self.logger.info('tavg period %s', self.tavg_period)
		self.logger.info('n_inner_steps %s', self.n_inner_steps)
		self.logger.info('params shape %s', self.params.shape)

		self.logger.info('nstep %d', self.args.n_time)
		self.logger.info('n_inner_steps %f', self.n_inner_steps)

		# self.logger.info('single connectome, %d x %d parameter space', self.args.n_sweep_arg0, self.args.n_sweep_arg1)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('number of states %d', self.states)
		self.logger.info('model %s', self.args.model)
		self.logger.info('real buf_len %d, using power of 2 %d', self.buf_len_, self.buf_len)
		self.logger.info('memory for states array on GPU %d MiB',
						 (self.buf_len * self.n_work_items * self.states * self.args.n_regions * 4) / 1024 ** 2)

	def checkargbounds(self):

		try:
			assert self.args.n_sweep_arg0 > 0, "Min value for [N_SWEEP_ARG0] is 1"
			assert self.args.n_time > 0, "Minimum number for [-n N_TIME] is 1"
			assert self.args.n_regions > 0, "Min value for  [-tvbn n_regions] for default data set is 68"
			assert self.args.blockszx > 0 and self.args.blockszx <= 32,	"Bounds for [-bx BLOCKSZX] are 0 < value <= 32"
			assert self.args.blockszy > 0 and self.args.blockszy <= 32, "Bounds for [-by BLOCKSZY] are 0 < value <= 32"
			assert self.args.delta_time > 0.0, "Min value for [-dt delta_time] is > 0.0, default is 0.1"
			assert self.args.speeds_min > 0.0, "Min value for [-sm speeds_min] is > 0.0, default is 3e-3"
			assert self.args.exposures > 0, "Min value for [-x exposures] is 1"
			assert self.args.states > 0, "Min value for [-s states] is 1"
		except AssertionError as e:
			self.logger.error('%s', e)
			raise

	def tvb_connectivity(self, tvbnodes):
		white_matter = connectivity.Connectivity.from_file(source_file="connectivity_"+str(tvbnodes)+".zip")
		# white_matter = connectivity.Connectivity.from_file(source_file="paupau.zip")
		white_matter.configure()
		return white_matter

	def parse_args(self):  # {{{
		parser = argparse.ArgumentParser(description='Run parameter sweep.')

		# for every parameter that needs to be swept, the size can be set
		parser.add_argument('-s0', '--n_sweep_arg0', default=4, help='num grid points for 1st parameter', type=int)
		parser.add_argument('-s1', '--n_sweep_arg1', default=4, help='num grid points for 2st parameter', type=int)
		parser.add_argument('-n', '--n_time', default=400, help='number of time steps to do', type=int)
		parser.add_argument('-v', '--verbose', default=False, help='increase logging verbosity', action='store_true')
		parser.add_argument('-m', '--model', default='epileptor', help="neural mass model to be used during the simulation")
		parser.add_argument('-s', '--states', default=6, type=int, help="number of states for model")
		parser.add_argument('-x', '--exposures', default=2, type=int, help="number of exposures for model")
		parser.add_argument('-l', '--lineinfo', default=False, help='generate line-number information for device code.', action='store_true')
		parser.add_argument('-bx', '--blockszx', default=8, type=int, help="gpu block size x")
		parser.add_argument('-by', '--blockszy', default=8, type=int, help="gpu block size y")
		parser.add_argument('-val', '--validate', default=False, help="enable validation with refmodels", action='store_true')
		parser.add_argument('-r', '--n_regions', default="68", type=int, help="number of tvb nodes")
		parser.add_argument('-p', '--plot_data', type=int, help="plot res data for selected state")
		parser.add_argument('-w', '--write_data', default=False, help="write output data to file: 'tavg_data", action='store_true')
		parser.add_argument('-g', '--gpu_info', default=False, help="show gpu info", action='store_true')
		parser.add_argument('-dt', '--delta_time', default=0.1, type=float, help="dt for simulation")
		parser.add_argument('-sm', '--speeds_min', default=1, type=float, help="min speed for temporal buffer")

		args = parser.parse_args()
		return args


	def setup_params(self,
		n0,
		n1,
		):
		'''
		This code generates the parameters ranges that need to be set
		'''
		sweeparam0 = np.linspace(0.0, 2.0, n0)
		sweeparam1 = np.linspace(1.6, 3.0, n1)
		params = itertools.product(
		sweeparam0,
		sweeparam1,
		)
		params = np.array([vals for vals in params], np.float32)
		return params


	def gpu_device_info(self):
		'''
		Get GPU device information
		TODO use this information to give user GPU setting suggestions
		'''
		dev = drv.Device(0)
		print('\n')
		self.logger.info('GPU = %s', dev.name())
		self.logger.info('TOTAL AVAIL MEMORY: %d MiB', dev.total_memory()/1024/1024)

		# get device information
		att = {'MAX_THREADS_PER_BLOCK': [],
			   'MAX_BLOCK_DIM_X': [],
			   'MAX_BLOCK_DIM_Y': [],
			   'MAX_BLOCK_DIM_Z': [],
			   'MAX_GRID_DIM_X': [],
			   'MAX_GRID_DIM_Y': [],
			   'MAX_GRID_DIM_Z': [],
			   'TOTAL_CONSTANT_MEMORY': [],
			   'WARP_SIZE': [],
			   # 'MAX_PITCH': [],
			   'CLOCK_RATE': [],
			   'TEXTURE_ALIGNMENT': [],
			   # 'GPU_OVERLAP': [],
			   'MULTIPROCESSOR_COUNT': [],
			   'SHARED_MEMORY_PER_BLOCK': [],
			   'MAX_SHARED_MEMORY_PER_BLOCK': [],
			   'REGISTERS_PER_BLOCK': [],
			   'MAX_REGISTERS_PER_BLOCK': []}

		for key in att:
			getstring = 'drv.device_attribute.' + key
			# att[key].append(eval(getstring))
			self.logger.info(key + ': %s', dev.get_attribute(eval(getstring)))

class Driver_Execute(Driver_Setup):

	def __init__(self, ds):
		self.args = ds.args
		self.set_CUDAmodel_dir()
		self.weights, self.lengths, self.params = ds.weights, ds.lengths, ds.params
		self.buf_len, self.states, self.n_work_items = ds.buf_len, ds.states, ds.n_work_items
		self.n_inner_steps, self.n_params, self.dt = ds.n_inner_steps, ds.n_params, ds.dt
		self.exposures, self.logger = ds.exposures, ds.logger

	def set_CUDAmodel_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels", self.args.model.lower() + '.c')

	def set_CUDA_ref_model_dir(self):
		self.args.filename = os.path.join((os.path.dirname(os.path.abspath(__file__))),
								 "../generatedModels/cuda_refs", self.args.model.lower() + '.c')

	def compare_with_ref(self, tavg0):
		self.args.model = self.args.model + 'ref'
		self.set_CUDA_ref_model_dir()
		tavg1 = self.run_simulation()

		# compare output to check if same as template
		# comparison = (tavg0.ravel() == tavg1.ravel())
		# self.logger.info('Templated version is similar to original %d:', comparison.all())
		self.logger.info('Corr.coef. of model with %s is: %f', self.args.model,
						 np.corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1])

		return np.corrcoef(tavg0.ravel(), tavg1.ravel())[0, 1]

	def make_kernel(self, source_file, warp_size, args, lineinfo=False, nh='nh'):

		try:
			with open(source_file, 'r') as fd:
				source = fd.read()
				source = source.replace('pi', '%ff' % (np.pi, ))
				source = source.replace('inf', 'INFINITY')
				opts = ['--ptxas-options=-v', '-maxrregcount=32']
				if lineinfo:
					opts.append('-lineinfo')
				opts.append('-DWARP_SIZE=%d' % (warp_size, ))
				opts.append('-DNH=%s' % (nh, ))

				idirs = [here]
				self.logger.info('nvcc options %r', opts)

				try:
					network_module = SourceModule(
							source, options=opts, include_dirs=idirs,
							no_extern_c=True,
							keep=False,)
				except drv.CompileError as e:
					self.logger.error('Compilation failure \n %s', e)
					exit(1)

				# generic func signature creation
				mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model, 'jjjjjfPfS_S_S_S_')

				step_fn = network_module.get_function(mod_func)

		except FileNotFoundError as e:
			self.logger.error('%s.\n  Generated model filename should match model on cmdline', e)
			exit(1)

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
				self.gpu_mem_info()
				self.logger.error(
					'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
					e, self.params.size)
				exit(1)
		return gpu_data#}}}

	def release_gpumem(self, gpu_data):
		for name, array in gpu_data.items():
			try:
				gpu_data[name].gpudata.free()
			except drv.MemoryError as e:
				self.logger.error('%s.\n\t Freeing mem error', e)
				exit(1)

	def gpu_mem_info(self):

		cmd = "nvidia-smi -q -d MEMORY"#,UTILIZATION"
		os.system(cmd)  # returns the exit code in unix

	def run_simulation(self):

		# setup data#{{{
		data = { 'weights': self.weights, 'lengths': self.lengths, 'params': self.params.T }
		base_shape = self.n_work_items,
		for name, shape in dict(
			tavg0=(self.exposures, self.args.n_regions,),
			tavg1=(self.exposures, self.args.n_regions,),
			state=(self.buf_len, self.states * self.args.n_regions),
			).items():
			# memory error exception for compute device
			try:
				data[name] = np.zeros(shape + base_shape, 'f')
			except MemoryError as e:
				self.logger.error('%s.\n\t Please check the parameter dimensions %d x %d, they are to large '
							 'for this compute device',
							 e, self.args.n_sweep_arg0, self.args.n_sweep_arg1)
				exit(1)

		gpu_data = self.make_gpu_data(data)#{{{

		# setup CUDA stuff#{{{
		step_fn = self.make_kernel(
			source_file=self.args.filename,
			warp_size=32,
			# block_dim_x=self.args.n_sweep_arg0,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			args=self.args,
			lineinfo=self.args.lineinfo,
			nh=self.buf_len,
			)#}}}

		# setup simulation#{{{
		tic = time.time()

		n_streams = 32
		streams = [drv.Stream() for i in range(n_streams)]
		events = [drv.Event() for i in range(n_streams)]
		tavg_unpinned = []

		try:
			tavg = drv.pagelocked_zeros((n_streams,) + data['tavg0'].shape, dtype=np.float32)
		except drv.MemoryError as e:
			self.logger.error(
				'%s.\n\t Please check the parameter dimensions, %d parameters are too large for this GPU',
				e, self.params.size)
			exit(1)

		# determine optimal grid recursively
		def dog(fgd):
			maxgd, mingd = max(fgd), min(fgd)
			maxpos = fgd.index(max(fgd))
			if (maxgd - 1) * mingd * bx * by >= nwi:
				fgd[maxpos] = fgd[maxpos] - 1
				dog(fgd)
			else:
				return fgd

		# n_sweep_arg0 scales griddim.x, n_sweep_arg1 scales griddim.y
		# form an optimal grid recursively
		bx, by = self.args.blockszx, self.args.blockszy
		nwi = self.n_work_items
		rootnwi = int(np.ceil(np.sqrt(nwi)))
		gridx = int(np.ceil(rootnwi / bx))
		gridy = int(np.ceil(rootnwi / by))

		final_block_dim = bx, by, 1

		fgd = [gridx, gridy]
		dog(fgd)
		final_grid_dim = fgd[0], fgd[1]

		assert gridx * gridy * bx * by >= nwi

		self.logger.info('history shape %r', gpu_data['state'].shape)
		self.logger.info('gpu_data %s', gpu_data['tavg0'].shape)
		self.logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))
		self.logger.info('final block dim %r', final_block_dim)
		self.logger.info('final grid dim %r', final_grid_dim)

		# run simulation#{{{
		nstep = self.args.n_time

		self.gpu_mem_info() if self.args.verbose else None

		try:
			for i in tqdm.trange(nstep, file=sys.stdout):

				try:
					event = events[i % n_streams]
					stream = streams[i % n_streams]

					if i > 0:
						stream.wait_for_event(events[(i - 1) % n_streams])

					step_fn(np.uintc(i * self.n_inner_steps), np.uintc(self.args.n_regions), np.uintc(self.buf_len),
							np.uintc(self.n_inner_steps), np.uintc(self.n_work_items), np.float32(self.dt),
							gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
							gpu_data['tavg%d' % (i%2,)],
							block=final_block_dim, grid=final_grid_dim)

					event.record(streams[i % n_streams])
				except drv.LaunchError as e:
					self.logger.error('%s', e)
					exit(1)

				tavgk = 'tavg%d' % ((i + 1) % 2,)

				# async wrt. other streams & host, but not this stream.
				if i >= n_streams:
					stream.synchronize()
					tavg_unpinned.append(tavg[i % n_streams].copy())

				drv.memcpy_dtoh_async(tavg[i % n_streams], gpu_data[tavgk].ptr, stream=stream)

			# recover uncopied data from pinned buffer
			if nstep > n_streams:
				for i in range(nstep % n_streams, n_streams):
					stream.synchronize()
					tavg_unpinned.append(tavg[i].copy())

			for i in range(nstep % n_streams):
				stream.synchronize()
				tavg_unpinned.append(tavg[i].copy())

		except drv.LogicError as e:
			self.logger.error('%s. Check the number of states of the model or '
						 'GPU block shape settings blockdim.x/y %r, griddim %r.',
						 e, final_block_dim, final_grid_dim)
			exit(1)
		except drv.RuntimeError as e:
			self.logger.error('%s', e)
			exit(1)


		# self.logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)

		# also release gpu_data
		self.release_gpumem(gpu_data)

		self.logger.info('kernel finished')
		return tavg

	def plot_output(self, tavg):
		plt.plot((tavg[:, self.args.plot_data, :, 0]), 'k', alpha=.2)
		plt.show()

	def write_output(self, tavg):
		tavg_file = open('tavg_data', 'wb')
		pickle.dump(tavg, tavg_file)
		tavg_file.close()

	def run_all(self):

		np.random.seed(79)

		tic = time.time()

		tavg0 = self.run_simulation()
		toc = time.time()
		elapsed = toc - tic

		if (self.args.validate == True):
			self.compare_with_ref(tavg0)

		self.plot_output(tavg0) if self.args.plot_data is not None else None
		self.write_output(tavg0) if self.args.write_data else None
		self.logger.info('Output shape (simsteps, states, bnodes, n_params) %s', tavg0.shape)
		self.logger.info('Finished CUDA simulation successfully in: {0:.3f}'.format(elapsed))
		self.logger.info('and in {0:.3f} M step/s'.format(
			1e-6 * self.args.n_time * self.n_inner_steps * self.n_work_items / elapsed))

		return tavg0


if __name__ == '__main__':

	driver_setup = Driver_Setup()
	tavgGPU = Driver_Execute(driver_setup).run_all()

	simtime = driver_setup.args.n_time
	# simtime = 10
	regions = driver_setup.args.n_regions
	g = 1.0
	# g = 0.0042
	s = 1.0
	dt = driver_setup.dt
	period = 1

	# generic model definition
	model = driver_setup.args.model.capitalize()+'T'

	# non generic model names
	# model = 'MontbrioT'
	# model = 'RwongwangT'
	# model = 'OscillatorT'
	# model = 'DumontGutkin'
	# model = 'MontbrioPazoRoxin'
	# model='Generic2dOscillator'
	(time, tavgCPU) = regularRun(simtime, g, s, dt, period).simulate_python(model)

	print('CPUshape', tavgCPU.shape)
	print('GPUshape', tavgGPU.shape)

	# check for deviation tolerance between GPU and CPU
	# for basic coupling and period = 1
	# using euler deterministic solver
	max_err = []
	x = 0
	for t in range(0, simtime):
		# print(t, 'tol:', np.max(np.abs(actual[t] - expected[t, :, :, 0])))
		# print(t, 'tol:', np.max(np.abs(tavgCPU[t,:,:,0], tavgGPU[t,:,:,0])))
		# print(t)
		# print('C', tavgCPU[t,:,:,0])
		# print('G', tavgGPU[t,:,:,0])
		# print(t, 'tol:', np.max(np.abs(tavgCPU[t,:,:,0] - tavgGPU[t,:,:,0])))
		np.testing.assert_allclose(tavgCPU[t, :, :, 0], tavgGPU[t, :, :, 0], 2e-5 * t * 2, 1e-5 * t * 2)