#!/usr/bin/env python3

from __future__ import print_function
import os.path
import numpy as np
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import time

here = os.path.dirname(os.path.abspath(__file__))

class CudaRun:

	def make_kernel(self, source_file, warp_size, block_dim_x, args, lineinfo=False, nh='nh'):
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
			network_module = SourceModule(
					source, options=opts, include_dirs=idirs,
					no_extern_c=True,
					keep=False,
			)
			# no API to know the mangled function name ahead of time
			# if the func sig changes, just copy-paste the new name here..
			# TODO parse verbose output of nvcc to get function name and make dynamic

		# mod_func = '_Z9EpileptorjjjjjffPfS_S_S_S_'
		# mod_func = '_Z8KuramotojjjjjffPfS_S_S_S_'
		# mod_func = '_Z9RwongwangjjjjjffPfS_S_S_S_'
		# mod_func = '_Z12KuratmotorefjjjjjffPfS_S_S_S_'
		mod_func = "{}{}{}{}".format('_Z', len(args.model), args.model.capitalize(), 'jjjjjffPfS_S_S_S_')

		step_fn = network_module.get_function(mod_func)

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
			gpu_data[name] = gpuarray.to_gpu(self.cf(array))
		return gpu_data#}}}

	def gpu_info(self):
		cmd = "nvidia-smi -q -d MEMORY,UTILIZATION"
		returned_value = os.system(cmd)  # returns the exit code in unix
		print('returned value:', returned_value)

	def run_simulation(self, weights, lengths, params_matrix, speeds, logger, args, n_nodes, n_work_items, n_params, nstep, n_inner_steps,
		buf_len, states, dt, min_speed):

		# setup data#{{{
		data = { 'weights': weights, 'lengths': lengths, 'params': params_matrix.T }
		base_shape = n_work_items,
		for name, shape in dict(
			tavg=(n_nodes,),
			state=(buf_len, states * n_nodes),
			).items():
			data[name] = np.zeros(shape + base_shape, 'f')

		gpu_data = self.make_gpu_data(data)#{{{
		# logger.info('history shape %r', data['state'].shape)
		logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))#}}}

		# setup CUDA stuff#{{{
		step_fn = self.make_kernel(
			source_file=args.filename,
			warp_size=32,
			block_dim_x=args.n_coupling,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			args=args,
			lineinfo=args.lineinfo,
			nh=buf_len,
			# model=args.model,
			)#}}}

		# setup simulation#{{{
		tic = time.time()
		# logger.info('nstep %i', nstep)
		streams = [drv.Stream() for i in range(32)]
		events = [drv.Event() for i in range(32)]
		tavg_unpinned = []
		tavg = drv.pagelocked_zeros(data['tavg'].shape, dtype=np.float32)
		# logger.info('data[tavg].shape %s', data['tavg'].shape)
		#}}}

		gridx = args.n_coupling // args.blockszx
		gridy = args.n_speed // args.blockszy
		final_block_dim = args.blockszx, args.blockszy, 1
		final_grid_dim = gridx, gridy

		# logger.info('final block dim %r', final_block_dim)
		logger.info('final grid dim %r', final_grid_dim)
		# assert n_coupling_per_block * n_coupling_blocks == args.n_coupling #}}}

		# logger.info('gpu_data[lengts] %s', gpu_data['lengths'].shape)
		# logger.info('nnodes %r', n_nodes)
		# logger.info('gpu_data[lengths] %r', gpu_data['lengths'])

		# run simulation#{{{
		# logger.info('submitting work')
		import tqdm
		for i in tqdm.trange(nstep):

			# event = events[i % 32]
			# stream = streams[i % 32]

			# stream.wait_for_event(events[(i - 1) % 32])

			step_fn(np.uintc(i * n_inner_steps), np.uintc(n_nodes), np.uintc(buf_len), np.uintc(n_inner_steps),
					np.uintc(n_params), np.float32(dt), np.float32(min_speed),
					gpu_data['weights'], gpu_data['lengths'], gpu_data['params'], gpu_data['state'],
					gpu_data['tavg'],
					block=final_block_dim,
					grid=final_grid_dim)

			# event.record(streams[i % 32])
			tavg_unpinned.append(tavg.copy())
			drv.memcpy_dtoh(
				tavg,
				gpu_data['tavg'].ptr)

		# logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)
		return tavg
