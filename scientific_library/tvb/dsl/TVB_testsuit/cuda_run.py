#!/usr/bin/env python3

from __future__ import print_function
import sys
import numpy as np
import os.path
import numpy as np
import itertools
import pycuda.autoinit
import pycuda.driver as drv
from pycuda.compiler import SourceModule
import pycuda.gpuarray as gpuarray
import pytools
import time
import argparse
import logging
import scipy.io as io

here = os.path.dirname(os.path.abspath(__file__))

class CudaRun:

	def make_kernel(self, source_file, warp_size, block_dim_x, lineinfo=False, nh='nh'):
		with open(source_file, 'r') as fd:
			source = fd.read()
			source = source.replace('M_PI_F', '%ff' % (np.pi, ))
			opts = ['--ptxas-options=-v', ]# '-maxrregcount=32']# '-lineinfo']
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
			# TODO parse verbose output of nvcc to get function name

		step_fn = network_module.get_function('_Z9integratejjjjjffPfS_S_S_S_')


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


	def run_simulation(self, weights, lengths, params_matrix, couplings, speeds, logger, args, n_nodes, n_work_items, n_params, nstep, n_inner_steps, 
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
		logger.info('history shape %r', data['state'].shape)
		logger.info('on device mem: %.3f MiB' % (self.nbytes(data) / 1024 / 1024, ))#}}}

		# setup CUDA stuff#{{{
		step_fn = self.make_kernel(
			source_file=args.filename,
			warp_size=32,
			block_dim_x=args.n_coupling,
			# ext_options=preproccesor_defines,
			# caching=args.caching,
			lineinfo=args.lineinfo,
			nh=buf_len,
			# model=args.model,
			)#}}}

		# setup simulation#{{{
		tic = time.time()
		logger.info('nstep %i', nstep)
		streams = [drv.Stream() for i in range(32)]
		events = [drv.Event() for i in range(32)]
		tavg_unpinned = []
		tavg = drv.pagelocked_zeros(data['tavg'].shape, dtype=np.float32)
		#}}}

		# adjust gridDim to keep block size <= 1024 {{{
		block_size_lim = 1024
		n_coupling_per_block = block_size_lim // args.node_threads
		n_coupling_blocks = args.n_coupling // n_coupling_per_block
		if n_coupling_blocks == 0:
			n_coupling_per_block = args.n_coupling
			n_coupling_blocks = 1
		final_block_dim = n_coupling_per_block, args.node_threads, 1
		final_grid_dim = speeds.size, n_coupling_blocks
		logger.info('final block dim %r', final_block_dim)
		logger.info('final grid dim %r', final_grid_dim)
		assert n_coupling_per_block * n_coupling_blocks == args.n_coupling #}}}
		logger.info('gpu_data[lengts] %s', gpu_data['lengths'].shape)
		logger.info('nnodes %r', n_nodes)
		# logger.info('gpu_data[lengths] %r', gpu_data['lengths'])

		# run simulation#{{{
		logger.info('submitting work')
		for i in range(nstep):

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

		logger.info('kernel finish..')
		# release pinned memory
		tavg = np.array(tavg_unpinned)
		return tavg
