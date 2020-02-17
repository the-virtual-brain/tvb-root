from __future__ import division, print_function
import math as m
import numpy as _lpy_np
import numba.cuda as _lpy_ncu
from numba.cuda.random import create_xoroshiro128p_states, xoroshiro128p_uniform_float32
import numba as _lpy_numba
from tvb_hpc import utils, network
from typing import List

# from tvb.simulator.lab import *

import logging

LOG = utils.getLogger('tvb_hpc')


@_lpy_ncu.jit#(debug=True)
def Kuramoto_and_Network_and_EulerStep_inner(i_step, n_nodes, buf_len, n_step, dt, weights, lengths, params, states, obsrv, rng_states):

	# work id & size
	thread_id = (_lpy_ncu.blockIdx.y * _lpy_ncu.gridDim.x + _lpy_ncu.blockIdx.x) * _lpy_ncu.blockDim.x + _lpy_ncu.threadIdx.x
	size = _lpy_ncu.blockDim.x * _lpy_ncu.gridDim.x * _lpy_ncu.gridDim.y

	# print('size',size)

	# unpack params 2d
	coupling = params[1][thread_id]
	speed = params[0][thread_id]

	# derived
	rec_n = 1.0 / n_nodes
	rec_speed_dt = 10 / speed / dt
	omega = 10.0 * 2.0 * _lpy_np.pi / 1e3
	sig = m.sqrt(dt) * m.sqrt(2.0 * 1e-5)

	#noise init here

	for time_step in range(i_step, (i_step + n_step)):
		for i_node in range(_lpy_ncu.threadIdx.y, n_nodes, _lpy_ncu.blockDim.y):
		# for i_node in range(0, n_nodes):
			if (i_node >= n_nodes):
				continue
			index = (((time_step) % buf_len) * n_nodes + i_node)*size + thread_id
			theta_i = states[index]
			# i_n = i_node * n_nodes
			sumCoup = 0.0

			# print('theta_i', time_step, theta_i)

			for j_node in range(0, n_nodes):
				wij = weights[i_node][j_node]
				if (wij == 0.0):
					continue
				# if (i_node<0 or i_node>=84):
				# 	print(i_node)
				# if (j_node<0 or j_node>=84):
				# 	print(j_node)

				dij = int(lengths[i_node][j_node] * rec_speed_dt)
				# theta_j = states[(((time_step - dij + buf_len) % buf_len) * n_nodes + j_node)*size + thread_id]
				
				index2=(((time_step - dij + buf_len) % buf_len) * n_nodes + j_node)*size + thread_id
			# 	if (index2<0 or index2>=22020098):
			# 		print(index2)

				theta_j = states[index2]
			# 	# sumCoup = sumCoup + m.sin((states[index2]) - (theta_i))
				sumCoup += (wij * m.sin(theta_j - theta_i))
				# print('sumCoup',sumCoup)
 	
			# input_tmp = sumCoup * rec_n * sumCoup
			# drift_tmp = omega + input_tmp
			# state_tmp = theta_tmp + theta_i
			state_tmp = theta_i + dt * (omega + coupling * rec_n * sumCoup)
			# # theta_i += sig * rng_next_normal(&rng); // noise(time_step - i_step, i_node)

			state_tmp = sig * xoroshiro128p_uniform_float32(rng_states, thread_id)

			# # theta_tmp = (theta_i < 0)*(theta_i + 6.283185307179586) + (theta_i > 6.283185307179586)*(theta_i + -6.283185307179586) + (theta_i >= 0)*(theta_i <= 6.283185307179586)*theta_i
			# # state_tmp = drift_tmp + theta_i
			wrapd_tmp =( (state_tmp <  0)*(state_tmp + 2*_lpy_np.pi)
					   + (state_tmp >  2 * _lpy_np.pi)*(state_tmp - 2 * _lpy_np.pi) 
					   + (state_tmp >= 0)*(state_tmp <= 2 * _lpy_np.pi)*state_tmp)
			# theta = 1.0
			# theta = wrapd_tmp

			# print(state_tmp)
			# states[0] = wrapd_tmp
			index3 = (((time_step+1) % buf_len) * n_nodes + i_node)*size + thread_id
			# if (index3<0 or index3>=22020096):
			# 	print(index3)
			# wrapd_tmp = theta_i + 1
			# print('wrapd_tmp', time_step, wrapd_tmp)
			states[index3] = wrapd_tmp
			# obsrv[i_node * size + id] += m.sin(theta_i)
			obsrv[i_node] += m.sin(theta_i)



class NumbaCudaRun:


	def run_simulation(self, blockspergrid, threadsperblock, n_inner_steps, n_nodes, buf_len, dt, weights, lengths, params, logger):
		_lpy_ncu.select_device(0)
		logging.getLogger("numba").setLevel(logging.WARNING)
		LOG.info(_lpy_ncu.gpus)

		# blockspergrid = 32
		# threadsperblock = 32
		# n_inner_steps = 10
		# n_nodes = 84
		# buf_len = 256
		
		states = _lpy_np.zeros((blockspergrid * threadsperblock * n_nodes * buf_len), _lpy_np.float32)
		# d_states = _lpy_ncu.to_device(states)
		obsrv = _lpy_np.zeros((blockspergrid * threadsperblock * n_nodes), _lpy_np.float32)

		logger.info('state size %d', states.size)
		logger.info('obsrv size %d', obsrv.size)

		rng_states = create_xoroshiro128p_states(threadsperblock * blockspergrid, seed=1)

		for i in range(400):
			Kuramoto_and_Network_and_EulerStep_inner[blockspergrid, threadsperblock](i * n_inner_steps, n_nodes, buf_len, n_inner_steps, dt, 
				weights, lengths, params, states, obsrv, rng_states)	

		return obsrv