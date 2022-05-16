from tvb.simulator.lab import *

import numpy as np
import math

import matplotlib.pyplot as plt

from tvb.rateML.XML2model import RateML

class regularRun:

	def __init__(self, sim_length, g, s, dt, period, omega = 60, filename='connectivity_zerlaut_68.zip'):
	# def __init__(self, sim_length, g, s, dt, period, omega = 60, filename='paupau.zip'):
		self.sim_length = sim_length
		self.g = np.array([g])
		self.s = np.array([s])
		self.dt = dt
		self.period = period
		self.omega = omega * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(filename)
		self.SC = self.connectivity.weights
		

	def tvb_connectivity(self, filename):
		white_matter = connectivity.Connectivity.from_file(source_file=filename)
		white_matter.configure()
		white_matter.speed = np.array([self.s])
		# white_matter_coupling = coupling.Linear(a=self.g)
		# white_matter_coupling = coupling.Scaling(a=self.g)
		white_matter_coupling = coupling.Coupling()
		return white_matter, white_matter_coupling
	
	def tvb_python_model(self, modelExec):
		# populations = models.Generic2dOscillator()	# original
		# populations = models.KuramotoT()				# generated
		# populations = models.OscillatorT()			# generated
		# populations = models.MontbrioT()				# generated
		# populations = models.RwongwangT()				# generated
		# populations = models.EpileptorT()				# generated
		model = 'models.' + modelExec + '()'
		populations = eval(model)
		populations.configure()
		populations.omega = np.array([self.omega])
		return populations


	def simulate_python(self, modelExec):
		# Initialize Model
		model = self.tvb_python_model(modelExec)
		# zerlaut setup
		noises = noise.Additive(nsig=np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]), ntau=0.0)
		integrator = integrators.HeunStochastic(dt=.1, noise=noises)
		# Initialize integrator
		# Initialize Monitors
		monitorsen = (monitors.TemporalAverage(period=self.period))
		# Initialize Simulator
		sim = simulator.Simulator(model=model, connectivity=self.connectivity,
								  coupling=coupling.Linear(a=np.array(self.g), b=np.array(0.0)),
								  integrator=integrator,
								  monitors=[monitorsen])
		sim.configure()
		# sim.history.buffer[:] = 0.0
		# sim.current_state[:] = 0.0
		# print('shb', sim.history.buffer.shape)  # ('n_time', 'n_cvar', 'n_node', 'n_mode')

		(time, data) = sim.run(simulation_length=self.sim_length)[0]

		# pad some zeros to make it equivalent to GPU for comparison
		# CPU is now +1 timestep longer
		# data = np.insert(data, 0, 0, axis=0)
		# data = data[:-1]

		# print('ds',data.shape)
		plt.plot((data[:, 0, :, 0]), 'k', alpha=.2)
		plt.show()

		return (time, data)

if __name__ == '__main__':

	# model_filename = 'Oscillator'
	# language='python'
	#
	# RateML(model_filename, language)

	simtime = 2000
	g = .4
	# g = 0.0042
	s = 4.0
	dt = .1
	period = 10

	model = 'Zerlaut_adaptation_second_order'

	(time, data) = regularRun(simtime, g, s, dt, period).simulate_python(model)