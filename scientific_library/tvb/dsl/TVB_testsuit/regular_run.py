from tvb.simulator.lab import *

import numpy as np
import numpy.random as rgn
import matplotlib.pyplot as plt
import math

from numpy import corrcoef
import seaborn as sns


class regularRun:

	def __init__(self):
		self.sim_length = 400
		self.g = np.array([1.0])
		self.s = np.array([1.0])
		self.dt = 0.1
		self.period = 10.0
		self.omega = 60.0 * 2.0 * math.pi / 1e3
		(self.connectivity, self.coupling) = self.tvb_connectivity(self.s, self.g, self.dt)
		self.SC = self.connectivity.weights
		self.integrator = integrators.EulerDeterministic(dt=self.dt)

	def tvb_connectivity(self, speed, global_coupling, dt=0.1):
		white_matter = connectivity.Connectivity.from_file(source_file="data/connectivity_68.zip")
		white_matter.configure()
		white_matter.speed = np.array([speed])
		white_matter_coupling = coupling.Linear(a=global_coupling)
		return white_matter, white_matter_coupling
	
	def tvb_python_model(self):
		populations = models.Kuramoto()
		populations.configure()
		populations.omega = np.array([self.omega])
		return populations


	def simulate_python(self, logger, args):
		# Initialize Model
		model = self.tvb_python_model()
		# Initialize Monitors
		monitorsen = (monitors.TemporalAverage(period=self.period))
		# Initialize Simulator
		sim = simulator.Simulator(model=model, connectivity=self.connectivity, coupling=self.coupling, integrator=self.integrator,
									  monitors=[monitorsen])
		sim.configure()
		tavg_data = sim.run(simulation_length=self.sim_length)

		#set data for check_results
		n_nodes = self.SC.shape[0]
		n_work_items = self.sim_length
		weights = self.SC
		speeds = self.connectivity.speed
		couplings = self.coupling
		logger.info('n_nodes %d', n_nodes)
		logger.info('n_work_items %d', n_work_items)
		# logger.info('speeds %d', speeds)
		# logger.info('couplings %d', couplings)

		# self.check_results(n_nodes, n_work_items, np.squeeze(tavg_data).shape, weights, speeds, couplings, logger, args)

		# FC = self.calculate_FC(np.squeeze(tavg_data))
		# r = self.plot_SC_FC(SC, FC,"python")
		FC, r = 0, 0
		return FC, r