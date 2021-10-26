#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 24 10:29:52 2021

@author: nuuria
"""

import os

class Parameter :
    def __init__(self):
        path = os.path.dirname(os.path.abspath(__file__))
        self.parameter_simulation={
            'path_result':'./result/synch/',
            'seed':10, # the seed for the random generator
            'save_time': 1000.0, # the time of simulation in each file
        }

        self.parameter_model ={
            'matteo':False,
            #order of the model
            'order':2,
            #parameter of the model
            'g_L':10.0,
            'E_L_e':-63.0,
            'E_L_i':-65.0,
            'C_m':200.0,
            'b_e':60.0,
            'a_e':0.0,
            'b_i':0.0,
            'a_i':0.0,
            'tau_w_e':500.0,
            'tau_w_i':1.0,
            'E_e':0.0,
            'E_i':-80.0,
            'Q_e':1.5,
            'Q_i':5.0,
            'tau_e':5.0,
            'tau_i':5.0,
            'N_tot':10000,
            'p_connect_e':0.05,
            'p_connect_i':0.05,
            'g':0.2,
            'T':40.0,
            'P_e':[-0.0498, 0.00506, -0.025, 0.0014, -0.00041, 0.0105, -0.036, 0.0074, 0.0012, -0.0407],
            'P_i':[-0.0514, 0.004, -0.0083, 0.0002, -0.0005, 0.0014, -0.0146, 0.0045, 0.0028, -0.0153],
            'external_input_ex_ex':0.315*1e-3,
            'external_input_ex_in':0.000,
            'external_input_in_ex':0.315*1e-3,
            'external_input_in_in':0.000,
            'tau_OU':5.0,
            'weight_noise': 1e-4, #10.5*1e-5,
            'K_ext_e':400,
            'K_ext_i':0,
            #Initial condition :
            'initial_condition':{
                "E": [0.000, 0.000],"I": [0.00, 0.00],"C_ee": [0.0,0.0],"C_ei": [0.0,0.0],"C_ii": [0.0,0.0],"W_e": [100.0, 100.0],"W_i": [0.0,0.0],"noise":[0.0,0.0]} # W_e changed from [100.0,100.0] to [0.0,0.0] (15/9/2021)
        }

        self.parameter_connection_between_region={
            ## CONNECTIVITY
            # connectivity by default
            'default':False,
            #from file (repertory with following files : tract_lengths.npy and weights.npy)
            'from_file':False,
            'from_h5':True,
            'from_folder':True,
            'path':path+'/../../data/Mouse_512/Connectivity_nuria_v2_sym_weights.h5', #The file "Connectivity_nuria_v2_sym_weights.h5" has been modified from the original ("Connectivity_nuria_v1.h5") anatomic information, so that the connectivity weights are symmetric
            'number_of_regions':512, # number of regions
            # lenghts of tract between region : dimension => (number_of_regions, number_of_regions)
            'tract_lengths':[],
            # weight along the tract : dimension => (number_of_regions, number_of_regions)
            'weights':[],
            # speed of along long range connection
            'speed':3.0,
            'normalised':True
        }

        self.parameter_coupling={
            ##COUPLING
            'type':'Linear', # choice : Linear, Scaling, HyperbolicTangent, Sigmoidal, SigmoidalJansenRit, PreSigmoidal, Difference, Kuramoto
            'parameter':{'a':0.25, # changed from 0.45 to 0.25 (14/09/2021) then to 0.20 (15/09/2021)
                         'b':0.0}
        }

        self.parameter_integrator={
            ## INTEGRATOR
            'type':'Heun', # choice : Heun, Euler
            'stochastic':True,
            'noise_type': 'Additive', #'Multiplicative', #'Additive', # choice : Additive
            'noise_parameter':{
                'nsig':[0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
                'ntau':0.0,
                'dt': 0.1
                                },
            'dt': 0.1 # in ms
        }

        self.parameter_monitor= {
            'Raw':True,
            'TemporalAverage':False,
            'parameter_TemporalAverage':{
                'variables_of_interest':[0,1,2,3,4,5,6,7],
                'period':self.parameter_integrator['dt']*10.0
            },
            'Bold':False,
            'parameter_Bold':{
                'variables_of_interest':[0],
                'period':self.parameter_integrator['dt']*2000.0
            },
            'Ca':False,
            'parameter_Ca':{
                'variables_of_interest':[0,1,2],
                'tau_rise':0.01,
                'tau_decay':0.1
            }
        }


        self.parameter_stimulus = {
            'onset': 99.0,
            "tau": 9.0,
            "T": 99.0,
            "weights": None,
            "variables":[0]
        }

        self.parameter_surface = {
            'run_surface' : True,
            'load_surface': False,
            'h5_filename' : 'Connectivity_nuria_v1.h5',
            'zip_filename': 'Cortex.zip',
            'vertices_to_region_filename' : 'vertices_to_region.npy',
            'region_map_filename' : 'region_map.txt'
        }
