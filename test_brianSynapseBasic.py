#!/bin/python3

##########################################
### Tests for brianSynapseBasic module ###
##########################################

### Copyright 2023 Jannik Luboeinski
### licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
### Contact: mail[at]jlubo.net

#########################################################
### Imports and initialization ###

import brianSynapseBasic as bsb
import numpy as np
import json
import os
import inspect
import pytest

epsilon = 6e-7 # similar to default tolerance value used by pytest.approx()

###############################################################################	
# Test basic protocols for early- and late-phase plasticity; test the recording switches as well
@pytest.mark.parametrize("paradigm, record_spikes, record_neuron_traces, config_file", 
                         [("early", True, True, "config_basic_early.json"),
                          ("early", True, False, "config_basic_early.json"),
                          ("late", False, True, "config_basic_late.json")])
def test_basic(paradigm, record_spikes, record_neuron_traces, config_file):

	# configuration
	config = json.load(open(config_file, "r")) # load JSON file
	s_desc = config["simulation"]["short_description"] # short description of the simulation
	data_dir = os.path.join("testing", s_desc)
	os.makedirs(data_dir, exist_ok=True) # create data directory
	dt = config["simulation"]["dt"] # timestep (in ms)
	op = config["simulation"]["output_period"] # sampling period (in timesteps) - should be 1 for early-phase and >>1 for late-phase test
	stim_prot = config["simulation"]["learn_protocol"] # stimulation protocol
	func_name = inspect.getframeinfo(inspect.currentframe()).function

	# run simulation
	data_path_trial = bsb.simulate(config_file, data_dir, record_spikes = record_spikes, record_neuron_traces = record_neuron_traces)

	# retrieve the data
	data_stacked = np.loadtxt(os.path.join(data_path_trial, "traces.txt"))

	# early phase: test the plasticity induced by some spikes evoked at determined times
	if paradigm == "early" and record_neuron_traces:

		n_stim = len(stim_prot["explicit_input"]["stim_times"]) # the number of stimulus pulses
		tb_before_last_spike = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["t_ax_delay"])/dt/op) # timestep right before the onset of the PSP caused by the last stimulus
		tb_before_last_Ca_increase = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["calcium"]["t_Ca_delay"])/dt/op) # timestep right before the presynapse-induced calcium increase caused by the last stimulus

		#assert data_stacked[tb_before_last_spike+2][2] == pytest.approx(0) # upon PSP onset: stimulation current equal zero (again) # TODO currently value is not provided
		assert data_stacked[tb_before_last_spike+2][1] > config["neuron"]["V_rev"] + epsilon # upon PSP onset: membrane potential greater than V_rev
		assert data_stacked[tb_before_last_Ca_increase+2][3] > config["synapses"]["h_0"] + epsilon # early-phase weight greater than h_0
		assert data_stacked[tb_before_last_Ca_increase+2][5] + epsilon < config["synapses"]["early_phase"]["theta_p"] # calcium amount less than theta_p
		assert data_stacked[tb_before_last_Ca_increase+2][5] > config["synapses"]["early_phase"]["theta_d"] + epsilon # calcium amount greater than theta_d
		assert data_stacked[tb_before_last_Ca_increase+2][6] == pytest.approx(0) # protein amount equal to zero
		assert data_stacked[tb_before_last_Ca_increase+2][4] == pytest.approx(0) # late-phase weight equal to zero

	# early phase: test the plasticity induced by some spikes evoked at determined times
	elif paradigm == "early" and not record_neuron_traces:

		n_stim = len(stim_prot["explicit_input"]["stim_times"]) # the number of stimulus pulses
		tb_before_last_spike = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["t_ax_delay"])/dt/op) # timestep right before the onset of the PSP caused by the last stimulus
		tb_before_last_Ca_increase = int((stim_prot["explicit_input"]["stim_times"][n_stim-1] + config["synapses"]["calcium"]["t_Ca_delay"])/dt/op) # timestep right before the presynapse-induced calcium increase caused by the last stimulus

		assert data_stacked[tb_before_last_Ca_increase+2][1] > config["synapses"]["h_0"] + epsilon # early-phase weight greater than h_0
		assert data_stacked[tb_before_last_Ca_increase+2][3] + epsilon < config["synapses"]["early_phase"]["theta_p"] # calcium amount less than theta_p
		assert data_stacked[tb_before_last_Ca_increase+2][3] > config["synapses"]["early_phase"]["theta_d"] + epsilon # calcium amount greater than theta_d
		assert data_stacked[tb_before_last_Ca_increase+2][2] == pytest.approx(0) # late-phase weight equal to zero

	# late phase: test the plasticity induced by strong stimulation
	elif paradigm == "late":

		tb_before_stim_end = int((stim_prot["time_start"] + stim_prot["duration"])*1000/dt/op) # timestep right before the end of the stimulation

		tau_h = config["synapses"]["early_phase"]["tau_h"]
		theta_pro = config["synapses"]["protein"]["theta_pro"]
		h_stim_end = data_stacked[tb_before_stim_end][3] # early-phase weight upon the end of the stimulation

		tb_before_ps_end = tb_before_stim_end + int(tau_h / 0.1 * np.log((h_stim_end - config["synapses"]["h_0"]) / theta_pro)/dt/op) # timestep right before the protein synthesis ends (in case there are no fast-forward timesteps)

		assert data_stacked[tb_before_stim_end][5] > config["synapses"]["early_phase"]["theta_p"] + epsilon # calcium amount greater than theta_p -- at the end of the stimulation
		assert h_stim_end > 1.8*config["synapses"]["h_0"] # early-phase weight built up -- at the end of the stimulation
		assert data_stacked[-1][4] > 0.9 # late-phase weight built up -- at the end of the whole simulation
		assert data_stacked[tb_before_ps_end][6] == pytest.approx(1, 0.1) # protein amount equal to one -- at the end of the protein synthesis

