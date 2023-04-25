#!/bin/python3

#############################################################################################################
### Brian 2 simulation of a single current-based synapse connecting two leaky integrate-and-fire neurons. ###
### Depending on the neuronal activities, the synapse undergoes calcium-based early-phase plasticity and  ###
### possibly late-phase plasticity, which is described by synaptic tagging and capture.                   ###
#############################################################################################################

### Copyright 2022-2023 Jannik Luboeinski
### licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
### Contact: mail[at]jlubo.net

#########################################################
### Imports and initialization ###

import brian2 as b
from brian2.units import msecond, second, mvolt, namp, ncoulomb, Mohm, farad, hertz
from matplotlib.ticker import MultipleLocator, FormatStrFormatter
import numpy as np
import json
import os
import re
import time
from datetime import datetime
#%matplotlib inline
b.set_device('cpp_standalone')

###############################################################################
# waitGetTimestamp
# Waits for two seconds and then returns the timestamp of the current point in time, or returns a previously determined timestamp
# - refresh [optional]: if True, forcibly retrieves a new timestamp; else, only returns a new timestamp if no previous one is known
# - return: timestamp in the format YY-MM-DD_HH-MM-SS
def waitGetTimestamp(refresh = False):
	global timestamp_var # make this variable static
		
	if refresh == True or not timestamp_var in globals():
		time.sleep(2)
		timestamp_var = datetime.now() # get new timestamp
		
	return timestamp_var.strftime("%y-%m-%d_%H-%M-%S")

###############################################################################
# writeLog
# Writes string(s) to the global log file 'logf' and prints to the console
# - ostrs: the string(s) to be written/printed
# - prnt [optional]: specifies whether to print to console or not
def writeLog(*ostrs, prnt = True):

	for i in range(len(ostrs)):
		ostr = str(ostrs[i])
		ostr = re.sub(r'\x1b\[[0-9]*m', '', ostr) # remove console formatting
		if i == 0:
			logf.write(ostr)
		else:
			logf.write(" " + ostr)
	logf.write("\n")
	
	if prnt:
		print(*ostrs)

#########################################################
# simulate
# Simulates a single neuron with a single input synapse 
# (to be used with basic protocol to induce early-phase potentiation)
# - config: configuration of model and simulation parameters (as a dictionary from JSON format)
# - data_dir: name of the data directory
# - record_spikes [optional]: specifies if spikes should be recorded
# - record_neuron_traces [optional]: specifies if traces of neuronal (excluding synaptic) variables should be recorded
# - return: the path of the trial's data
def simulate(config, data_dir, record_spikes = False, record_neuron_traces = True):

	#########################################################
	### Initialization ###
	b.device.reinit()
	b.device.activate()

	data_path_trial = os.path.join(data_dir, waitGetTimestamp(refresh = True) + "_data")
	if not os.path.isdir(data_path_trial):
		os.mkdir(data_path_trial)

	global logf # global handle to the log file (to need less code for output commands)
	logf = open(os.path.join(data_path_trial, "log.txt"), "w")

	writeLog("Brian version", b.__version__)
	b.start_scope()

	#########################################################
	### Parameters ###
	# Simulation parameters
	t_max = config["simulation"]["runtime"]*second # biological duration of the simulation
	delta_t = config["simulation"]["dt"]*msecond # duration of one timestep for simulating
	delta_t_sample = config["simulation"]["output_period"]*config["simulation"]["dt"]*msecond # duration of one timestep for sampling
	
	t_stim_start = config["simulation"]["learn_protocol"]["time_start"]*second # beginning of fluctuating input stimulation
	t_stim_end = t_stim_start + config["simulation"]["learn_protocol"]["duration"]*second # end of fluctuating input stimulation
	f_stim = config["simulation"]["learn_protocol"]["freq"]*hertz # mean firing rate of the putative input population
	N_stim = config["simulation"]["learn_protocol"]["N_stim"] # number of neurons in the putative input population

	explicit_stim_receivers = config["simulation"]["learn_protocol"]["explicit_input"]["receivers"] # neurons receiving explicit stimulus pulses
	explicit_stim_times = config["simulation"]["learn_protocol"]["explicit_input"]["stim_times"] # timing of explicit stimulus pulses

	# Neuron parameters
	R_mem = config["neuron"]["R_leak"]*Mohm
	tau_mem = R_mem * config["neuron"]["C_mem"]*farad
	V_rev = config["neuron"]["V_rev"]*mvolt
	t_ref = config["neuron"]["t_ref"]*msecond
	V_reset = config["neuron"]["V_reset"]*mvolt
	V_th = config["neuron"]["V_th"]*mvolt
	I_0 = config["simulation"]["bg_protocol"]["I_0"]*namp
	sigma_wn = config["simulation"]["bg_protocol"]["sigma_WN"]*namp*second**(1/2)
	w_spike = 100*(V_th-V_reset)*np.exp(delta_t/tau_mem) # value that is sufficently large to cause spike in postsyn. neuron

	# Synapse parameters
	h_0 = config["synapses"]["h_0"]*mvolt
	tau_syn = config["synapses"]["tau_syn"]*msecond
	tau_OU = tau_syn
	t_ax_delay = config["synapses"]["t_ax_delay"]*msecond
	tau_Ca = config["synapses"]["calcium"]["tau_Ca"]*second
	t_Ca_delay = config["synapses"]["calcium"]["t_Ca_delay"]*msecond
	Ca_pre = config["synapses"]["calcium"]["Ca_pre"]
	Ca_post = config["synapses"]["calcium"]["Ca_post"]
	tau_h = config["synapses"]["early_phase"]["tau_h"]*second
	theta_p = config["synapses"]["early_phase"]["theta_p"]
	theta_d = config["synapses"]["early_phase"]["theta_d"]
	gamma_p = config["synapses"]["early_phase"]["gamma_p"]
	gamma_d = config["synapses"]["early_phase"]["gamma_d"]
	sigma_pl = config["synapses"]["early_phase"]["sigma_pl"]*mvolt
	tau_pro_c = config["synapses"]["protein"]["tau_pro_c"]*second
	alpha_c = config["synapses"]["protein"]["alpha_c"]
	theta_pro_c = config["synapses"]["protein"]["theta_pro"]*mvolt
	tau_z = config["synapses"]["late_phase"]["tau_z"]*second
	z_max = config["synapses"]["late_phase"]["z_max"]
	z_min = config["synapses"]["late_phase"]["z_min"]
	theta_tag_p = config["synapses"]["late_phase"]["theta_tag"]*mvolt
	theta_tag_d = config["synapses"]["late_phase"]["theta_tag"]*mvolt

	#########################################################
	### Equations ###
	# Neuron dynamics: LIF membrane potential and amount of plasticity-related proteins
	neuron_eqs = '''
	dV/dt = ( -(V - V_rev) + V_psp + R_mem*(I_bg + I_learn) ) / tau_mem: volt (unless refractory)
	dI_bg/dt = ( -I_bg + I_0 + sigma_wn*xi_bg ) / tau_OU : amp
	dI_learn/dt = ( -I_learn + int(t >= t_stim_start)*int(t < t_stim_end)*(N_stim*f_stim + sqrt(N_stim*f_stim)*xi_stim) * second*h_0/R_mem ) / tau_OU : amp
	sum_h_diff : volt
	dp/dt = ( -p + alpha_c*0.5*(1+sign(sum_h_diff - theta_pro_c)) ) / tau_pro_c : 1
	dV_psp/dt = -V_psp / tau_syn : volt
	'''

	# Synapse dynamics: calcium amount, early-phase weight, tag, late-phase weight
	synapse_eqs = '''
	dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
	dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
	sum_h_diff_post = abs(h - h_0) : volt (summed)
	dz/dt = ( alpha_c*p_post *(z_max-z)*0.5*(1+sign((h - h_0) - theta_tag_p)) - alpha_c*p_post *(z - z_min) * 0.5*(1+sign((h_0 - h) - theta_tag_d)) ) / tau_z : 1 (clock-driven)
	'''
	synapse_eqs_early_only = '''
	dh/dt = ( 0.1*(h_0 - h) + gamma_p*(10*mvolt - h)*0.5*(1+sign(Ca - theta_p)) - gamma_d*h*0.5*(1+sign(Ca - theta_d)) + sqrt(tau_h * (0.5*(1+sign(Ca - theta_p)) + 0.5*(1+sign(Ca - theta_d)))) * sigma_pl * xi_pl ) / tau_h : volt (clock-driven)
	dCa/dt = -Ca/tau_Ca : 1 (clock-driven)
	sum_h_diff_post = 0*volt : volt (summed)
	dz/dt = 0 / tau_z : 1 (clock-driven)
	'''
	synapse_pre = {
	'pre_voltage': 'V_psp_post += (h + z*h_0)',
	'pre_calcium': 'Ca += Ca_pre'
	}
	synapse_post='''Ca += Ca_post'''
	
	#########################################################
	### Running the simulation ###
	#b.seed(0) # set seed for xi functionals to zero
	neur_pop = b.NeuronGroup(2, neuron_eqs, threshold='V>V_th', reset='V=V_reset', refractory='t_ref', method='heun')
	neur_pop.V = V_rev # initialize membrane potential at reversal potential
	neur_pop.V_psp = 0*mvolt # initialize postsynaptic potential at zero

	neur_pop.I_bg = I_0 # initialize background current at mean value
	writeLog(f"Backgrund current: mean = {I_0}; st. dev. = {sigma_wn/np.sqrt(2*tau_OU)}")

	mean_I_learn = N_stim*f_stim*second*h_0/R_mem
	stdev_I_learn = np.sqrt(N_stim*f_stim)*second*h_0/R_mem/np.sqrt(2*tau_OU)
	neur_pop.I_learn = mean_I_learn # initialize stimulus current at mean value
	#writeLog(f"Learning current: mean = {mean_I_learn}; st. dev. = {stdev_I_learn}")
	writeLog(f"Learning current: mean = {mean_I_learn}; st. dev. = {stdev_I_learn}")
	if record_spikes:
		spike_mon = b.SpikeMonitor(neur_pop)
	if record_neuron_traces:
		neur_state_mon = b.StateMonitor(neur_pop, ['V', 'p'], record=[0,1], dt=delta_t_sample)

	spike_gen_indices = explicit_stim_receivers*len(explicit_stim_times) # 0: index of the neuron in SpikeGeneratorGroup
	spike_gen_times = explicit_stim_times*msecond # spike times
	writeLog(f"Explicit stimulation: spike_gen_indices = {spike_gen_indices}; spike_gen_times = {spike_gen_times}")

	#spike_gen = b.SpikeGeneratorGroup(1,[],[]*second)
	spike_gen = b.SpikeGeneratorGroup(1, spike_gen_indices, spike_gen_times) # one spike-generating neuron
	#spike_gen.set_spikes(spike_gen_indices, spike_gen_times)
	syn_spike_gen_inp = b.Synapses(spike_gen, neur_pop, on_pre='''V_post+=w_spike''') # model of synapses from 'spike_gen_gen' to 'neur_pop'
	syn_spike_gen_inp.connect(i=0,j=1) # connect neuron 'i' (in 'spike_gen_gen') to neuron 'j' (in 'neur_pop')
	syn_neur_pop = b.Synapses(neur_pop, neur_pop, model=synapse_eqs, on_pre=synapse_pre, on_post=synapse_post,method='heun') # model of synapses within within population ('neur_pop')
	#syn_neur_pop = b.Synapses(neur_pop, neur_pop, model=synapse_eqs, on_pre='V_post += (h + z*h_0)',method='euler') # model of synapses within within population ('neur_pop')
	#syn_neur_pop = b.Synapses(neur_pop, neur_pop, on_pre='''V_post += h_0''',method='euler') # model of synapses within within population ('neur_pop')
	#syn_neur_pop = b.Synapses(neur_pop, neur_pop, model="h : volt", on_pre='''V_post += h''',method='euler') # model of synapses within within population ('neur_pop')
	syn_neur_pop.connect(i=1,j=0) # connect neurons in 'neur_pop': 1 -> 0
	#neur_pop_recurr_conn = b.Synapses(neur_pop, neur_pop, on_pre='''V_post+=h''') # recurrent synapses within 'neur_pop'
	syn_neur_pop.h = h_0
	syn_neur_pop.z = 0
	syn_neur_pop.Ca = 0
	syn_neur_pop.pre_voltage.delay = t_ax_delay
	syn_neur_pop.pre_calcium.delay = t_Ca_delay
	syn_state_mon = b.StateMonitor(syn_neur_pop, ['h', 'z', 'Ca'], record=[0], dt=delta_t_sample)

	#b.device.activate(directory="output", compile=True, run=True, build_on_run=True, debug=False)

	b.defaultclock.dt = delta_t
	b.run(t_max, report="text", report_period=60*second)
	#b.device.build(directory="output", compile=True, run=True, debug=False)

	#########################################################
	### Storing the data in files ###
	if record_neuron_traces:
		data_stacked = np.column_stack(
				       [neur_state_mon.t/msecond,
				        neur_state_mon.V[0]/mvolt,
				        np.nan*np.zeros(len(neur_state_mon.t)),
				        syn_state_mon.h[0]/mvolt,
				        syn_state_mon.z[0],
				        syn_state_mon.Ca[0],
				        neur_state_mon.p[0]])
	else:
		data_stacked = np.column_stack(
		       [syn_state_mon.t,
		        syn_state_mon.h[0]/mvolt,
		        syn_state_mon.z[0],
		        syn_state_mon.Ca[0]])

	if record_spikes:
		spike_times = np.column_stack((spike_mon.t/msecond, spike_mon.i))
		np.savetxt(os.path.join(data_path_trial, 'spikes.txt'), spike_times, fmt="%.4f")

	np.savetxt(os.path.join(data_path_trial, 'traces.txt'), data_stacked, fmt="%.4f")

	json.dump(config, open(os.path.join(data_path_trial, 'config.json'), "w"), indent="\t")

	#########################################################
	### Plotting ###
	b.plt.figure(figsize=(18,4))

	# Spike raster plot
	b.plt.subplot(131)
	if record_spikes:
		b.plt.subplot(131)
		b.plt.plot(spike_mon.t/msecond, spike_mon.i, '.', c="purple") # alpha=0.4
		b.plt.xlabel('Time (ms)')
		b.plt.ylabel('Neuron index');
		ax = b.plt.gca()
		ax.yaxis.set_major_locator(MultipleLocator(1))
		ax.yaxis.set_major_formatter(FormatStrFormatter('% 1.0f'))
		b.plt.ylim(-0.5,1.5)

	# Voltage trace neuron 0
	b.plt.subplot(132)
	if record_neuron_traces:
		b.plt.plot(neur_state_mon.t/msecond, neur_state_mon.V[0]/mvolt, c="#ff0000")
		b.plt.xlabel('Time (ms)')
		b.plt.ylabel('Membrane potential (mV)');
		#b.plt.xlim(-0.01/msecond, 0.06/msecond)
		if record_spikes and t_max < 30*second:
			for t in spike_mon.t:
				b.plt.axvline(t/msecond, ls='dotted', c="purple", alpha=0.4) # dotted line indicating spikes

	# Voltage trace neuron 1
	b.plt.subplot(133)
	if record_neuron_traces:
		b.plt.plot(neur_state_mon.t/msecond, neur_state_mon.V[1]/mvolt, c="#ff0000")
		b.plt.xlabel('Time (ms)')
		b.plt.ylabel('Membrane potential (mV)');
		#b.plt.xlim(-0.01/msecond, 0.06/msecond)

		b.plt.savefig(os.path.join(data_path_trial, 'traces_voltage_and_spike_raster.png'), dpi=800)
		b.plt.close()

	if record_spikes:
		writeLog("Spike times neuron 0:", spike_mon.t)
		writeLog("Spike times neuron 0:", spike_mon.i)

	# Synaptic calcium amount and protein amount of neuron 0
	b.plt.figure(figsize=(12,4))
	b.plt.plot(syn_state_mon.t/msecond, syn_state_mon.Ca[0], c="#c8c896")
	if record_neuron_traces:
		b.plt.plot(syn_state_mon.t/msecond, neur_state_mon.p[0], c="#008000")
	b.plt.xlabel('Time (ms)')
	b.plt.ylabel('Calcium or protein amount');
	#b.plt.xlim(-0.01/msecond, 0.06/msecond)
	if record_spikes and t_max < 30*second:
		for t in spike_mon.t:
			b.plt.axvline(t/msecond, ls='dotted', c="purple", alpha=0.4) # dotted line indicating spikes
	b.plt.axhline(theta_p, ls='dotted', c='#969664')
	b.plt.axhline(theta_d, ls='dotted', c='#969696')

	b.plt.savefig(os.path.join(data_path_trial, 'traces_calcium_protein.png'), dpi=800)
	b.plt.close()
		
	# Early-/late-phase synaptic weight
	b.plt.figure(figsize=(12,4))
	b.plt.plot(syn_state_mon.t/msecond, syn_state_mon.h[0]/h_0*100, c="#800000")
	b.plt.plot(syn_state_mon.t/msecond, (syn_state_mon.z[0]+1)*100, c="#1f77b4")
	b.plt.xlabel('Time (ms)')
	b.plt.ylabel('Synaptic weight (%)');
	#b.plt.xlim(-0.01/msecond, 0.06/msecond)
	if record_spikes and t_max < 30*second:
		for t in spike_mon.t:
			b.plt.axvline(t/msecond, ls='dotted', c="purple", alpha=0.4) # dotted line indicating spikes
	#b.plt.axhline(h_0/mvolt, ls='dotted', c='C2')

	b.plt.savefig(os.path.join(data_path_trial, 'traces_weight.png'), dpi=800)
	b.plt.close()

	return data_path_trial

