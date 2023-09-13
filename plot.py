#!/bin/python3

########################################################################################################
### Plots the mean and the standard deviation across batches of the mean across trials, for synaptic ###
###  weight, membrane potential, calcium amount, and amount of plasticity-related proteins           ### 
########################################################################################################

### Copyright 2021-2023 Jannik Luboeinski
### licensed under Apache-2.0 (http://www.apache.org/licenses/LICENSE-2.0)
### Contact: mail[at]jlubo.net

#########################################################
### Imports and initialization ###

import numpy as np
import matplotlib.pyplot as plt
import json
import os
	
#####################################
# plotMeanAndSD
# Plots the mean and the standard deviation across batches of the mean across trials, for synaptic weight, membrane voltage, calcium amount, and amount of plasticity-related proteins
# - config: configuration parameters in JSON format
# - data_stacked: two-dimensional array containing the values of the membrane potential, weights, calcium amount, etc. over time
# - X_cols: dictionary pointing to the data columns of the measures whose mean and st.dev. is to be plotted (must point to the mean, the st.dev. is sought in the next column)
# - store_path [optional]: path to store resulting graphics file
def plotMeanAndSD(config, data_stacked,
                  X_cols = {"voltage": 1, "weight-e": 3, "weight-l": 5, "calcium": 7, "protein": 9}, store_path = '.'):

	FIGSIZE = 8 # fonts inversely scale with this 
	h_0 = config["synapses"]["h_0"]
	fig, axes = plt.subplots(nrows=3, ncols=1, sharex=False, figsize=(FIGSIZE, FIGSIZE))
	fig.set_tight_layout(True)

	# set axis labels for synaptic weight plot
	if np.max(data_stacked[:,0]) < 60000:
		time_axis_label = "Time (ms)"
	else:
		data_stacked[:,0] /= 60000
		time_axis_label = "Time (min)"
	axes[0].set_xlabel(time_axis_label)
	axes[0].set_ylabel(f"Mean synaptic weight (%)")
	
	# plot data for synaptic weight plot
	axes[0].plot(data_stacked[:,0], data_stacked[:,X_cols["weight-e"]]/h_0*100, color="#800000", label='Early-phase', marker='None', zorder=8)
	axes[0].fill_between(data_stacked[:,0], (data_stacked[:,X_cols["weight-e"]]-data_stacked[:,X_cols["weight-e"]+1])/h_0*100, 
                                            (data_stacked[:,X_cols["weight-e"]]+data_stacked[:,X_cols["weight-e"]+1])/h_0*100, color="#800000", alpha=0.5)
	axes[0].plot(data_stacked[:,0], (data_stacked[:,X_cols["weight-l"]]+1)*100, color="#1f77b4", label='Late-phase', marker='None', zorder=9)
	axes[0].fill_between(data_stacked[:,0], ((data_stacked[:,X_cols["weight-l"]]-data_stacked[:,X_cols["weight-l"]+1])+1)*100, 
                                            ((data_stacked[:,X_cols["weight-l"]]+data_stacked[:,X_cols["weight-l"]+1])+1)*100, color="#1f77b4", alpha=0.5)
	if np.max(np.abs(data_stacked[:,X_cols["weight-e"]] - h_0)) > config["synapses"]["late_phase"]["theta_tag"]:
		axes[0].axhline(y=(config["synapses"]["protein"]["theta_pro"]/h_0+1)*100, label='Protein thresh.', linestyle='-.', color="#dddddd", zorder=5)
		axes[0].axhline(y=(config["synapses"]["late_phase"]["theta_tag"]/h_0+1)*100, label='Tag thresh.', linestyle='dashed', color="#dddddd", zorder=4)
	
	# create y-axis and legend for synaptic weight plot
	#axes[0].set_ylim(0,1)
	axes[0].legend(loc="upper right")

	# set axis labels for membrane potential plot
	axes[1].set_xlabel(time_axis_label)
	axes[1].set_ylabel(f"Mean membrane potential (mV)")
	
	# plot data for membrane potential
	axes[1].plot(data_stacked[:,0], data_stacked[:,X_cols["voltage"]], color="#ff0000", marker='None', zorder=10)
	axes[1].fill_between(data_stacked[:,0], data_stacked[:,X_cols["voltage"]]-data_stacked[:,X_cols["voltage"]+1], 
                                            data_stacked[:,X_cols["voltage"]]+data_stacked[:,X_cols["voltage"]+1], color="#ff0000", alpha=0.5)

	# create y-axis for membrane potential plot
	#axes[1].set_ylim(0,0.006)

	# set axis labels for protein/calcium plot
	axes[2].set_xlabel(time_axis_label)
	axes[2].set_ylabel(f"Mean protein amount")
	axLtwin = axes[2].twinx() # create twin axis
	axLtwin.set_ylabel("Mean calcium amount")
	
	# plot data for protein/calcium amount
	axLtwin.plot(data_stacked[:,0], data_stacked[:,X_cols["calcium"]], color="#c8c896", label="Calcium", marker='None', zorder=8)
	axLtwin.fill_between(data_stacked[:,0], data_stacked[:,X_cols["calcium"]]-data_stacked[:,X_cols["calcium"]+1], 
	                                        data_stacked[:,X_cols["calcium"]]+data_stacked[:,X_cols["calcium"]+1], color="#c8c896", alpha=0.5)
	axes[2].plot(data_stacked[:,0], data_stacked[:,X_cols["protein"]], color="#008000", label="Protein", marker='None', zorder=9)
	axes[2].fill_between(data_stacked[:,0], data_stacked[:,X_cols["protein"]]-data_stacked[:,X_cols["protein"]+1], 
	                                        data_stacked[:,X_cols["protein"]]+data_stacked[:,X_cols["protein"]+1], color="#008000", alpha=0.5)
	axLtwin.axhline(y=config["synapses"]["early_phase"]["theta_p"], label='LTP threshold', linestyle='dashed', color="#969664", zorder=7)
	axLtwin.axhline(y=config["synapses"]["early_phase"]["theta_d"], label='LTD threshold', linestyle='dashed', color="#969696", zorder=6)

	# create y-axis and legend for protein/calcium plot
	#axes[2].set_ylim(0,30)
	handles, labels = axes[2].get_legend_handles_labels()
	handles_twin, labels_twin = axLtwin.get_legend_handles_labels()
	axes[2].legend(handles + handles_twin, labels + labels_twin, loc="upper right")

	# save figure as vector graphics
	fig.savefig(store_path, pad_inches=2.5)

#####################################
if __name__ == '__main__':
	for (protocol, num_trials) in [("basic_early", 100), ("basic_late", 10)]:
		try:
			config = json.load(open(f"config_{protocol}.json", "r"))
			data_stacked_brian = np.loadtxt(f'brian-heun_data_{protocol}_10x{num_trials}/meta_mean_averaged_traces.txt')
			plotMeanAndSD(config, data_stacked_brian,
					      X_cols = {"voltage": 1, "weight-e": 3, "weight-l": 5, "calcium": 7, "protein": 9}, store_path = f"./{protocol}_10x{num_trials}.png")
		except:
			print(f"An error occurred tried to plot '{basic_early}' data.")
		
