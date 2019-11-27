import numpy as np

from main import *
from sinn.histories import Spiketrain, Series

from fsGIF.main import init_spiking_model

# n_bins x n, i.e. one time bin per row, one col. per exp.
spike_trains = np.random.randint(30, size=(1000,17))  # 100 time bins, 17 nodes, up to 30 spikes per bin
state_labels_1d = np.random.randint(3, size=(1000,1))  # state labels as state 0, 1, or 2
# states converted to a binary n_bins x n_states matrix
states_binary_2d = np.hstack((state_labels_1d==0, state_labels_1d==1, state_labels_1d==2)) + \
                   np.zeros((state_labels_1d.shape[0], 1))

print('spike_trains.shape:', spike_trains.shape)
print('states_binary_2d.shape:', states_binary_2d.shape)

# Temporarily unload Theano since it isn't supported by spike history
use_theano = shim.config.use_theano
shim.load(load_theano=False)

param_dt = 4
param_t0 = 0
param_tn = (1000-1)*param_dt  # 100 bins, each lasting 4 seconds
spiketrain = Spiketrain(pop_sizes=(17), t0=param_t0, tn=param_tn, dt=param_dt)
# spiketrain = Spiketrain(pop_sizes=(3,17), t0=param_t0, tn=param_tn, dt=param_dt)
spiketrain.set(source=spike_trains)
# composite_trains = np.hstack((states_binary_2d, spike_trains))
# spiketrain.set(source=composite_trains)

state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(3,))
state_hist.set(source=states_binary_2d)

# TODO: Fix init., then call fit
# init_spiking_model(spike_history=None, input_history=state_hist, datalen=100)
init_spiking_model(spike_history=spiketrain, input_history=state_hist)

# fit(output="test_filename", batch=100)
