import numpy as np
import theano_shim as shim
from parameters import ParameterSet

from sinn.histories import Spiketrain, Series

#from fsGIF.main import init_spiking_model
from fsGIF.fsgif_model import GIF_spiking
from fsGIF.core import get_model_params

# n_bins x n, i.e. one time bin per row, one col. per node
spike_trains = np.random.randint(30, size=(1000,17))  # 100 time bins, 17 nodes, up to 30 spikes per bin
state_labels_1d = np.random.randint(3, size=(1000,1))  # state labels as state 0, 1, or 2
# states converted to a binary n_bins x n_states matrix
states_binary_2d = np.hstack((state_labels_1d==0, state_labels_1d==1, state_labels_1d==2)) + \
                   np.zeros((state_labels_1d.shape[0], 1))
# states_17d = state_labels_1d * np.ones((1, 17))

print('spike_trains.shape:', spike_trains.shape)
print('states_binary_2d.shape:', states_binary_2d.shape)
# print('states_17d.shape:', states_17d.shape)

# Temporarily unload Theano since it isn't supported by spike history
use_theano = shim.config.use_theano
shim.load(load_theano=False)

param_dt = 4.
tarr = np.arange(1000)*param_dt    # 100 bins, each lasting 4 seconds
spiketrain = Spiketrain(pop_sizes=(9,8), time_array=tarr, dt=param_dt)
spiketrain.set(source=spike_trains)
# spiketrain.set(source=np.hstack((states_binary_2d, spike_trains)))

# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(3,))
# state_hist.set(source=states_binary_2d)
state_hist = Series(time_array=tarr, dt=param_dt, shape=(1,))
state_hist.set(source=state_labels_1d)
# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(17,))
# state_hist.set(source=states_17d)

#spiking_model = init_spiking_model(spike_history=spiketrain, input_history=state_hist)
# init_spiking_model(spike_history=None, input_history=state_hist, datalen=100)

model_params = get_model_params(ParameterSet("spike_model_params.ntparameterset"), "GIF_spiking")
# HACK: Casting to PopTerm should be automatic
model_params = model_params._replace(
        τ_θ=spiketrain.PopTerm(model_params.τ_θ),
        τ_m=spiketrain.PopTerm(model_params.τ_m))
spiking_model = GIF_spiking(model_params,
                            spiketrain, state_hist,
                            initializer='silent',
                            set_weights=True)


use_theano = shim.config.use_theano
shim.load(load_theano=True)

# TODO: Fix GD function
gradient_descent(input_filename=None, output_filename="test_output.test",
                 batch_size=100,
                 model=spiking_model)

