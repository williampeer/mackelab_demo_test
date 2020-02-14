import numpy as np
import theano_shim as shim
from parameters import ParameterSet
from sinn.histories import Series, PopulationSeries

from core import get_model_params
from izhikevich_model import Izhikevich
from sys import exit

# Use load_theano=False to make debugging easier
# load_theano_flag = True
load_theano_flag = False
shim.load(load_theano=load_theano_flag)

shim.config.compute_test_value = 'warn'

t_n = 1001
pop_sizes=(6,6,5)

spike_trains = np.random.randint(30, size=(t_n,sum(pop_sizes)))  # n_bins x n; one time bin per row, one col. per node
state_labels_1d = np.random.randint(3, size=(t_n,1))  # state labels as state 0, 1, or 2
broadcast_state_labels = 0.5 * state_labels_1d + np.zeros_like(spike_trains)  # see "Writing" for discussion

print('spike_trains.shape:', spike_trains.shape)
print('broadcast_state_labels.shape:', broadcast_state_labels.shape)

param_dt = 0.002
tarr = np.arange(t_n)*param_dt  # avoid floating point imprecision w/ duration/dt later by defining tarr like this
spiketrain = PopulationSeries(name='s', time_array=tarr, pop_sizes=pop_sizes)
spiketrain.set(source=spike_trains)

state_hist = Series(name='z', time_array=tarr, dt=param_dt, shape=(sum(pop_sizes),))
state_hist.set(source=broadcast_state_labels)

# Locking histories identifies them as data
# The model will not modify them, and treats them as known inputs when
# constructing computation graphs.
spiketrain.lock()
state_hist.lock()

model_params = get_model_params(ParameterSet("params_izhikevich.ntparameterset"), "Izhikevich")
spiking_model = Izhikevich(model_params, spiketrain, state_hist)

print('initialising..')
spiking_model.initialize()

# Integrate the model forward to the time point with index X
print("advancing..")
spiking_model.advance(t_n-1)
# print("lp:", spiking_model.logp(1, 2))     # Int argument => Interpreted as time index
print(spiking_model.logp_numpy(10, 20))     # Int argument => Interpreted as time index
# print(spiking_model.logp(160., 400.))  # Float argument => Interpreted as time in seconds
#gradient_descent(input_filename=None, output_filename="test_output.test",
                 #batch_size=100,
                 #model=spiking_model)

# if not load_theano_flag:
#     exit() # don't run GD with pure numpy
exit() # don't run GD with pure numpy
