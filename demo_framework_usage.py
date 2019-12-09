import numpy as np
import theano_shim as shim
from parameters import ParameterSet

from sinn.histories import Series, PopulationSeries
from sinn.popterm import PopTermMeso

#from fsGIF.main import init_spiking_model
from fsGIF.core import get_model_params
from mesogif_model_series import GIF

# Use load_theano=False to make debugging easier
shim.load(load_theano=True)

shim.config.compute_test_value = 'ignore'


pop_sizes=(9,8)
# n_bins x n, i.e. one time bin per row, one col. per node
spike_trains = np.random.randint(30, size=(1000,sum(pop_sizes))) # 100 time bins, 17 nodes, up to 30 spikes per bin
state_labels_1d = np.random.randint(3, size=(1000,1))  # state labels as state 0, 1, or 2
# states converted to a binary n_bins x n_states matrix
states_binary_2d = np.hstack((state_labels_1d==0, state_labels_1d==1, state_labels_1d==2)) + \
                   np.zeros((state_labels_1d.shape[0], 1))
# states_17d = state_labels_1d * np.ones((1, 17))

print('spike_trains.shape:', spike_trains.shape)
print('states_binary_2d.shape:', states_binary_2d.shape)
# print('states_17d.shape:', states_17d.shape)


param_dt = 4.
tarr = np.arange(1000)*param_dt    # 100 bins, each lasting 4 seconds
spiketrain = PopulationSeries(name='s', time_array=tarr, pop_sizes=pop_sizes)
# spiketrain = Series(name='s', time_array=tarr, shape=(sum(pop_sizes),))
spiketrain.set(source=spike_trains)
# spiketrain.set(source=np.hstack((states_binary_2d, spike_trains)))

# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(3,))
# state_hist.set(source=states_binary_2d)
state_hist = Series(name='z', time_array=tarr, dt=param_dt, shape=(1,))
state_hist.set(source=state_labels_1d)
# state_hist = Series(t0=param_t0, tn=param_tn, dt=param_dt, shape=(17,))
# state_hist.set(source=states_17d)


model_params = get_model_params(ParameterSet("spike_model_params.ntparameterset"), "GIF_spiking")
# HACK: Casting to PopTerm should be automatic
# model_params = model_params._replace(
#         τ_θ=PopTermMeso(pop_sizes, model_params.τ_θ, ('Meso',)),
#         τ_m=PopTermMeso(pop_sizes, model_params.τ_m, ('Meso',)))

spiking_model = GIF(model_params,
                    spiketrain, state_hist,
                    initializer='silent',
                    set_weights=True)



print("loglikelihood")
# Integrate the model forward to the time point with index 40
spiking_model.advance(40)
print(spiking_model.logp(40, 100))     # Int argument => Interpreted as time index
print(spiking_model.logp(160., 400.))  # Float argument => Interpreted as time in seconds
#gradient_descent(input_filename=None, output_filename="test_output.test",
                 #batch_size=100,
                 #model=spiking_model)


########
## Gradient descent

import sinn.optimize.gradient_descent as gd

# First create symbolic variables which are used as inputs in GD function
t = shim.symbolic.scalar('tidx', dtype=spiketrain.tidx_dtype)
batch_size_var = shim.symbolic('batch_size', dtype=spiketrain.tidx_dtype)
batch_size     = 100
    # Note: instead of `spiketrain`, you can also use `spiking_model.s`
    # It's a bit dumb that we have to specify both a symbolic variable
    # and definite the value; eventually both will be replaced by a shared variable

# Next get the computational graph for the log likelihood
logL, updates = spiking_model.loglikelihood(t, batch_size)

# In theory you could compile this graph with `shim.compile([t, batch_size], logL)` and use
# the resulting function it as is.
# The optimizer included in *sinn* has some niceties wrt automatically selecting minibatches,
# so I explain it below.

###
# Instantiating the optimizer

# This requires setting bounds, e.g. in case some initial data segment should be discarded
# Batches will be selected from the interval [start:start+datalen], where burnin & datalen
# are specified in seconds, as specified if the docstring.
# There's lots more info in there, so make sure to type `gd.SeriesSGD?` in a Jupyter
# console or notebook !
# Also: I'm basically picking numbers at random here; you'll have to figure out what makes
# sense for your data.
start = 12.
datalen = tarr[-1] - burnin
# We also need to specify the burnin before each batch
burnin = 4.
# Values of start and burnin mostly depend on how quickly the model's internal state
# converges to something sensible.
# For the mesoscopic model this is quite long, but for the spiking model it should be short.

# Normalize logL
logL *= datalen / batch_size

sgd = gd.SeriesSGD(
    cost           = logL,
    start_var      = t,
    batch_size_var = batch_size_var,
    batch_size     = batch_size,
    cost_format    = 'logL',
    start          = start,
    datalen        = datalen,
    burnin         = burnin,
    advance        = spiking_model.advance_updates
    )
