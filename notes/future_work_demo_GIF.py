########
## Gradient descent

import sinn.optimize.gradient_descent as gd
gd.raise_on_exception = True
    # Better for testing. For batch jobs best to leave this False

# First create symbolic variables which are used as inputs in GD function
# t = shim.symbolic.scalar('tidx', dtype=spiketrain.tidx_dtype)
# batch_size_var = shim.symbolic.scalar('batch_size', dtype=spiketrain.tidx_dtype)
t = shim.shared(shim.cast(1, spiking_model.tidx_dtype), name='tidx')
batch_size_var = shim.shared(shim.cast(10, spiking_model.tidx_dtype),
                             name='batch_size')
batch_size     = 10
    # Note: instead of `spiketrain`, you can also use `spiking_model.s`
    # It's a bit dumb that we have to specify both a symbolic variable
    # and definite the value; eventually both will be replaced by a shared variable
# t.tag.test_value = 1
# batch_size_var.tag.test_value = 100
    # Always give assign a test_value to symbolic variables

# Next get the computational graph for the log likelihood
logL = spiking_model.logp(t, batch_size=batch_size_var)

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
start = spiking_model.get_tidx(12.)
datalen = len(tarr) - start
burnin = spiking_model.index_interval(4.)
# We also need to specify the burnin before each batch
# Values of start and burnin mostly depend on how quickly the model's internal state
# converges to something sensible.
# For the mesoscopic model this is quite long, but for the spiking model it should be short.

# Normalize logL
logL *= datalen / batch_size

# Select variables we want to optimize
# Others will be fixed to the value in the parameters file
# `original_params` is something I added to mesogif_model_series,
# because `params` is overwritten with expanded parameters
# w = spiking_model.original_params.w
    # To fit w: reduce bin size so that kernels aren't zero
c = spiking_model.original_params.c
R = spiking_model.original_params.R
vars_to_optimize = [c, R]

# Select which variables to track
# You can also track transformations; the one below is just
# a silly example, but this can be useful if you are actually fitting
# a transformation (such as log)
vars_to_track = {
    'c': c,
    'R': R,
    'log(cR)': shim.log(c*R)
}

sgd = gd.SeriesSGD(
    cost           = logL,
    start_var      = t,
    batch_size_var = batch_size_var,
    batch_size     = batch_size,
    cost_format    = 'logL',
    optimizer_kwargs = {'lr': 0.005},
    optimize_vars  = vars_to_optimize,
    track_vars     = vars_to_track,
    start          = start,
    datalen        = datalen,
    burnin         = burnin,
    advance        = spiking_model.advance_updates,
    mode           = 'sequential'  # One of 'sequential', 'random'
    )

sgd.fit(10)
sgd.trace
