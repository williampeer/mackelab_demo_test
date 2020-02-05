
import various_libs as vlib

all_data = vlib.import_data

pops = (6,6,5); asser(sum(pops) == all_data.spike_train.shape[1])
spiking_model = vlib.spiking_model(spike_hist = all_data.spike_train,
                                   populations = pops)

# TODO: Fix computation of lambda here..
spiking_model.compute_spike_hist()

# TODO: Verify log likelihood computation
spiking_model.log_likelihood()  # ref. hist. & computed hist.

# TODO: Pass the loss function to the framework, let it fit the free population model parameters
spiking_model.fit_mesoscopic()  # uses log_likelihood to fit parameters for a mesoscopic model using SGD

# TODO: Use the attained model to infer the most likely parameter distributions for a mesoscopic model - how?
spiking_model.infer_microscopic_dist()  # uses PyMC3 / Hamilitonian MC to infer most likely microscopic param. distributions

# TODO: Plot end result
spiking_model.plot_microscopic_dist()