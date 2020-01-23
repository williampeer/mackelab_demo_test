from collections import OrderedDict, namedtuple
import sinn
import sinn.models as models
from sinn.histories import Series
from sinn.history_functions import GaussianWhiteNoise
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams

class Demo(models.Model):
    requires_rng = True
    Parameter_info = OrderedDict((
        ('N', 'int16'),
        ('p', 'floatX'),
        ('a', 'floatX'),
        ('b', 'floatX'),
        ('c', 'floatX'),
        ('d', 'floatX')
    ))
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['I', 'V'])
    # TODO: May make assumption about pathway and activity relative to state, and transform state to input.

    def __init__(self, params, spike_history, state_history,
                 initializer=None, set_weights=True, random_stream=None, memory_time=None):

        self.s = spike_history
        super().__init__(params,
                         t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
                         reference_history=self.s)

        N = self.params.N.get_value()
        assert (N.ndim == 1)
        self.Npops = len(N)

        self.V = Series(name='V', t0=self._t0, tn=self._tn, dt=self._dt, shape=(N.sum(),))
        V = self.V
        # `V` is passed as a template: defines default time stops & shape - COPY SETUP
        self.Vbar = Series(V, name='Vbar')
        self.I = Series(V, name='I')
        self.u = Series(V, name='u')

        self.s_modelled = Series(V, name='s_modelled')

        # TODO: Implement rndstream using Theano
        # if random_stream is not None:
        #     self.rndstream = random_stream
        # else:
        #     self.rndstream = RandomStreams(seed=234)
        # self.η = GaussianWhiteNoise(V, name='η', random_stream=self.rng,
        #                             shape=V.shape, std=1.)

        models.Model.same_dt(self.s, self.V)
        # models.Model.same_dt(self.s, self.I_ext)
        # models.Model.output_rng(self.s, self.rndstream)

        self.add_history(self.V)
        self.add_history(self.Vbar)
        self.add_history(self.u)
        self.add_history(self.I)
        # self.add_history(self.η)

        # self.V.set_update_function(self.V_fn, inputs=[self.u, self.I])
        self.V.set_update_function(self.V_fn, inputs=[self.Vbar, self.u, self.I])
        # self.Vbar.set_update_function(self.Vbar_fn, inputs=self.V)
        self.u.set_update_function(self.u_fn, inputs=[self.V, self.u])
        self.I.set_update_function(self.I_fn, inputs=self.V)

        # Pad every history corresponding to a differential equation
        # by one to make space for the initial condition
        self.V.pad(1)
        self.u.pad(1)
        self.I.pad(1)


        self.original_params = self.params  # Store unexpanded params
        # self.params = self.params._replace(
        #     **{name: self.expand_param(getattr(self.params, name), N)
        #        for name in params._fields
        #        if name != 'N'})

    # -------- Initialization ----------- #
    def initialize(self, t=None, symbolic=False):
        """
        :param:t is the time at which we want to start.
        Initialization will be for all times strictly smaller than :param:t.
        By default the initialization removes any symbolic dependencies by
        calling `eval()` on the initial values; setting :param:symbolic to
        `True` keeps symbolic dependencies.
        """
        self.clear_unlocked_histories()
        if t is None: t = 0
        tidx = self.get_tidx(t) - 1
        if not self.V.locked:
            i = self.get_tidx_for(tidx, self.V)
            self.V[i] = -65.
        if not self.u.locked:
            i = self.get_tidx_for(tidx, self.u)
            self.u[i] = 8

        if not symbolic:
            self.eval()

    def V_fn(self, t):
        iV = self.V.get_tidx_for(t, self.V)
        # iVbar = self.V.get_tidx_for(t, self.Vbar)
        V_t = self.V[iV-1]
        # Vbar_t = self.Vbar[iVbar]

        iu = self.V.get_t_idx(t, self.u)
        u_t = self.u[iu-1]

        iI = self.V.get_t_idx(t, self.I)
        I_t = self.u[iI-1]

        # if (V_t >= 30.):  # TODO: fix discontinuity per node
        #     return -65.0

        # dVi = 0.04 * Vbar_t ** 2 + 5 * Vbar_t + 140 - u_t + I_t
        dVi = 0.04 * V_t ** 2 + 5. * V_t + 140. - u_t + I_t
        # return Vbar_t + dVi
        return V_t + dVi

    def Vbar_fn(self, t):
        return -65.0 * np.reshape(np.zeros_like(t), (t[0], 1))

    def u_fn(self, t):
        iV = self.V.get_tidx_for(t, self.V)
        # iVbar = self.V.get_tidx_for(t, self.Vbar)
        V_t = self.V[iV - 1]
        # Vbar_t = self.Vbar[iVbar]

        iu = self.V.get_t_idx(t, self.u)
        u_t = self.u[iu - 1]

        # if (V_t >= 30.):
        #     return u_t + 8.0
        return 0.1 * (0.25 * V_t - u_t)  # this makes one bin one timestep.
        # return 16.0 * np.reshape(np.zeros_like(t), (t[0], 1))

    def I_fn(self, t):
        # iη = self.get_tidx_for(t, self.η)
        # η_i = self.η[iη]
        # return η_i
        # return 0.001 * np.reshape(np.random.random(t.shape), (t[0], 1))
        return np.zeros_like(self.I[0])

    def logp_numpy(self, t0, tn):
        return "logp."


sinn.models.register_model(Demo)
