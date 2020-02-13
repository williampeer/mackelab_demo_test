from collections import OrderedDict, namedtuple
import sinn
import sinn.models as models
from sinn.histories import Series
from sinn.history_functions import GaussianWhiteNoise
import numpy as np
from theano.tensor.shared_randomstreams import RandomStreams


class Izhikevich(models.Model):
    requires_rng = True
    Parameter_info = OrderedDict((
        ('N', 'int16'),
        ('p', 'floatX'),
        ('a', 'floatX'),
        ('b', 'floatX'),
        ('c', 'floatX'),
        ('d', 'floatX'),
        ('V_m', 'floatX')
    ))
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['I', 'V'])

    # TODO: May make assumption about pathway and activity relative to state, and transform state to input.

    def __init__(self, params, spike_history, state_history,
                 initializer=None, set_weights=True, random_stream=None, memory_time=None):

        self.s = spike_history
        self.I_ext = state_history  # TODO: incorporate

        super().__init__(params,
                         t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
                         reference_history=self.s)

        N = self.params.N.get_value()
        assert (N.ndim == 1)
        self.Npops = len(N)

        self.V = Series(name='V', t0=self._t0, tn=self._tn, dt=self._dt, shape=(N.sum(),))
        V = self.V
        # `V` is passed as a template: defines default time stops & shape - COPY SETUP
        # self.Vbar = Series(V, name='Vbar')
        self.I = Series(V, name='I')
        self.u = Series(V, name='u')

        self.s_per_bin = Series(V, name='s_per_bin')

        # TODO: rndstream in Theano
        # if random_stream is not None:
        #     self.rndstream = random_stream
        # else:
        #     self.rndstream = RandomStreams(seed=234)
        # self.η = GaussianWhiteNoise(V, name='η', random_stream=self.rng,
        #                             shape=V.shape, std=1.)

        models.Model.same_dt(self.s, self.V)
        models.Model.same_dt(self.s_per_bin, self.s)
        # models.Model.same_dt(self.s, self.I_ext)
        # models.Model.output_rng(self.s, self.rndstream)

        self.add_history(self.V)
        # self.add_history(self.Vbar)
        self.add_history(self.u)
        self.add_history(self.I)
        # self.add_history(self.η)

        self.add_history(self.s_per_bin)

        # self.V.set_update_function(self.V_fn, inputs=[self.u, self.I])
        self.V.set_update_function(self.V_fn, inputs=[self.u, self.I])
        # self.Vbar.set_update_function(self.Vbar_fn, inputs=self.V)
        self.u.set_update_function(self.u_fn, inputs=[self.V, self.u])
        self.I.set_update_function(self.I_fn, inputs=self.V)
        self.s_per_bin.set_update_function(self.s_per_bin_fn, inputs=[self.u, self.I])

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
        if t is None:
            t = 0
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
        ind_V = self.V.get_tidx_for(t, self.V)
        V_prev = self.V[ind_V - 1]

        ind_u = self.V.get_t_idx(t, self.u)
        u_prev = self.u[ind_u - 1]

        ind_I = self.V.get_t_idx(t, self.I)
        I_prev = self.u[ind_I - 1]

        dV_t = 0.04 * V_prev ** 2 + 5. * V_prev + 140. - u_prev + I_prev
        V_t = V_prev + dV_t
        if (V_t > self.params.V_m):
            return self.params.c
        else:
            return V_t

    def s_per_bin_fn(self, t):
        # ind_V = self.V.get_tidx_for(t, self.V); V_prev = self.V[ind_V - 1]
        # ind_u = self.V.get_t_idx(t, self.u); u_prev = self.u[ind_u - 1]
        # ind_I = self.V.get_t_idx(t, self.I); I_prev = self.u[ind_I - 1]
        #
        # dV_t = 0.04 * V_prev ** 2 + 5. * V_prev + 140. - u_prev + I_prev
        # V_t = V_prev + dV_t

        ind_V = self.V.get_tidx_for(t, self.V)
        V_t = self.V[ind_V]
        if (V_t > self.params.V_m):
            s = 1
        else:
            s = 0
        return s

    # def Vbar_fn(self, t):
    #     return -65.0 * np.reshape(np.zeros_like(t), (t[0], 1))

    def u_fn(self, t):
        ind_V = self.V.get_tidx_for(t, self.V)
        V_t = self.V[ind_V]

        ind_u = self.V.get_t_idx(t, self.u)
        u_prev = self.u[ind_u - 1]

        if (V_t == self.params.c):
            return u_prev + self.params.d
        else:
            V_prev = self.V[ind_V - 1]
            return self.params.a * (self.params.b * V_prev - u_prev)  # this makes one bin one timestep.

    def I_fn(self, t):
        # iη = self.get_tidx_for(t, self.η)
        # η_i = self.η[iη]
        # return η_i
        # return 0.001 * np.reshape(np.random.random(t.shape), (t[0], 1))
        return np.zeros_like(self.I[0])

    def logp_numpy(self, t0, tn):
        return "logp."


sinn.models.register_model(Izhikevich)
