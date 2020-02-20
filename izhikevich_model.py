from collections import OrderedDict, namedtuple
import sinn
import sinn.models as models
from sinn.histories import Series
import numpy as np
import theano_shim as shim
import theano.tensor as T

class Izhikevich(models.Model):
    requires_rng = True
    Parameter_info = OrderedDict((
        ('N', 'int16'),
        ('a', 'floatX'),
        ('b', 'floatX'),
        ('c', 'floatX'),
        ('d', 'floatX'),
        ('spike_threshold', 'floatX'),
        ('syn_decay', 'floatX')
    ))
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['I', 'V'])

    # TODO: May make assumption about pathway and activity relative to state, and transform state to input.

    def __init__(self, params, spike_history, state_history,
                 initializer=None, set_weights=True, random_stream=None, memory_time=None):

        self.s = spike_history
        self.I_ext = state_history

        super().__init__(params,
                         t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
                         reference_history=self.s)

        N = self.params.N.get_value()
        assert (N.ndim == 1)
        self.Npops = len(N)

        self.V = Series(name='V', t0=self._t0, tn=self._tn, dt=self._dt, shape=(N.sum(),))
        V = self.V
        # `V` is passed as a template: defines default time stops & shape - COPY SETUP
        self.I = Series(V, name='I')
        self.u = Series(V, name='u')
        self.g = Series(V, name='g')
        self.s_count = Series(V, name='s_count')

        self.micro_ws = self.init_micro_weights()
        self.meso_ws = self.init_meso_weights()

        models.Model.same_dt(self.s, self.V)

        self.add_history(self.V)
        self.add_history(self.u)
        self.add_history(self.g)
        self.add_history(self.I)
        self.add_history(self.s_count)

        self.V.set_update_function(self.V_fn, inputs=[self.u, self.I])
        self.u.set_update_function(self.u_fn, inputs=[self.V, self.u])
        # self.I.set_update_function(self.I_fn, inputs=[self.V])
        self.g.set_update_function(self.g_fn, inputs=[self.g, self.V])
        self.I.set_update_function(self.I_fn, inputs=[self.g])
        self.s_count.set_update_function(self.s_count_fn, inputs=[self.V, self.s_count])

        # Pad every history corresponding to a differential equation
        # by one to make space for the initial condition
        self.V.pad(1)
        self.u.pad(1)
        self.I.pad(1)
        self.g.pad(1)

        self.original_params = self.params  # Store unexpanded params
        self.params = self.params._replace(
            **{name: self.expand_param(getattr(self.params, name), N)
               for name in params._fields
               if name != 'N'})

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
        tot_num_neurons = self.params.N.get_value().sum()
        if not self.V.locked:
            ind_V = self.get_tidx_for(tidx, self.V)
            self.V[ind_V] = -65. * np.ones(shape=(tot_num_neurons,))
        else:
            raise EnvironmentError("Could not set V. V was locked")
        if not self.u.locked:
            ind_u = self.get_tidx_for(tidx, self.u)
            self.u[ind_u] = 8 * np.ones(shape=(tot_num_neurons,))
        else:
            raise EnvironmentError("Could not set u. u was locked")

        # if not self.s_count.locked:
        #     ind_s_count = self.get_tidx_for(tidx, self.s_count)
        #     self.s_count[ind_s_count] = np.zeros(shape=(tot_num_neurons,))

        if not self.g.locked:
            ind_g = self.get_tidx_for(tidx, self.g)
            self.g[ind_g] = np.ones(shape=(tot_num_neurons,))
        else:
            raise EnvironmentError("Could not set g. g was locked")

        if not symbolic:
            self.eval()

    def init_meso_weights(self):
        # pop_ws = np.array([[0.1, 0.8, 0.8],
        #                    [0.8, 0.1, 0.8],
        #                    [0.8, 0.8, 0.1]])
        # return pop_ws

        assert(self.micro_ws is not None)

        pop_ws = np.zeros((self.Npops, self.Npops))
        for pop_ind in range(0, self.Npops):
            cur_avgs = []
            prev_ind_sum = 0
            pop_size_sum = 0
            for pop_size in self.params.N.get_value():
                pop_size_sum += pop_size
                cur_avgs.append(np.mean(self.micro_ws[prev_ind_sum:pop_size_sum]))
            pop_ws[pop_ind] = np.array(cur_avgs)
        return pop_ws

    def init_micro_weights(self):
        N_neurons = self.params.N.get_value().sum()
        w_shape = (N_neurons, N_neurons)
        # micro_ws = 0.2 * (np.random.random(size=w_shape) - 0.5 * np.ones(shape=w_shape))
        micro_ws = 3.5 * np.random.random(size=w_shape)  # only positive, for now
        return micro_ws
        # return theano.tensor.as_tensor_variable(micro_ws, 'micro_ws')

    # --------------------------------------------------------------------------------

    def V_fn(self, t):
        ind_V = self.V.get_tidx_for(t-1, self.V)
        V_prev = self.V[ind_V]

        ind_u = self.V.get_t_idx(t-1, self.u)
        u_prev = self.u[ind_u]

        ind_I = self.V.get_t_idx(t-1, self.I)
        I_prev = self.I[ind_I]

        dV_t_1 = 0.5 * (0.04 * V_prev ** 2 + 5. * V_prev + 140. - u_prev + I_prev)
        v_half_step = V_prev + dV_t_1  # half step for numerical stability, ref. Izhikevich, IEEE (2003)
        dV_t_2 = 0.5 * (0.04 * v_half_step ** 2 + 5. * v_half_step + 140. - u_prev + I_prev)
        V_t = v_half_step + dV_t_2

        fired = (V_t >= self.params.spike_threshold)
        not_fired = (V_t < self.params.spike_threshold)
        V_t_thresh = self.params.c * fired + V_t * (not_fired)
        return V_t_thresh

    def u_fn(self, t):
        ind_V = self.u.get_tidx_for(t, self.V)
        V_prev = self.V[ind_V - 1]
        V_t = self.V[ind_V]  # Make sure this has been calculated before use in u_fn

        fired_prev = 1.0 - ((V_prev > self.params.c) + (V_prev < self.params.c))  # broadcasting, check if reset val.

        ind_u = self.u.get_t_idx(t-1, self.u)
        u_prev = self.u[ind_u]
        u_t = self.params.a * (self.params.b * V_t - u_prev) + self.params.d * fired_prev
        return u_t

    def g_fn(self, t):
        ind_V = self.u.get_tidx_for(t-1, self.V)
        V_prev = self.V[ind_V]
        fired_prev = 1.0 - ((V_prev > self.params.c) + (V_prev < self.params.c))  # broadcasting, check if reset val.

        ind_g_prev = self.g.get_tidx_for(t-1, self.g)
        g_prev = self.g[ind_g_prev]
        dg_t = -g_prev / self.params.syn_decay
        g_decayed = (g_prev + dg_t)
        g_t = fired_prev + (1.0 - fired_prev) * g_decayed
        return g_t

    def I_fn(self, t):
        ind_g = self.I.get_tidx_for(t-1, self.g)
        g_t = self.g[ind_g]

        ind_I_ext = self.I.get_tidx_for(t, self.I_ext)
        I_ext_t = self.I_ext[ind_I_ext]

        I_syn_t = np.dot(g_t, self.micro_ws)  # note: presynaptic propagation considered
        return I_syn_t + I_ext_t

    # ----------------------------
    def s_count_fn(self, t):
        ind_V = self.s.get_tidx_for(t, self.V)
        V_prev = self.V[ind_V]

        fired_prev = 1.0 - ((V_prev > self.params.c) + (V_prev < self.params.c))  # broadcasting, check if reset val.
        return fired_prev

    # ------------------------------------------------------------------

    @models.batch_function_scan('s', 's_count')
    def logp(self):
        spike_train_data = self.s
        spike_train_model = self.s_count
        p = sinn.clip_probabilities(spike_train_model * spike_train_data.dt)
        s = spike_train_data
        return (s * p - (1 - p) + s * (1 - p)).sum()

    def logp_numpy(self, t0, t_k):
        s_count = self.s_count[t0:t_k]
        s = self.s[t0:t_k]
        p = sinn.clip_probabilities(s_count*self.s.dt)
        return ( s*p - (1-p) + s*(1-p) ).sum()  # sum over batch and neurons

    @staticmethod
    def expand_param(param, N):
        """
        Expand a population parameter such that it can be multiplied directly
        with the spiketrain.

        Parameters
        ----------
        param: ndarray
            Parameter to expand

        N: tuple or ndarray
            Number of neurons in each population
        """
        block_types = []
        for s in sinn.shim.get_test_value(param).shape:
            block_types.append('Macro' if s == 1
                               else 'Meso' if s == len(N)
            else 'Micro' if s == sum(N)
            else None)
        assert None not in block_types
        return sinn.popterm.expand_array(N, param, tuple(block_types))

        # Npops = len(N)
        # if param.ndim == 1:
        #     return shim.concatenate( [ param[i]*np.ones((N[i],))
        #                                for i in range(Npops) ] )
        #
        # elif param.ndim == 2:
        #     return shim.concatenate(
        #         [ shim.concatenate( [ param[i, j]* np.ones((N[i], N[j]))
        #                               for j in range(Npops) ],
        #                             axis = 1 )
        #           for i in range(Npops) ],
        #         axis = 0 )
        # else:
        #     raise ValueError("Parameter {} has {} dimensions; can only expand "
        #                      "dimensions of 1d and 2d parameters."
        #                      .format(param.name, param.ndim))


sinn.models.register_model(Izhikevich)
