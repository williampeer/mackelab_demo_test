from collections import OrderedDict, namedtuple
import sinn
import sinn.models as models
from sinn.histories import Series
from sinn.history_functions import GaussianWhiteNoise


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
    State = namedtuple('State', ['u', 'v', 'I'])
    # TODO: May make assumption about pathway and activity relative to state, and transform state to input.

    def __init__(self, params, spike_history,
                 initializer=None, set_weights=True, random_stream=None, memory_time=None):

        self.s = spike_history
        super().__init__(params,
                         t0=self.s.t0, tn=self.s.tn, dt=self.s.dt,
                         reference_history=self.s)

        self.V = Series(name='v', t0=self._t0, tn=self._tn, dt=self._dt, shape=(1,))
        V = self.V
        self.I = Series(V, name='I', shape=V.shape)
        self.u = Series(V, name='u')

        self.η = GaussianWhiteNoise(V, name='η', random_stream=self.rng,
                                    shape=V.shape, std=1.)
        self.rndstream = random_stream

        models.Model.same_dt(self.s, self.V)
        # models.Model.same_dt(self.s, self.I_ext)
        models.Model.output_rng(self.s, self.rndstream)

        self.add_history(self.V)
        self.add_history(self.u)
        self.add_history(self.I)
        self.add_history(self.η)

        self.V.set_update_function(self.V_fn, inputs=[self.u, self.I])
        self.u.set_update_function(self.u_fn, inputs=self.V)
        self.I.set_update_function(self.I_fn, inputs=self.η)

    def V_fn(self, t):


sinn.models.register_model(Demo)
