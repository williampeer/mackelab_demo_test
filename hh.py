from collections import Callable, deque, namedtuple
import numpy as np
from odictliteral import odict

import theano_shim as shim

import sinn
import sinn.models as models
from sinn.models import Model
from sinn.histories import Series
from sinn.history_functions import GaussianWhiteNoise


def efun(z):
    return shim.switch(shim.abs(z) < 1e-4,
                       1 - z/2,
                       z / (shim.exp(z) - 1))

class HodgkinHuxley(Model):
    requires_rng = True
    Parameter_info = odict['V0'     : 'floatX',
                           'gbar_Na': 'floatX',  # mS/cm²
                           'gbar_K' : 'floatX',
                           'g_leak' : 'floatX',
                           'gbar_M' : 'floatX',
                           'τ_max'  : 'floatX',  # ms
                           'Vt'     : 'floatX',
                           'σ'      : 'floatX',  # noise factor
                           'E_leak' : 'floatX',
                           'C'      : 'floatX',  # μF/cm²
                           'E_Na'   : 'floatX',  # mV
                           'E_K'    : 'floatX'
                          ]
    Parameters = sinn.define_parameters(Parameter_info)
    State = namedtuple('State', ['I', 'V', 'n', 'm', 'h', 'p'])

    def __init__(self, params, t0, tn, dt, rng=None):

        # Call parent initializer
        super().__init__(params, t0, tn, dt)

        # Make V, I into class attributes, so we can access them in methods
        self.V = Series(name='V', t0=t0, tn=tn, dt=dt, shape=(1,))
        V = self.V
        self.I = Series(V, name='I')
        self.rng = rng
        self.set_reference_history(V)

        # Create the internal histories.
        # `V` is passed as a template: defines default time stops & shape
        self.Vbar = Series(V, name='Vbar')
        self.n = Series(V, name='n')
        self.m = Series(V, name='m')
        self.h = Series(V, name='h')
        self.p = Series(V, name='p')
        self.η = GaussianWhiteNoise(V, name='η', random_stream=self.rng,
                                    shape=V.shape, std=1.)

        # Add histories to the model
        self.add_history(self.V)
        self.add_history(self.Vbar)
        self.add_history(self.I)
        self.add_history(self.n)
        self.add_history(self.m)
        self.add_history(self.h)
        self.add_history(self.p)
        self.add_history(self.η)

        # Attach the update functions to the internal histories
        tauV_inputs = set([self.n, self.m, self.h, self.p])
        Vinf_inputs = set([self.I, self.n, self.m, self.h, self.p, self.η]) \
                      .union(tauV_inputs)
        self.V.set_update_function(
            self.V_fn, inputs=tauV_inputs.union([self.Vbar, self.η]))
        self.Vbar.set_update_function(
            self.Vbar_fn, inputs=tauV_inputs.union(Vinf_inputs).union([self.V]))
        self.n.set_update_function(self.n_fn, inputs=[self.n, self.V])
        self.m.set_update_function(self.m_fn, inputs=[self.m, self.V])
        self.h.set_update_function(self.h_fn, inputs=[self.h, self.V])
        self.p.set_update_function(self.p_fn, inputs=[self.p, self.V])

        # Tell histories which other histories they depend on
        # self.V.add_inputs([self.Vbar, self.η])
        # self.Vbar.add_inputs([self.V, self.I, self.n, self.m, self.h, self.p])
        # self.n.add_inputs([self.V, self.n])
        # self.m.add_inputs([self.V, self.m])
        # self.h.add_inputs([self.V, self.h])
        # self.p.add_inputs([self.V, self.p])

        # Pad every history correspoding to a differential equation
        # by one to make space for the initial condition
        self.V.pad(1)
        self.n.pad(1)
        self.m.pad(1)
        self.h.pad(1)
        self.p.pad(1)

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
        V0 = self.V0
        if not self.V.locked:
            i = self.get_tidx_for(tidx, self.V)
            self.V[i] = V0
        if not self.n.locked:
            i = self.get_tidx_for(tidx, self.n)
            self.n[i] = self.n_inf(V0)
        if not self.m.locked:
            i = self.get_tidx_for(tidx, self.m)
            self.m[i] = self.m_inf(V0)
        if not self.h.locked:
            i = self.get_tidx_for(tidx, self.h)
            self.h[i] = self.h_inf(V0)
        if not self.p.locked:
            i = self.get_tidx_for(tidx, self.p)
            self.p[i] = self.p_inf(V0)

        if not symbolic:
            self.eval()

    # ------ Kinetics ------- #

    def α_m(self, x):
        v1 = x - self.Vt - 13.
        return 0.32*efun(-0.25*v1)/0.25
    def β_m(self, x):
        v1 = x - self.Vt - 40
        return 0.28*efun(0.2*v1)/0.2
    def α_h(self, x):
        v1 = x - self.Vt - 17.
        return 0.128*shim.exp(-v1/18.)
    def β_h(self, x):
        v1 = x - self.Vt - 40.
        return 4.0/(1 + shim.exp(-0.2*v1))
    def α_n(self, x):
        v1 = x - self.Vt - 15.
        return 0.032*efun(-0.2*v1)/0.2
    def β_n(self, x):
        v1 = x - self.Vt - 10.
        return 0.5*shim.exp(-v1/40)

    # slow non-inactivating K+
    @staticmethod
    def p_inf(x):
        v1 = x + 35.
        return 1.0/(1. + shim.exp(-0.1*v1))

    def τ_p(self, x):
        v1 = x + 35.
        return self.τ_max/(3.3*shim.exp(0.05*v1) + shim.exp(-0.05*v1))

    def τ_n(self, x):
         return 1/(self.α_n(x) + self.β_n(x))
    def n_inf(self, x):
        return self.α_n(x)/(self.α_n(x) + self.β_n(x))
    def τ_m(self, x):
        return 1/(self.α_m(x) + self.β_m(x))
    def m_inf(self, x):
        return self.α_m(x)/(self.α_m(x) + self.β_m(x))
    def τ_h(self, x):
        return 1/(self.α_h(x) + self.β_h(x))
    def h_inf(self, x):
        return self.α_h(x)/(self.α_h(x) + self.β_h(x))

    # ------ Update equations ------ #

    def V_fn(self, t):
        iVbar = self.V.get_tidx_for(t, self.Vbar) ; Vbar = self.Vbar[iVbar]
        iη = self.V.get_tidx_for(t, self.η)  ; η  = self.η[iη]
        iHH = self.V.get_tidx_for(t, self)
        tau_V_inv = self.tau_V_inv(iHH)
        return Vbar + self.σ/self.C * η

    def tau_V_inv(self, t):
        i_n = self.get_tidx_for(t, self.n) ; n0 = self.n[i_n-1]
        im = self.get_tidx_for(t, self.m)  ; m0 = self.m[im-1]
        ih = self.get_tidx_for(t, self.h)  ; h0 = self.h[ih-1]
        ip = self.get_tidx_for(t, self.p)  ; p0 = self.p[ip-1]
        return ( (m0**3)*self.gbar_Na*h0 + (n0**4)*self.gbar_K
                  +self.g_leak + self.gbar_M*p0 ) / self.C

    def V_inf(self, t):
        iI = self.get_tidx_for(t, self.I)  ; I1 = self.I[iI]
        i_n = self.get_tidx_for(t, self.n) ; n0 = self.n[i_n-1]
        im = self.get_tidx_for(t, self.m)  ; m0 = self.m[im-1]
        ih = self.get_tidx_for(t, self.h)  ; h0 = self.h[ih-1]
        ip = self.get_tidx_for(t, self.p)  ; p0 = self.p[ip-1]
        iHH = self.get_tidx_for(t, self)
        iη = self.get_tidx_for(t, self.η)  ; η  = self.η[iη]
        tau_V_inv = self.tau_V_inv(iHH)
        res =  ( (m0**3)*self.gbar_Na*h0*self.E_Na
                  + (n0**4)*self.gbar_K*self.E_K
                  + self.g_leak*self.E_leak + self.gbar_M*p0*self.E_K
                  + I1 ) / (tau_V_inv*self.C)
        #res += self.σ/self.C/tau_V_inv * η
        return res

    def Vbar_fn(self, t):
        # Get time index for each history (may differ b/c padding)
        iV = self.Vbar.get_tidx_for(t, self.V)  ; V0 = self.V[iV-1]
        iHH = self.Vbar.get_tidx_for(t, self)
        tau_V_inv = self.tau_V_inv(iHH)
        V_inf = self.V_inf(iHH)
        return V_inf + (V0-V_inf) * shim.exp(-self.V.dt*tau_V_inv)

    def n_fn(self, t):
        i_n = self.n.get_tidx(t)             ; n0 = self.n[i_n-1]
        i_V = self.n.get_tidx_for(t, self.V) ; V1 = self.V[i_V]
        ninf = self.n_inf(V1)
        return ninf + (n0-ninf)*shim.exp(-self.n.dt/self.τ_n(V1))

    def m_fn(self, t):
        i_V = self.m.get_tidx_for(t, self.V) ; V1 = self.V[i_V]
        i_m = self.m.get_tidx(t)             ; m0 = self.m[i_m-1]
        minf = self.m_inf(V1)
        return minf + (m0-minf)*shim.exp(-self.m.dt/self.τ_m(V1))

    def h_fn(self, t):
        i_V = self.h.get_tidx_for(t, self.V) ; V1 = self.V[i_V]
        i_h = self.h.get_tidx(t)             ; h0 = self.h[i_h-1]
        hinf = self.h_inf(V1)
        return hinf + (h0-hinf)*shim.exp(-self.h.dt/self.τ_h(V1))

    def p_fn(self,t):
        i_V = self.p.get_tidx_for(t, self.V)  ; V1 = self.V[i_V]
        i_p = self.p.get_tidx(t)              ; p0 = self.p[i_p-1]
        pinf = self.p_inf(V1)
        return pinf + (p0-pinf)*shim.exp(-self.p.dt/self.τ_p(V1))

sinn.models.register_model(HodgkinHuxley)
