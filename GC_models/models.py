import numpy as np
from helper import *
from scipy.integrate import quad
from scipy.special import kn

Pi = np.pi

class fermionic_dm_spin0_med_schannel(object):
    """
    Lagrangian = [\bar{\chi} (\lamba_{\chi,s} + \lambda_{\chi,p} i \gamma^5) \chi +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """
    def __init__(self, mx, dm_type, f, m_a, lam_chi_s, lam_chi_p, lam_f_s, lam_f_p):
        self.mx = mx
        self.dm_type = dm_type
        self.m_a = m_a
        self.lam_chi_s = lam_chi_s
        self.lam_chi_p = lam_chi_p
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        if self.dm_type == 'majorana':
            self.lam_chi_s *= 2.
            self.lam_chi_p *= 2.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f
            lam_f_p = self.lam_f_p * mass_f
            sigma += (nc / (16. * np.pi * s * ((s - self.m_a ** 2.) ** 2. + (self.m_a * gamma) ** 2.)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                (lam_f_s ** 2. * (s - 4. * mass_f ** 2.) + lam_f_p ** 2. * s) *
                (self.lam_chi_s ** 2. * (s - 4. * self.mx ** 2.) + self.lam_chi_p ** 2. * s))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                lam_f_s = self.lam_f_s * mass_f
                lam_f_p = self.lam_f_p * mass_f
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + lam_f_p ** 2.))
        return 1.

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = (3. * lam ** 4. * self.mx ** 2. /
                                      (np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) *
                (1. + 9. * v ** 2. / (8. * (1. - 4. * self.mx ** 2 / self.m_a ** 2.))))
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        sv = 0.0
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f
        lam_f_p = self.lam_f_p * mass_f
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv += (nc * self.lam_chi_p ** 2. * kin_mass *
                (self.mx ** 2. * (lam_f_p ** 2. + lam_f_s ** 2.) - (mass_f * lam_f_s) ** 2.) /
                (2. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) +
                nc * v ** 2. / (16. * np.pi * self.mx ** 2. * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. * kin_mass) *
                (self.lam_chi_p ** 2. *
                 ((self.mx * lam_f_p) ** 2. *
                  (self.m_a ** 2. * (mass_f ** 2. - 2. * self.mx ** 2.) + 12. *
                  (self.mx * mass_f) ** 2. - 8. * self.mx ** 4.) + lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) *
                  (self.m_a ** 2. * (mass_f ** 2. + 2. * self.mx ** 2.) - 20. * (self.mx * mass_f) ** 2. + 8. * self.mx ** 4.)) -
                 2. * self.lam_chi_s ** 2. * (self.m_a ** 2. - 4. * self.mx ** 2.) * (self.mx ** 2. - mass_f ** 2.) *
                 (self.mx ** 2. * (lam_f_p ** 2. + lam_f_s ** 2.) - mass_f ** 2. * lam_f_s ** 2.)))
        if self.m_a < self.mx:
            gxp = np.sqrt(self.lam_chi_p*self.lam_f_p)
            gxs = np.sqrt(self.lam_chi_s*self.lam_f_s)
            sv += (8*np.sqrt(self.mx**4)*gxp**2*gxs**2*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2)))/\
                  (self.mx**2*(2*self.mx**2 - self.m_a**2)**2*np.pi) + \
                  (np.sqrt(self.mx**4)*(2*self.mx**6*gxp**4 - 36*self.mx**6*gxp**2*gxs**2 + 18*self.mx**6*gxs**4 -
                                        6*self.mx**4*gxp**4*self.m_a**2 +
                                60*self.mx**4*gxp**2*gxs**2*self.m_a**2 - 34*self.mx**4*gxs**4*self.m_a**2 +
                                        6*self.mx**2*gxp**4*self.m_a**4 -
                                24*self.mx**2*gxp**2*gxs**2*self.m_a**4 + 20*self.mx**2*gxs**4*self.m_a**4 -
                                        2*gxp**4*self.m_a**6 +
                                3*gxp**2*gxs**2*self.m_a**6 - 4*gxs**4*self.m_a**6)*v**2)/\
                  (3.*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2))*(2*self.mx**2 - self.m_a**2)**4*np.pi)

        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h




class fermionic_dm_spin1_med_schannel(object):
    """
    Lagrangian = [\bar{\chi} \gamma^\mu (\lamba_{\chi,v} + \lambda_{\chi,a} \gamma^5) \chi +
    \bar{f}\gamma^\mu (\lamba_{f,v} + \lambda_{f,a} \gamma^5) f]V_\mu
    """
    def __init__(self, mx, dm_type, f, m_v, lam_chi_v, lam_chi_a, lam_f_v, lam_f_a, c_ratio=0.):
        self.mx = mx
        self.dm_type = dm_type
        self.m_v = m_v
        self.lam_chi_v = lam_chi_v
        self.lam_chi_a = lam_chi_a
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        self.c_ratio = c_ratio
        if self.dm_type == 'majorana':
            self.lam_chi_v *= 2.
            self.lam_chi_a *= 2.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if self.mx > mass_f:
                sigma += (nc / (12. * np.pi * s * ((s - self.m_v ** 2.) ** 2. + (self.m_v * gamma) ** 2.)) *
                    np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                    (self.lam_f_a ** 2. *
                     (self.lam_chi_a ** 2. *
                      (4. * self.mx ** 2. * (mass_f ** 2. * (7. - 6. * s / self.m_v ** 2. + 3. * s ** 2. / self.m_v ** 4.) - s)
                     + s * (s - 4. * mass_f ** 2.)) + self.lam_chi_v ** 2. * (s - 4. * mass_f ** 2.) * (2. * self.mx ** 2. + s))
                     + self.lam_f_v ** 2. * (2. * mass_f ** 2. + s) * (self.lam_chi_a ** 2. * (s - 4. * self.mx ** 2.) +
                                                                       self.lam_chi_v ** 2. * (2. * self.mx ** 2. + s))))

        return sigma


    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_v:
                width += nc * self.m_v / (12. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_v) ** 2.) *\
                         (self.lam_f_a ** 2. * (1. - 4. * (mass_f / self.m_v) ** 2.) +
                          self.lam_f_v ** 2. * (1. + 2. * (mass_f / self.m_v) ** 2.))
        return 1.


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        nc = color_number(channel)
        mass_f = get_mass(channel)
        sv = 0.
        gfa = self.lam_f_a
        gfv = self.lam_f_v
        gxa = self.lam_chi_a
        gxv = self.lam_chi_v
        if self.mx > mass_f:
            sv += nc * ((8*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                   (16*gfa**2*gxa**2*self.mx**4*mass_f**2 - 8*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**2 +
                    2*gfa**2*gxv**2*self.mx**2*self.m_v**4 + 2*gfv**2*gxv**2*self.mx**2*self.m_v**4 +
                    gfa**2*gxa**2*mass_f**2*self.m_v**4 - 2*gfa**2*gxv**2*mass_f**2*self.m_v**4 +
                    gfv**2*gxv**2*mass_f**2*self.m_v**4))/(np.sqrt(self.mx**4)*self.m_v**4*
                                                           (4*self.mx**2 - self.m_v**2)**2*Pi) + \
                   (((np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*
                     (96*gfa**2*gxa**2*self.mx**6*mass_f**2 - 24*gfa**2*gxa**2*self.mx**4*mass_f**2*self.m_v**2 +
                      4*gfa**2*gxa**2*self.mx**4*self.m_v**4 + 4*gfv**2*gxa**2*self.mx**4*self.m_v**4 +
                      10*gfa**2*gxv**2*self.mx**4*self.m_v**4 + 10*gfv**2*gxv**2*self.mx**4*self.m_v**4 -
                      4*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 +
                      2*gfv**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 -
                      4*gfa**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4 +
                      2*gfv**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4))/
                    (np.sqrt(self.mx**4)*(-4*self.mx**2 + self.m_v**2)**2) +
                    (192*gfa**2*gxa**2*self.mx**6*mass_f**2 - 96*gfa**2*gxa**2*self.mx**4*mass_f**2*self.m_v**2 +
                     24*gfa**2*gxv**2*self.mx**4*self.m_v**4 + 24*gfv**2*gxv**2*self.mx**4*self.m_v**4 +
                     12*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 -
                     24*gfa**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4 +
                     12*gfv**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4)*
                    ((2*self.mx**4 - self.mx**2*mass_f**2)/
                     (4.*np.sqrt(self.mx**4)*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                      (-4*self.mx**2 + self.m_v**2)**2) + np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*
                     (-1/(4.*np.sqrt(self.mx**4)*(-4*self.mx**2 + self.m_v**2)**2) +
                      ((2*self.mx**2)/(-4*self.mx**2 + self.m_v**2)**3 -
                       1/(4.*(-4*self.mx**2 + self.m_v**2)**2))/np.sqrt(self.mx**4))))*v**2)/\
                  (3.*self.mx**2*self.m_v**4*Pi)) / 16.
        
        if self.m_v < self.mx:
            gxa = np.sqrt(self.c_ratio * self.lam_chi_a * self.lam_f_a)
            gxv = np.sqrt(self.c_ratio * self.lam_chi_v * self.lam_f_v)
            
            sv += (4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                   (8*self.mx**4*gxa**2*gxv**2 + self.mx**2*gxa**4*self.m_v**2 -
                    14*self.mx**2*gxa**2*gxv**2*self.m_v**2 + self.mx**2*gxv**4*self.m_v**2 -
                    gxa**4*self.m_v**4 + 6*gxa**2*gxv**2*self.m_v**4 - gxv**4*self.m_v**4))/\
                  (np.sqrt(self.mx**4)*self.m_v**2*(2*self.mx**2 - self.m_v**2)**2*Pi) + \
                  (np.sqrt(self.mx**4)*(128*self.mx**12*gxa**4 - 320*self.mx**10*gxa**4*self.m_v**2 -
                                        128*self.mx**10*gxa**2*gxv**2*self.m_v**2 + 248*self.mx**8*gxa**4*self.m_v**4 +
                                        656*self.mx**8*gxa**2*gxv**2*self.m_v**4 + 24*self.mx**8*gxv**4*self.m_v**4 +
                                        68*self.mx**6*gxa**4*self.m_v**6 - 1016*self.mx**6*gxa**2*gxv**2*self.m_v**6 +
                                        4*self.mx**6*gxv**4*self.m_v**6 - 212*self.mx**4*gxa**4*self.m_v**8 +
                                        624*self.mx**4*gxa**2*gxv**2*self.m_v**8 - 64*self.mx**4*gxv**4*self.m_v**8 +
                                        105*self.mx**2*gxa**4*self.m_v**10 - 142*self.mx**2*gxa**2*gxv**2*self.m_v**10 +
                                        53*self.mx**2*gxv**4*self.m_v**10 - 17*gxa**4*self.m_v**12 +
                                        6*gxa**2*gxv**2*self.m_v**12 - 17*gxv**4*self.m_v**12)*v**2)/\
                  (6.*self.mx**2*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                   (2*self.mx**2 - self.m_v**2)**4*Pi)

        return sv

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 6. * lam ** 4. * self.mx ** 2. / (np.pi * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2.) *\
                (1. + 3. * v ** 2. * (1. + 2. * self.mx ** 2. / self.m_v ** 2.) /
                 (4. * (1. - 4. * self.mx ** 2 / self.m_v ** 2.)))
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h


    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h




class scalar_dm_spin0_mass_fd_schannel(object):
    """
    Lagrangian = [\lambda_phi |\phi|^2 + \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """

    def __init__(self, mx, dm_real, f, m_a, lam_phi, lam_f_s, lam_f_p):
        self.mx = mx
        self.dm_real = dm_real
        self.m_a = m_a
        self.lam_p = lam_phi
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        if dm_real:
            self.lam_p *= 2.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f
            lam_f_p = self.lam_f_p * mass_f
            sigma += (nc * self.lam_p ** 2. / (8. * np.pi * s * ((s - self.m_a ** 2) ** 2 + (self.m_a * gamma) ** 2.)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                     (lam_f_s **2. * (s - 4. * mass_f ** 2.) + s * lam_f_p ** 2.))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                lam_f_s = self.lam_f_s * mass_f
                lam_f_p = self.lam_f_p * mass_f
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + lam_f_p ** 2.))
        return 1.

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f
        lam_f_p = self.lam_f_p * mass_f
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_p ** 2. * kin_mass * (lam_f_p ** 2. + lam_f_s ** 2. *
                                                     (1. - mass_f ** 2. / self.mx ** 2.)) /
                 (4 * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2) ** 2) -
                 nc * self.lam_p ** 2. * v ** 2. /
                 (32. * np.pi * self.mx ** 4 * (4. * self.mx ** 2 - self.m_a ** 2.) ** 3. * kin_mass) *
                 (lam_f_p ** 2. * self.mx ** 2. * (self.m_a ** 2. * mass_f ** 2. - 20. * mass_f ** 2. *
                                                        self.mx ** 2. + 16. * self.mx ** 4.) +
                  lam_f_s ** 2. * (self.mx ** 2. - mass_f ** 2.) *
                  (3. * self.m_a ** 2. * mass_f ** 2. - 28. * mass_f ** 2. * self.mx ** 2. + 16. * self.mx ** 4.)))
        if self.m_a < self.mx:
            gx = np.sqrt(self.lam_f_s * self.lam_p + self.lam_p * self.lam_f_p)

            sv += (np.sqrt(self.mx**4)*gx**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2)))/\
                  (self.mx**6*(2*self.mx**2 - self.m_a**2)**2*Pi) - (gx**4*(56*self.mx**6*np.sqrt(self.mx**4) -
                                                                            100*(self.mx**4)**1.5*self.m_a**2 +
                                                                            50*self.mx**2*np.sqrt(self.mx**4)*
                                                                            self.m_a**4 - 9*np.sqrt(self.mx**4)*
                                                                            self.m_a**6)*v**2)/\
                                                                    (24.*self.mx**4*np.sqrt(self.mx**2*(self.mx**2 -
                                                                                                        self.m_a**2))*
                                                                     (2*self.mx**2 - self.m_a**2)**4*Pi)

        return sv

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class scalar_dm_spin1_med_schannel(object):
    """
    Lagrangian = [i \lambda_p \phi^\dag \dderiv_\mu \phi +
    \bar{f}\gamma^\mu (\lamba_{f,v} + \lambda_{f,a} \gamma^5) f]V_\mu
    """
    def __init__(self, mx, dm_real, f, m_v, lam_p, lam_f_v, lam_f_a):
        self.mx = mx
        self.dm_real = dm_real
        self.m_v = m_v
        self.lam_p = lam_p
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        if self.dm_real:
            self.lam_p = 0.
            self.lam_f_v = 0.
            self.lam_f_a = 0.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)

            sigma += nc * self.lam_p ** 2 * s / (12. * np.pi * ((self.m_v ** 2 - s) ** 2 + (self.m_v * gamma) ** 2.)) *\
                np.sqrt(1. - 4. * self.mx ** 2. / s) * np.sqrt(1. - 4. * mass_f ** 2. / s) * \
                (self.lam_f_a ** 2. * (1. - 4. * mass_f ** 2. / s) + self.lam_f_v ** 2. * (1. + 2. * mass_f ** 2. / s))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_v:
                width += nc * self.m_v / (12. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_v) ** 2.) *\
                         (self.lam_f_a ** 2. * (1. - 4. * (mass_f / self.m_v) ** 2.) +
                          self.lam_f_v ** 2. * (1. + 2. * (mass_f / self.m_v) ** 2.))
        return 1.

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv += nc * self.lam_p ** 2. * self.mx ** 2 * v ** 2 / (6. * np.pi) * kin_mass / \
            (self.m_v ** 2. - 4. * self.mx ** 2) ** 2 * (self.lam_f_a ** 2. * (1. - mass_f ** 2. / self.mx ** 2.) +
                                                         self.lam_f_v ** 2. * (1 + 0.5 * mass_f ** 2. / self.mx ** 2.))

        if self.m_v < self.mx:
            gx = np.sqrt(self.lam_f_v * self.lam_p + self.lam_p * self.lam_f_a)
            
            sv += (16*gx**4*(self.mx**10*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2)) -
                             2*self.mx**8*self.m_v**2*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2)) +
                             self.mx**6*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))))/\
                  ((self.mx**4)**1.5*self.m_v**4*(2*self.mx**2 - self.m_v**2)**2*Pi) + \
                  (2*np.sqrt(self.mx**4)*gx**4*(24*self.mx**10 - 60*self.mx**8*self.m_v**2 +
                                                58*self.mx**6*self.m_v**4 - 35*self.mx**4*self.m_v**6 +
                                                16*self.mx**2*self.m_v**8 - 3*self.m_v**10)*v**2)/\
                  (3.*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*(2*self.mx**2 - self.m_v**2)**4*Pi)
            
        return sv

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)


    def x_freeze_out(self, exac=True):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class vector_dm_spin0_med_schannel(object):
    """
    Lagrangian = [\lamba_x X^\mu X_\mu^\dag +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """
    def __init__(self, mx, dm_real, f, m_a, lam_x, lam_f_s, lam_f_p):
        self.mx = mx
        self.dm_real = dm_real
        self.m_a = m_a
        self.lam_x = lam_x
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        if self.dm_real:
            self.lam_x *= 2.


    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f
            lam_f_p = self.lam_f_p * mass_f
            sigma += (nc * self.lam_x ** 2. / (72. * np.pi * ((s - self.m_a ** 2.) ** 2 + (self.m_a * gamma) ** 2)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2 / s)) *
                     (s / self.mx ** 2 * (s / (4. * self.mx ** 2) - 1.) + 3.) *
                     (lam_f_s ** 2. * (1. - 4. * mass_f ** 2. / s) + lam_f_p ** 2.))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f
            lam_f_p = self.lam_f_p * mass_f
            if mass_f < self.m_a:
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + lam_f_p ** 2.))
        return 1.


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f
        lam_f_p = self.lam_f_p * mass_f
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_x ** 2. * kin_mass * (lam_f_p ** 2. + lam_f_s ** 2. * (1. - (mass_f / self.mx) ** 2.)) /
                 (12. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) +
                nc * self.lam_x ** 2. * v ** 2 / (288. * np.pi * self.mx ** 4 * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. *
                                                  kin_mass) *
                (lam_f_p ** 2 * self.mx ** 2 * (4. * self.mx ** 2. * (7. * mass_f ** 2 - 2 * self.m_a ** 2) + 5. *
                                                     self.m_a ** 2 * mass_f ** 2. - 16. * self.mx ** 4.) +
                 lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) * (self.m_a ** 2. * (mass_f ** 2 + 8. * self.mx ** 2.) -
                                                                        52. * mass_f ** 2. * self.mx ** 2. +
                                                                        16. * self.mx ** 4.)))
        if self.m_a < self.mx:
            gx = np.sqrt(self.lam_f_s * self.lam_x + self.lam_x * self.lam_f_p)

            sv += (np.sqrt(self.mx**4)*gx**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2))*
                   (6*self.mx**4 - 4*self.mx**2*self.m_a**2 + self.m_a**4))/\
                  (9.*self.mx**10*(2*self.mx**2 - self.m_a**2)**2*Pi) - \
                  (gx**4*(80*self.mx**10 - 232*self.mx**8*self.m_a**2 + 260*self.mx**6*self.m_a**4 -
                          158*self.mx**4*self.m_a**6 + 46*self.mx**2*self.m_a**8 - 5*self.m_a**10)*v**2)/\
                  (216.*(self.mx**4)**1.5*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2))*
                   (2*self.mx**2 - self.m_a**2)**4*Pi)

        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h

class vector_dm_spin1_med_schannel(object):
    """
    Lagrangian = [\lamba_x (X^{\nu,\dag} \del_\nu X^\mu + h.c. ) +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] V_\mu
    """
    def __init__(self, mx, dm_real, f, m_v, lam_x, lam_f_v, lam_f_a):
        self.mx = mx
        self.dm_real = dm_real
        self.m_v = m_v
        self.lam_x = lam_x
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        if self.dm_real:
            self.lam_x *= 2.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += (self.lam_x ** 2. * (s - 4 * self.mx ** 2) / (72. * np.pi * (self.mx * self.m_v) ** 4. *
                                                                  ((self.m_v * gamma) ** 2 +
                                                                   (self.m_v ** 2 - s) ** 2)) *
            np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
            (2. * self.lam_f_v ** 2. * self.mx ** 2 * self.m_v ** 4 * (2. * mass_f ** 2 + s) +
             self.lam_f_a ** 2 * (2 * self.mx ** 2 * (s * self.m_v ** 4 - 2 * mass_f ** 2 *
                                                      (5 * self.m_v ** 4 - 6. * s * self.m_v ** 2 +
                                                       3 * s ** 2)) +
                                  3 * mass_f ** 2 * s * (self.m_v ** 2 - s) ** 2.)))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_a = self.lam_f_a * mass_f
            lam_f_v = self.lam_f_v * mass_f
            if mass_f < self.m_v:
                width += (nc * self.m_v / (12. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_v) ** 2.) *
                         (lam_f_a ** 2. * (1. - 4. * (mass_f / self.m_v) ** 2.) +
                          lam_f_v ** 2. * (1. + 2. * (mass_f / self.m_v) ** 2.)))
        return width


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_a = self.lam_f_a * mass_f
        lam_f_v = self.lam_f_v * mass_f
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_x ** 2. * v ** 2 * kin_mass / (27. * np.pi * (self.m_v ** 2. - 4. * self.mx ** 2)) ** 2 *
                 (mass_f ** 2. * (lam_f_v ** 2. - 2. * lam_f_a ** 2.) + 2. * self.mx ** 2. *
                  (lam_f_a ** 2. + lam_f_v ** 2.)))

        if self.m_v < self.mx:
            gx = np.sqrt(self.lam_f_v * self.lam_x + self.lam_x * self.lam_f_a)

            sv += (np.sqrt(self.mx**4)*gx**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                   (18*self.mx**8 - 38*self.mx**6*self.m_v**2 + 35*self.mx**4*self.m_v**4 -
                    15*self.mx**2*self.m_v**6 + 3*self.m_v**8))/(72.*self.mx**6*self.m_v**4*
                                                                 (2*self.mx**2 - self.m_v**2)**2*Pi) + \
                  (gx**4*(624*self.mx**14 - 1752*self.mx**12*self.m_v**2 + 1864*self.mx**10*self.m_v**4 -
                          818*self.mx**8*self.m_v**6 - 52*self.mx**6*self.m_v**8 + 211*self.mx**4*self.m_v**10 -
                          77*self.mx**2*self.m_v**12 + 9*self.m_v**14)*v**2)/(1728.*np.sqrt(self.mx**4)*self.m_v**4*
                                                                              np.sqrt(self.mx**2*(self.mx**2 -
                                                                                                  self.m_v**2))*
                                                                              (2*self.mx**2 - self.m_v**2)**4*Pi)

        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class dirac_fermionic_dm_spin0_med_tchannel(object):
    """
    Lagrangian = [\bar{\chi} (\lambda_s + \lambda_p \gamma^5)f A +
    \bar{f} (\lambda_s - \lambda_p \gamma^5) \chi A^\dagger]

    ONLY FOR DIRAC FERMION! Majorana has a different formula!
    """
    def __init__(self, mx, f, m_a, lam_s, lam_p):
        self.mx = mx
        self.m_a = m_a
        self.lam_s = lam_s / 2.
        self.lam_p = lam_p / 2.
        self.f = f

    def mediator_width(self):
        width = 0.
        # for ferm in self.f:
        #     sym = 2.
        #     nc = color_number(ferm)
        #     mass_f = get_mass(ferm)
        #     if mass_f < self.m_a:
        #         width += nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *\
        #                  (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.)
        return 1.

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 3. * (lam / 2.) ** 4. * self.mx ** 2. / (2. * np.pi * (self.m_a ** 2. + self.mx ** 2.) ** 2.) *\
                (1. + 3. * v ** 2. * (1. - 3. * self.mx ** 2. / self.m_a ** 2. - self.mx ** 4. / self.m_a ** 4.) /
                 (4. * (1. + self.mx ** 2 / self.m_a ** 2.)))
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_s = self.lam_s
        lam_p = self.lam_p
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = nc * kin_mass * (lam_p ** 2. * (self.mx - mass_f) + lam_s ** 2. * (mass_f + self.mx)) ** 2. / \
                 (8 * np.pi * (self.m_a ** 2. - mass_f ** 2. + self.mx ** 2.) ** 2.) - \
                nc * v ** 2. / (192. * np.pi * self.mx ** 2. * kin_mass * (self.m_a ** 2. - mass_f ** 2. + self.mx ** 2.) ** 4) * \
                (self.m_a ** 4 * (6. * mass_f ** 3 * self.mx * (lam_p ** 4. - lam_s ** 4) +
                                 mass_f ** 2 * self.mx ** 2 * (13. * lam_p ** 4 + 2 * lam_p ** 2 *
                                                               lam_s ** 2 + 13 * lam_s ** 4) +
                                 mass_f ** 4 * (-11. * lam_p ** 4. + 14. * lam_p ** 2 *
                                                lam_s ** 2 - 11. * lam_s ** 4) -
                                 8. * self.mx ** 4. * (lam_s ** 2 + lam_p ** 2.) ** 2) +
                 2. * self.m_a ** 2. * (mass_f ** 2 - self.mx ** 2) *
                 (lam_p ** 4. * (mass_f - self.mx) ** 2 * (8. * mass_f * self.mx + 11. * mass_f ** 2. - 12 * self.mx ** 2.) -
                  2. * lam_p ** 2. * lam_s ** 2 * (-19. * mass_f ** 2. * self.mx **2 +
                                                             7. * mass_f ** 4. + 12 * self.mx ** 4) +
                  lam_s ** 4. * (mass_f + self.mx) ** 2. * (-8. * mass_f * self.mx + 11. * mass_f ** 2. - 12. * self.mx ** 2)) -
                 lam_p ** 4. * (mass_f - self.mx) ** 4. * (mass_f + self.mx) ** 2. * (11. * mass_f ** 2. - 8. * self.mx ** 2) +
                 2. * lam_p ** 2. * lam_s ** 2. * (7. * mass_f ** 2. - 8. * self.mx ** 2.) *
                 (mass_f ** 2. - self.mx ** 2) ** 3. - lam_s ** 4 * (mass_f - self.mx) ** 2. *
                 (mass_f + self.mx) ** 4. * (11. * mass_f ** 2. - 8. * self.mx ** 2.))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def sig_therm_approx(self, lam, v):
        sigv = 3. * lam ** 4. / (2. * np.pi) * self.mx**2. / (self.mx**2. + self.m_a**2.) ** 2. * \
               (1. + 3. * v ** 2. * (1. - 3. * self.mx ** 2. / self.m_a**2. - self.mx**4. / self.m_a**4.) /
                (4. * (1. + self.mx**2. / self.m_a**2)))
        return sigv

    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', 'dirac')
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class dirac_fermionic_dm_spin1_med_tchannel(object):
    """
    Lagrangian = 1/2 [\bar{\chi} \gamma^\mu (g_\chi,s + g_\chi,p \gamma^5)f V_\mu +
    \bar{f} \gamma^\mu (g_f,s + g_f,p \gamma^5) \chi V_\mu^\dagger]

    """
    def __init__(self, mx, f, m_v, lam_v, lam_a):
        self.mx = mx
        self.m_v = m_v
        self.lam_v = lam_v / 2.
        self.lam_a = lam_a / 2.
        self.f = f

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 3. * (lam / 2.) ** 4. * self.mx ** 2. / (2. * np.pi * (self.m_v ** 2 + self.mx **2) ** 2.) * \
                ((2 + self.mx ** 2. / self.m_v ** 2.) ** 2. + 3. * v ** 2. / 4. *
                 (4. + 4 * self.mx ** 2. / self.m_v ** 2. + self.mx ** 4. / self.m_v ** 4. - 3. *
                  (self.mx / self.m_v) ** 6. - (self.mx / self.m_v) ** 8.) / (1. + self.mx ** 2. / self.m_v ** 2) ** 2.)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = nc * kin_mass / (8. * np.pi * self.m_v ** 4 * (-mass_f ** 2 + self.m_v ** 2. + self.mx ** 2.) ** 2.) *\
                 (mass_f ** 6. * (self.lam_a ** 2 - self.lam_v ** 2) ** 2 + 2 * mass_f ** 5 * self.mx *
                  (self.lam_a ** 4 - self.lam_v ** 4) + mass_f ** 4. *
                  (2. * self.lam_a ** 2 * self.lam_v ** 2 * (4 * self.m_v ** 2 + 3. * self.mx ** 2) -
                   self.lam_a ** 4 * self.mx ** 2 - self.lam_v ** 4. * self.mx ** 2) - 4 * mass_f ** 3 *
                  (self.mx * (self.lam_a ** 4 - self.lam_v ** 4) * (self.m_v ** 2 + self.mx ** 2)) +
                  mass_f ** 2 * (-2. * self.lam_a ** 2. * self.lam_v ** 2. *
                                 (8 * self.m_v ** 2 * self.mx ** 2 + 2. * self.m_v **4 + 3. * self.mx ** 4) +
                                 self.lam_a ** 4. * (-4. * self.m_v ** 2. * self.mx ** 2. + 2. *
                                                     self.m_v ** 4. - self.mx ** 4.) + self.lam_v ** 4. *
                                 (-4 * self.m_v ** 2. * self.mx ** 2. + 2 * self.m_v ** 4. - self.mx ** 4)) +
                  2. * mass_f * self.mx * (self.lam_a ** 4 - self.lam_v ** 4.) * (2. * self.m_v ** 2. * self.mx ** 2. +
                                                                                  2 * self.m_v ** 4 + self.mx ** 4.) +
                  self.mx ** 2. * (4. * self.m_v ** 2 * self.mx ** 2. * (self.lam_a ** 2. + self.lam_v ** 2.) ** 2. +
                                   4 * self.lam_a ** 2. * self.lam_v ** 2. * self.m_v ** 4. +
                                   6. * self.lam_a ** 4. * self.m_v ** 4. + 6 * self.lam_v ** 4. * self.m_v ** 4 +
                                   self.mx ** 4. * (self.lam_a ** 2. + self.lam_v ** 2.) ** 2.)) + \
                nc * v ** 2. / (192. * np.pi * self.m_v ** 4. * self.mx**2 *
                                kin_mass*(-mass_f**2. + self.m_v ** 2 + self.mx ** 2.) ** 4.) *\
                (mass_f ** 12. * (11 * (self.lam_a ** 4+self.lam_v ** 4) - 14 * self.lam_a ** 2 * self.lam_v ** 2) +
                 22. * mass_f ** 11 * self.mx * (self.lam_a ** 4 - self.lam_v ** 4) -
                     mass_f ** 10 * ((46 * self.m_v ** 2 + 41 * self.mx ** 2.) * (self.lam_a ** 4 + self.lam_v ** 4) -
                                     2 * self.lam_v ** 2. * (26 * self.m_v ** 2 + 43. * self.mx ** 2) * self.lam_a ** 2) -
                 8 * mass_f ** 9 * self.mx * (self.lam_a ** 4 - self.lam_v ** 4) * (11. * self.m_v **2 + 13 * self.mx ** 2) +
                 mass_f ** 8 * ((105. * self.m_v ** 4 + 136 * self.mx ** 2 * self.m_v ** 2 +46 * self.mx ** 4) *
                                (self.lam_a ** 4 + self.lam_v ** 4) -
                                2 * self.lam_a ** 2 * self.lam_v ** 2 * (61. * self.m_v ** 4 + 128 * self.mx ** 2 * self.m_v ** 2 + 110 * self.mx ** 4))
                 + 2 * mass_f ** 7 * self.mx * (self.lam_a ** 4 - self.lam_v ** 4) * (85 * self.m_v ** 4 + 164 * self.mx ** 2 * 98 * self.mx ** 4) -
                 mass_f ** 6 * ((116 * self.m_v ** 6 + 296 * self.mx ** 2 * self.m_v ** 4 + 108 * self.mx ** 4 * self.m_v **2 + 150 * self.mx ** 6)) -
                 4 * mass_f **5 * self.mx * ((self.lam_a ** 4 - self.lam_v ** 4)*(29*self.m_v**6+117*self.mx**2*self.m_v**4 -
                                                                                  114*self.mx**4.*self.m_v**2 + 46*self.mx**6))+
                 mass_f**4*((46*self.m_v**8 + 332*self.mx**2*self.m_v**6 + 231*self.mx**4*self.m_v**4 -
                             32*self.mx**6 * self.m_v**2 - 49 * self.mx ** 8) * (self.lam_a**4 + self.lam_v**4) -
                            2*self.lam_a**2*self.lam_v**2 * (30*self.m_v**8 + 96*self.mx**2*self.m_v**6 + 159*self.mx**4*self.m_v**4 +
                                                             248*self.mx**6*self.m_v**2 + 115*self.mx**8)) +
                 2*mass_f**3*self.mx * (self.lam_a**4-self.lam_v**4) * (6. * self.m_v**8+122*self.mx**2*self.m_v**6+213*self.mx**4*self.m_v**4 +
                                                                        140*self.mx**6*self.m_v**2 + 43*self.mx**8)
                 +mass_f**2*self.mx**2 * ((-86*self.m_v**8 - 232*self.mx**2*self.m_v**6 - 75*self.mx**4*self.m_v**4 + 74*self.mx**6*self.m_v**2 +
                                          35*self.mx**8)*(self.lam_a**4 + self.lam_v**4) +
                                          2*self.lam_a**2*self.lam_v**2 * (30*self.m_v**8 - 24*self.mx**2*self.m_v**6 + 37*self.mx**4*self.m_v**4 +
                                                                           122*self.mx**6*self.m_v**2 + 47*self.mx**8)) -
                 16*mass_f*self.mx**5 *(self.lam_a**4-self.lam_v**4) *
                 (8*self.m_v**6 + 8*self.mx**2*self.m_v**4 + 4*self.mx**4*self.m_v**2 + self.mx**6) +
                 8*self.mx**4*((8*self.m_v**8 + 2*self.mx**2*self.m_v**6+self.mx**4*self.m_v**4 - 3*self.mx**6*self.m_v**2 - self.mx**8) *
                               (self.lam_a**4 + self.lam_v**4) - 2*self.lam_v**2*self.lam_a**2 * self.mx**2 *
                               (-6*self.m_v**6 - self.mx**2*self.m_v**4 + 3.*self.mx**4*self.m_v**2 + self.mx**6)))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', 'dirac')
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class complex_vector_dm_spin_half_med_tchannel(object):
    """
    Lagrangian = 1/2 [\bar{\psi} \gamma^\mu (g_v + g_a \gamma^5) f X_\mu^\dagger +
    \bar{f} \gamma^mu (g_v + g_a \gamma^5) \psi X_\mu]

    """
    def __init__(self, mx, f, mm, lam_v, lam_a):
        self.mx = mx
        self.mm = mm
        self.lam_v = lam_v
        self.lam_a = lam_a
        self.f = f

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma


    def sigma_v_thermal_approx(self, lam, v):
        sigma = 8. * (lam/2.) ** 4. * self.mx ** 2. / (3. * np.pi * (self.mm ** 2 + self.mx **2) ** 2.) * \
                (1. + 3. * v ** 2. / 32. *
                 (37. + 18 * self.mx ** 2. / self.mm ** 2. + 5. * self.mx ** 4. / self.mm ** 4.) /
                 (1. + self.mx ** 2. / self.mm ** 2) ** 2.)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        nc = color_number(channel)
        mf = get_mass(channel)
        mx = self.mx
        mm = self.mm
        lv = self.lam_v / 2.
        la = self.lam_a / 2.
        if self.mx > mf:
            kin_mass = np.sqrt(1. - (mf / self.mx) ** 2.)
            sv = nc * kin_mass ** 3 / (36 * np.pi * (-mf**2 + mx**2 + mm**2) ** 2) * \
                 (-2*la**2*lv**2 * (3*mf**2 - 12*mx**2 + 5*mm**2) + la**4 * (6*mf*mm + 5*mf**2 +4*mx**2 + 5*mm**2) +
                  lv**4 * (-6*mf*mm + 5*mf**2+4*mx**2+5*mm**2)) + \
                nc * v**2 *kin_mass / (864 * np.pi * mx**2 * (-mf**2+mx**2*mm**2)**4) * \
                (3*mm**6*(la**2-lv**2)**2 *(15*mf**2+16*mx**2) + 6*mf*mm**5*(la**4-lv**4)*(13*mf**2-16*mx**2) +
                 mm**4 * (2*la**2*lv**2 *(-452*mf**2*mx**2 + 87*mf**4+500*mx**4) +
                          la**4*(80*mf**2*mx**2 - 37*mf**4 +92*mx**4) + lv**4*(80*mf**2*mx**2 - 37*mf**4+92*mx**4)) -
                 12*mf*mm**3 * (la**4-lv**4) * (-37*mf**2*mx**2 + 13*mf**4 + 24*mx**4) - mm**2 *(mf**2-mx**2) *
                 (-mf**2 * mx ** 2 * (658*la**2*lv**2 + 239*la**4 + 239*lv**4) + mf**4 *(78*la**2*lv**2 + 61*la**4 + 61*lv**4) +
                  8*mx**4*(50*la**2*lv**2+11*la**4+11*lv**4)) + 6*mf*mm*(la**4-lv**4)*(13*mf**2-8*mx**2)*(mf**2-mx**2)**2 +
                 (mf**2-mx**2)**2*(-2*mf**2*mx**2*(66*la**2*lv**2+41*la**2+41*lv**2) +
                                   mf**4*(-6*la**2*lv**2+53*la**4+53*lv**4) +
                                   20*mx**4*(6*la**2*lv**2+la**4+lv**4)))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', False)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h

    def mediator_width(self):
        mass_f = get_mass('b')
        width = self.lam_v ** 2. * ((self.mx - mass_f - self.mm) * (self.mx + mass_f - self.mm) *
                                    (self.mx - mass_f + self.mm) * (self.mx + mass_f + self.mm)) ** (1. / 2.) *\
                (-2. * self.mx ** 4. + (mass_f**2. - self.mm**2.) ** 2. + self.mx ** 2. *
                 (mass_f**2. + self.mm**2.)) /\
                (8. * np.pi * self.mx**2. * self.mm**3.)
        return width

class real_vector_dm_spin_half_med_tchannel(object):
    """
    Lagrangian = 1/2 (\bar{\psi} \gamma^\mu (g_v + g_a \gamma^5) f X_\mu +
    \bar{f} \gamma^mu (g_v + g_a \gamma^5) \psi X_\mu)

    """
    def __init__(self, mx, f, mm, lam_v, lam_a):
        self.mx = mx
        self.mm = mm
        self.lam_v = lam_v
        self.lam_a = lam_a
        self.f = f

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma


    def sigma_v_thermal_approx(self, lam, v):
        sigma = 32. * (lam/2.) ** 4. * self.mx ** 2. / (3. * np.pi * (self.mm ** 2 + self.mx **2) ** 2.) * \
                (1. + 3. * v ** 2. / 16. *
                 (5. - 10. * self.mx ** 2. / self.mm ** 2. - 7. * self.mx ** 4. / self.mm ** 4.) /
                 (1. + self.mx ** 2. / self.mm ** 2) ** 2.)
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        nc = color_number(channel)
        mf = get_mass(channel)
        mx = self.mx
        mm = self.mm
        lv = self.lam_v
        la = self.lam_a
        if self.mx > mf:
            kin_mass = np.sqrt(1. - (mf / self.mx) ** 2.)
            sv = nc * kin_mass ** 3 / (9 * np.pi * (-mf**2 + mx**2 + mm**2) ** 2) * \
                 (-2*(la/2.)**2*(lv/2.)**2 * (5*mf**2 + 3*mm**2 - 12*mx**2) + (la/2.)**4 * (2*mf*mm + 3*mf**2 +4*mx**2 + 3*mm**2) +
                  (lv / 2.)**4 * (-2*mf*mm + 3*mf**2+4*mx**2+3*mm**2)) + \
                nc * v**2 *kin_mass / (216 * np.pi * mx**2 * (-mf**2+mx**2*mm**2)**4) * \
                (3*mm**6*((la/2.)**2-(lv/2.)**2)**2 *(mf**2+8*mx**2) + 6*mf*mm**5*((la/2.)**4-(lv/2.)**4)*(mf**2-4*mx**2) +
                 mm**4 * (-2*(la/2.)**2*(lv/2.)**2 * (-46*mf**2*mx**2 + 39*mf**4-160*mx**4) +
                          5*(la/2.)**4*(14*mf**2*mx**2 + mf**4) + 5*mf**2*(lv/2.)**4*(mf**2 + 14*mx**2)) +
                 12*mf*mm**3 * ((la/2.)**4-(lv/2.)**4) * (-3*mf**2*mx**2 + mf**4 + 2*mx**4) - mm**2 *(mf**2-mx**2) *
                 (mf**2 * mx ** 2 * (474*(la/2.)**2*(lv/2.)**2 - 29*(la/2.)**4 - 29*(lv/2.)**4) + mf**4 *
                  (-174*(la/2.)**2*(lv/2.)**2 + 19*(la/2.)**4 + 19*(lv/2.)**4)
                  -80*mx**4*(6*(la/2.)**2*(lv/2.)**2+(la/2.)**4+(lv/2.)**4)) - 2*mf*mm*((la/2.)**4-(lv/2.)**4)*(3*mf**2-32*mx**2)*(mf**2-mx**2)**2 +
                 (mf**2-mx**2)**2*(8*mf**2*mx**2*(44*(la/2.)**2*(lv/2.)**2+(la/2.)**4+(lv/2.)**4) +
                                   mf**4*(-90*(la/2.)**2*(lv/2.)**2+11*(la/2.)**4+11*(lv/2.)**4) -
                                   56*mx**4*(6*(la/2.)**2*(lv/2.)**2+(la/2.)**4+(lv/2.)**4)))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.


    def si_cross_section(self, nuclide, v=1.):
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        lam = self.lam_v * 2. / self.mm
        fn = lam * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = lam * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * lam ** 2. / (np.pi) * \
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]


    def integrd_cs(self, eps, x):
        vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', True)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            if exac:
                sv = self.sig_therm_exact(tnew)
            else:
                a = self.sigma_v_all(0.)
                b = self.sigma_v_all(1.)
                b -= a
                sv = a + 9. / 4. * b * np.sqrt(tnew / self.mx)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h_approx(self):
        xfo = self.x_freeze_out(exac=False)
        gstar = effective_dof(self.mx / xfo)
        a = self.sigma_v_all(0.)
        b = self.sigma_v_all(1.)
        b -= a
        o_h = 1.07 * 10 ** 9. * xfo / (m_planck * np.sqrt(gstar) * (a + 3. * b / xfo))
        return o_h

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h

    def mediator_width(self):
        mass_f = get_mass('b')
        width = self.lam_v ** 2. * ((self.mx - mass_f - self.mm) * (self.mx + mass_f - self.mm) *
                                    (self.mx - mass_f + self.mm) * (self.mx + mass_f + self.mm)) ** (1. / 2.) *\
                (-2. * self.mx ** 4. + (mass_f**2. - self.mm**2.) ** 2. + self.mx ** 2. *
                 (mass_f**2. + self.mm**2.)) /\
                (8. * np.pi * self.mx**2. * self.mm**3.)
        return width