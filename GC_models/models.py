import numpy as np
from helper import *
from scipy.integrate import quad
from scipy.special import kn

Pi = np.pi
yuk = np.sqrt(2.) / 246.

class fermionic_dm_spin0_med_schannel(object):
    """
    Lagrangian = [\bar{\chi} (\lamba_{\chi,s} + \lambda_{\chi,p} i \gamma^5) \chi +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """
    def __init__(self, mx, dm_type, f, m_a, lam_chi_s, lam_chi_p, lam_f_s, lam_f_p, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_type = dm_type
        self.m_a = m_a
        self.lam_chi_s = lam_chi_s
        self.lam_chi_p = lam_chi_p
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = (3. * lam ** 4. * self.mx ** 2. /
                                      (np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) *
                (1. + 9. * v ** 2. / (8. * (1. - 4. * self.mx ** 2 / self.m_a ** 2.))))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        sv = 0.0
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f * yuk
        lam_f_p = self.lam_f_p * mass_f * yuk
        if self.mx > mass_f:
            gxp = self.lam_chi_p
            gxs = self.lam_chi_s
            gfp = lam_f_p
            gfs = lam_f_s

            if up_like(channel):
                gfs /= self.tbeta
                gfp /= self.tbeta
            
            sv += nc*((gxp**2*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                      (self.mx**2*gfp**2 + self.mx**2*gfs**2 - gfs**2*mass_f**2))/(2.*np.sqrt(self.mx**4)*
                                                                                   (4*self.mx**2 - self.m_a**2)**2*Pi) +
                  (((4*self.mx**2*(self.mx**2*gfp**2 + self.mx**2*gfs**2)*gxp**2*
                     np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2))/(np.sqrt(self.mx**4)*
                                                                      (-4*self.mx**2 + self.m_a**2)**2) +
                    (4*self.mx**2*gfp**2 + 4*self.mx**2*gfs**2 - 4*gfs**2*mass_f**2)*
                    (((self.mx**2*gxp**2 + self.mx**2*gxs**2)*np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2))/
                     (np.sqrt(self.mx**4)*(-4*self.mx**2 + self.m_a**2)**2) + 4*self.mx**2*gxp**2*
                     ((2*self.mx**4 - self.mx**2*mass_f**2)/(4.*np.sqrt(self.mx**4)*
                                                             np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                                                             (-4*self.mx**2 + self.m_a**2)**2) +
                      np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*(-1/(4.*np.sqrt(self.mx**4)*
                                                                          (-4*self.mx**2 + self.m_a**2)**2) +
                                                                      ((2*self.mx**2)/(-4*self.mx**2 + self.m_a**2)**3 -
                                                                       1/(4.*(-4*self.mx**2 + self.m_a**2)**2))/
                                                                      np.sqrt(self.mx**4)))))*v**2)/(64.*self.mx**2*Pi))
        
        if self.m_a < self.mx:
            gxp = np.sqrt(self.c_ratio * self.lam_chi_p * (self.lam_f_p + self.lam_f_p))
            gxs = np.sqrt(self.c_ratio * self.lam_chi_s * (self.lam_f_s + self.lam_f_p))

            sv += (1./totferms) * ((np.sqrt(self.mx**4)*gxp**2*gxs**2*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2)))/\
                  (2.*self.mx**2*(2*self.mx**2 - self.m_a**2)**2*Pi) + \
                  (np.sqrt(self.mx**4)*(2*self.mx**6*gxp**4 - 36*self.mx**6*gxp**2*gxs**2 +
                                        18*self.mx**6*gxs**4 - 6*self.mx**4*gxp**4*self.m_a**2 +
                                        60*self.mx**4*gxp**2*gxs**2*self.m_a**2 -
                                        34*self.mx**4*gxs**4*self.m_a**2 + 6*self.mx**2*gxp**4*self.m_a**4 -
                                        24*self.mx**2*gxp**2*gxs**2*self.m_a**4 + 20*self.mx**2*gxs**4*self.m_a**4 -
                                        2*gxp**4*self.m_a**6 + 3*gxp**2*gxs**2*self.m_a**6 -
                                        4*gxs**4*self.m_a**6)*v**2)/\
                  (48.*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2))*(2*self.mx**2 - self.m_a**2)**4*Pi))

        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.1:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            # print 'T_{i}: ', told, 'T_{i+1}:', tnew
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.
        gxs = np.sqrt(self.c_ratio * self.lam_chi_s * (self.lam_f_p + self.lam_f_s))
        gxp = np.sqrt(self.c_ratio * self.lam_chi_p * (self.lam_f_p + self.lam_f_s))
        if dm:
            if 2. * self.mx < self.m_a:
                width += ((-4 * self.mx ** 2 * gxs ** 2 + (gxp ** 2 + gxs ** 2) * self.m_a ** 2) *
                          np.sqrt(-4. * self.mx ** 2 * self.m_a ** 2 + self.m_a ** 4)) /\
                         (8. * (self.m_a ** 2) ** 1.5 * Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfs = mass_f * np.sqrt(self.lam_f_s * (self.lam_chi_p + self.lam_chi_s) / self.c_ratio) * yuk
                gfp = mass_f * np.sqrt(self.lam_f_p * (self.lam_chi_p + self.lam_chi_s) / self.c_ratio) * yuk
                if up_like(f):
                    gfs /= self.tbeta
                    gfp /= self.tbeta

                if 2. * mass_f < self.m_a:
                    width += nc*((-4*gfs**2*mass_f**2 + (gfp**2 + gfs**2)*self.m_a**2)*
                              np.sqrt(-4*mass_f**2*self.m_a**2 + self.m_a**4))/(8.*(self.m_a**2)**1.5*Pi)
        return width


class fermionic_dm_spin1_med_schannel(object):
    """
    Lagrangian = [\bar{\chi} \gamma^\mu (\lamba_{\chi,v} + \lambda_{\chi,a} \gamma^5) \chi +
    \bar{f}\gamma^\mu (\lamba_{f,v} + \lambda_{f,a} \gamma^5) f]V_\mu
    """
    def __init__(self, mx, dm_type, f, m_v, lam_chi_v, lam_chi_a, lam_f_v, lam_f_a, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_type = dm_type
        self.m_v = m_v
        self.lam_chi_v = lam_chi_v
        self.lam_chi_a = lam_chi_a
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta

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



    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
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
            sv += nc * ((np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                         (16*gfa**2*gxa**2*self.mx**4*mass_f**2 - 8*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**2 +
                          2*gfa**2*gxv**2*self.mx**2*self.m_v**4 + 2*gfv**2*gxv**2*self.mx**2*self.m_v**4 +
                          gfa**2*gxa**2*mass_f**2*self.m_v**4 - 2*gfa**2*gxv**2*mass_f**2*self.m_v**4 +
                          gfv**2*gxv**2*mass_f**2*self.m_v**4))/(2.*np.sqrt(self.mx**4)*self.m_v**4*
                                                                 (4*self.mx**2 - self.m_v**2)**2*Pi) +
                        (((np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*
                           (96*gfa**2*gxa**2*self.mx**6*mass_f**2 - 24*gfa**2*gxa**2*self.mx**4*mass_f**2*self.m_v**2 +
                            4*gfa**2*gxa**2*self.mx**4*self.m_v**4 + 4*gfv**2*gxa**2*self.mx**4*self.m_v**4 +
                            10*gfa**2*gxv**2*self.mx**4*self.m_v**4 + 10*gfv**2*gxv**2*self.mx**4*self.m_v**4 -
                            4*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 +
                            2*gfv**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 -
                            4*gfa**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4 +
                            2*gfv**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4))/(np.sqrt(self.mx**4)*
                                                                                (-4*self.mx**2 + self.m_v**2)**2) +
                          (192*gfa**2*gxa**2*self.mx**6*mass_f**2 - 96*gfa**2*gxa**2*self.mx**4*mass_f**2*self.m_v**2 +
                           24*gfa**2*gxv**2*self.mx**4*self.m_v**4 + 24*gfv**2*gxv**2*self.mx**4*self.m_v**4 +
                           12*gfa**2*gxa**2*self.mx**2*mass_f**2*self.m_v**4 -
                           24*gfa**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4 +
                           12*gfv**2*gxv**2*self.mx**2*mass_f**2*self.m_v**4)*
                          ((2*self.mx**4 - self.mx**2*mass_f**2)/(4.*np.sqrt(self.mx**4)*
                                                                  np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                                                                  (-4*self.mx**2 + self.m_v**2)**2) +
                           np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*(-1/(4.*np.sqrt(self.mx**4)*
                                                                               (-4*self.mx**2 + self.m_v**2)**2) +
                                                                           ((2*self.mx**2)/
                                                                            (-4*self.mx**2 + self.m_v**2)**3 -
                                                                            1/(4.*(-4*self.mx**2 + self.m_v**2)**2))/
                                                                           np.sqrt(self.mx**4))))*v**2)/
                        (48.*self.mx**2*self.m_v**4*Pi))
        
        if self.m_v < self.mx:
            gxa = np.sqrt(self.c_ratio * self.lam_chi_a * (self.lam_f_a + self.lam_f_v))
            gxv = np.sqrt(self.c_ratio * self.lam_chi_v * (self.lam_f_v + self.lam_f_a))
            
            sv += (1./totferms)*((np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*(8*gxa**2*gxv**2*self.mx**4 +
                                                                   gxa**4*self.mx**2*self.m_v**2 -
                                                                   14*gxa**2*gxv**2*self.mx**2*self.m_v**2 +
                                                                   gxv**4*self.mx**2*self.m_v**2 -
                                                                   gxa**4*self.m_v**4 + 6*gxa**2*gxv**2*self.m_v**4 -
                                                                   gxv**4*self.m_v**4))/\
                  (4.*np.sqrt(self.mx**4)*self.m_v**2*(2*self.mx**2 - self.m_v**2)**2*Pi) + \
                  (np.sqrt(self.mx**4)*(128*gxa**4*self.mx**12 - 320*gxa**4*self.mx**10*self.m_v**2 -
                                        128*gxa**2*gxv**2*self.mx**10*self.m_v**2 + 248*gxa**4*self.mx**8*self.m_v**4 +
                                        656*gxa**2*gxv**2*self.mx**8*self.m_v**4 + 24*gxv**4*self.mx**8*self.m_v**4 +
                                        68*gxa**4*self.mx**6*self.m_v**6 - 1016*gxa**2*gxv**2*self.mx**6*self.m_v**6 +
                                        4*gxv**4*self.mx**6*self.m_v**6 - 212*gxa**4*self.mx**4*self.m_v**8 +
                                        624*gxa**2*gxv**2*self.mx**4*self.m_v**8 - 64*gxv**4*self.mx**4*self.m_v**8 +
                                        105*gxa**4*self.mx**2*self.m_v**10 - 142*gxa**2*gxv**2*self.mx**2*self.m_v**10 +
                                        53*gxv**4*self.mx**2*self.m_v**10 - 17*gxa**4*self.m_v**12 +
                                        6*gxa**2*gxv**2*self.m_v**12 - 17*gxv**4*self.m_v**12)*v**2)/\
                  (96.*self.mx**2*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                   (2*self.mx**2 - self.m_v**2)**4*Pi))

        return sv

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 6. * lam ** 4. * self.mx ** 2. / (np.pi * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2.) *\
                (1. + 3. * v ** 2. * (1. + 2. * self.mx ** 2. / self.m_v ** 2.) /
                 (4. * (1. - 4. * self.mx ** 2 / self.m_v ** 2.)))
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.1:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.

        gxa = np.sqrt(self.c_ratio * self.lam_chi_a*(self.lam_f_a+self.lam_f_v))
        gxv = np.sqrt(self.c_ratio * self.lam_chi_v*(self.lam_f_a+self.lam_f_v))

        if dm:
            if 2.*self.mx < self.m_v:
                width += ((2*(-2*gxa**2 + gxv**2)*self.mx**2 + (gxa**2 + gxv**2)*self.m_v**2)*
                         np.sqrt(-4.*self.mx**2*self.m_v**2 + self.m_v**4))/(12.*(self.m_v**2)**1.5*Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfa = np.sqrt((self.lam_chi_a + self.lam_chi_v)*self.lam_f_a / self.c_ratio)
                gfv = np.sqrt((self.lam_chi_a + self.lam_chi_v)*self.lam_f_v / self.c_ratio)
                if up_like(f):
                    gfa /= self.tbeta
                    gfv /= self.tbeta
                if 2.*mass_f < self.m_v:
                    width += nc*((2*(-2*gfa**2 + gfv**2)*mass_f**2 + (gfa**2 + gfv**2)*self.m_v**2)*
                              np.sqrt(-4.*mass_f**2*self.m_v**2 + self.m_v**4))/(12.*(self.m_v**2)**1.5*Pi)

        return width


class scalar_dm_spin0_med_schannel(object):
    """
    Lagrangian = [\lambda_phi |\phi|^2 + \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """

    def __init__(self, mx, dm_real, f, m_a, lam_phi, lam_f_s, lam_f_p, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_real = dm_real
        self.m_a = m_a
        self.lam_p = lam_phi
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f * yuk
            lam_f_p = self.lam_f_p * mass_f * yuk
            sigma += (nc * self.lam_p ** 2. / (8. * np.pi * s * ((s - self.m_a ** 2) ** 2 + (self.m_a * gamma) ** 2.)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                     (lam_f_s **2. * (s - 4. * mass_f ** 2.) + s * lam_f_p ** 2.))
        return sigma



    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f * yuk
        lam_f_p = self.lam_f_p * mass_f * yuk
        if self.mx > mass_f:
            gx = self.lam_p
            gfp = lam_f_p
            gfs = lam_f_s
            sv += nc * ((gx**2*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                         (self.mx**2*gfp**2 + self.mx**2*gfs**2 - gfs**2*mass_f**2))/
                        (4.*self.mx**2*np.sqrt(self.mx**4)*(4*self.mx**2 - self.m_a**2)**2*Pi) -
                        (gx**2*(16*self.mx**6*gfp**2 + 16*self.mx**6*gfs**2 - 20*self.mx**4*gfp**2*mass_f**2 -
                                44*self.mx**4*gfs**2*mass_f**2 + 28*self.mx**2*gfs**2*mass_f**4 +
                                self.mx**2*gfp**2*mass_f**2*self.m_a**2 + 3*self.mx**2*gfs**2*mass_f**2*self.m_a**2 -
                                3*gfs**2*mass_f**4*self.m_a**2)*v**2)/
                        (32.*np.sqrt(self.mx**4)*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                         (4*self.mx**2 - self.m_a**2)**3*Pi))
        if self.m_a < self.mx:
            gx = np.sqrt(self.c_ratio * (self.lam_f_s * self.lam_p + self.lam_p * self.lam_f_p))

            sv += (1./totferms)*((gx**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2)))/\
                  (16.*self.mx**2*np.sqrt(self.mx**4)*
                   (2*self.mx**2 - self.m_a**2)**2*Pi) - \
                  (gx**4*(56*self.mx**6 - 100*self.mx**4*self.m_a**2 +
                          50*self.mx**2*self.m_a**4 - 9*self.m_a**6)*v**2)/\
                  (384.*np.sqrt(self.mx**4)*np.sqrt(self.mx**2*(self.mx**2 - self.m_a**2))*
                   (2*self.mx**2 - self.m_a**2)**4*Pi))

        return sv

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.01:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.
        gx = np.sqrt(self.c_ratio * self.lam_p * (self.lam_f_p + self.lam_f_s))
        if dm:
            if 2. * self.mx < self.m_a:
                width += (gx**2*np.sqrt(-4*self.mx**2*self.m_a**2 + self.m_a**4))/(16.*(self.m_a**2)**1.5*Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfs = mass_f * np.sqrt(self.lam_f_s * self.lam_p / self.c_ratio) * yuk
                gfp = mass_f * np.sqrt(self.lam_f_p * self.lam_p / self.c_ratio) * yuk
                if up_like(f):
                    gfs /= self.tbeta
                    gfp /= self.tbeta
                if 2. * mass_f < self.m_a:
                    width += nc * (((-4*gfs**2*mass_f**2 + (gfp**2 + gfs**2)*self.m_a**2)*
                                    np.sqrt(-4*mass_f**2*self.m_a**2 + self.m_a**4))/(8.*(self.m_a**2)**1.5*Pi))
        return width


class scalar_dm_spin1_med_schannel(object):
    """
    Lagrangian = [i \lambda_p \phi^\dag \dderiv_\mu \phi +
    \bar{f}\gamma^\mu (\lamba_{f,v} + \lambda_{f,a} \gamma^5) f]V_\mu
    """
    def __init__(self, mx, dm_real, f, m_v, lam_p, lam_f_v, lam_f_a, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_real = dm_real
        self.m_v = m_v
        self.lam_p = lam_p
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta
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


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            gx = self.lam_p
            gfa = self.lam_f_a
            gfv = self.lam_f_v
            sv += nc * ((gx**2*np.sqrt(4*self.mx**4 - 4*self.mx**2*mass_f**2)*
                         (4*self.mx**2*gfa**2 + 4*self.mx**2*gfv**2 - 4*gfa**2*mass_f**2 + 2*gfv**2*mass_f**2)*v**2)/
                        (48.*np.sqrt(self.mx**4)*(-4*self.mx**2 + self.m_v**2)**2*Pi))

        if self.m_v < self.mx:
            gx = np.sqrt(self.c_ratio * (self.lam_f_v * self.lam_p + self.lam_p * self.lam_f_a))
            
            sv += (1./totferms)*((gx**4*(self.mx**10*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2)) -
                          2*self.mx**8*self.m_v**2*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2)) +
                          self.mx**6*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))))/\
                  ((self.mx**4)**1.5*self.m_v**4*(2*self.mx**2 - self.m_v**2)**2*Pi) + \
                  (np.sqrt(self.mx**4)*gx**4*(24*self.mx**10 - 60*self.mx**8*self.m_v**2 +
                                              58*self.mx**6*self.m_v**4 - 35*self.mx**4*self.m_v**6 +
                                              16*self.mx**2*self.m_v**8 - 3*self.m_v**10)*v**2)/\
                  (24.*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*(2*self.mx**2 - self.m_v**2)**4*Pi))
            
        return sv

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.01:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.
        gx = np.sqrt(self.c_ratio * self.lam_p * (self.lam_f_v + self.lam_f_a))
        if dm:
            if 2. * self.mx < self.m_v:
                width += (gx**2*(-4*self.mx**2 + self.m_v**2)*
                         np.sqrt(-4.*self.mx**2*self.m_v**2 + self.m_v**4))/(48.*(self.m_v**2)**1.5*Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfa = np.sqrt(self.lam_f_a * self.lam_p / self.c_ratio)
                gfv = np.sqrt(self.lam_f_v * self.lam_p / self.c_ratio)
                if up_like(f):
                    gfv /= self.tbeta
                    gfa /= self.tbeta
                if 2. * mass_f < self.m_v:
                    width += nc * ((2*(-2*gfa**2 + gfv**2)*mass_f**2 + (gfa**2 + gfv**2)*self.m_v**2)*
                                   np.sqrt(-4.*mass_f**2*self.m_v**2 + self.m_v**4))/(12.*(self.m_v**2)**1.5*Pi)
        return width


class vector_dm_spin0_med_schannel(object):
    """
    Lagrangian = [\lamba_x X^\mu X_\mu^\dag +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """
    def __init__(self, mx, dm_real, f, m_a, lam_x, lam_f_s, lam_f_p, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_real = dm_real
        self.m_a = m_a
        self.lam_x = lam_x
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            lam_f_s = self.lam_f_s * mass_f * yuk
            lam_f_p = self.lam_f_p * mass_f * yuk
            sigma += (nc * self.lam_x ** 2. / (72. * np.pi * ((s - self.m_a ** 2.) ** 2 + (self.m_a * gamma) ** 2)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2 / s)) *
                     (s / self.mx ** 2 * (s / (4. * self.mx ** 2) - 1.) + 3.) *
                     (lam_f_s ** 2. * (1. - 4. * mass_f ** 2. / s) + lam_f_p ** 2.))
        return sigma


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_s = self.lam_f_s * mass_f * yuk
        lam_f_p = self.lam_f_p * mass_f * yuk
        if self.mx > mass_f:
            gx = self.lam_x
            gfp = lam_f_p
            gfs = lam_f_s
            sv = nc * ((np.sqrt(self.mx**4)*gx**2*np.sqrt(self.mx**2*(self.mx**2 - 1.*mass_f**2))*
                        (self.mx**2*gfp**2 + self.mx**2*gfs**2 - gfs**2*mass_f**2))/
                       (12.*self.mx**6*(4*self.mx**2 - self.m_a**2)**2*Pi) -
                       (gx**2*(16*self.mx**6*gfp**2 + 16*self.mx**6*gfs**2 - 28*self.mx**4*gfp**2*mass_f**2 -
                               68*self.mx**4*gfs**2*mass_f**2 + 52*self.mx**2*gfs**2*mass_f**4 +
                               8*self.mx**4*gfp**2*self.m_a**2 + 8*self.mx**4*gfs**2*self.m_a**2 -
                               5*self.mx**2*gfp**2*mass_f**2*self.m_a**2 - 7*self.mx**2*gfs**2*mass_f**2*self.m_a**2 -
                               gfs**2*mass_f**4*self.m_a**2)*v**2)/
                       (288.*np.sqrt(self.mx**4)*np.sqrt(self.mx**2*(self.mx**2 - 1.*mass_f**2))*
                        (4*self.mx**2 - self.m_a**2)**3*Pi))
        
        if self.m_a < self.mx:
            gx = np.sqrt(self.c_ratio * (self.lam_f_s * self.lam_x + self.lam_x * self.lam_f_p))

            sv += (1./totferms)*((np.sqrt(self.mx**4)*gx**4*np.sqrt(self.mx**2*(self.mx**2 - 1.*self.m_a**2))*
                   (6*self.mx**4 - 4*self.mx**2*self.m_a**2 + self.m_a**4))/\
                  (9.*self.mx**10*(2*self.mx**2 - self.m_a**2)**2*Pi) - \
                  (gx**4*(80*self.mx**10 - 232*self.mx**8*self.m_a**2 + 260*self.mx**6*self.m_a**4 -
                          158*self.mx**4*self.m_a**6 + 46*self.mx**2*self.m_a**8 - 5*self.m_a**10)*v**2)/\
                  (216.*(self.mx**4)**1.5*np.sqrt(self.mx**2*(self.mx**2 - 1.*self.m_a**2))*
                   (2*self.mx**2 - self.m_a**2)**4*Pi))
        
        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.1:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.
        gx = np.sqrt(self.c_ratio * self.lam_x * (self.lam_f_s + self.lam_f_p))
        if dm:
            if 2. * self.mx < self.m_a:
                width += (gx**2*np.sqrt(-4.*self.mx**2*self.m_a**2 + self.m_a**4)*
                         (12*self.mx**4 - 4*self.mx**2*self.m_a**2 + self.m_a**4))/\
                         (64.*self.mx**4*(self.m_a**2)**1.5*Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfs = mass_f * np.sqrt(self.lam_f_s * self.lam_x / self.c_ratio) * yuk
                gfp = mass_f * np.sqrt(self.lam_f_p * self.lam_x / self.c_ratio) * yuk
                if up_like(f):
                    gfs /= self.tbeta
                    gfp /= self.tbeta

                if 2. * mass_f < self.m_a:
                    width += nc * ((-4*gfs**2*mass_f**2 + (gfp**2 + gfs**2)*self.m_a**2)*
                                   np.sqrt(-4.*mass_f**2*self.m_a**2 + self.m_a**4))/(8.*(self.m_a**2)**1.5*Pi)
        return width

class vector_dm_spin1_med_schannel(object):
    """
    Lagrangian = [\lamba_x (X^{\nu,\dag} \del_\nu X^\mu + h.c. ) +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] V_\mu
    """
    def __init__(self, mx, dm_real, f, m_v, lam_x, lam_f_v, lam_f_a, c_ratio=0., tbeta=1.):
        self.mx = mx
        self.dm_real = dm_real
        self.m_v = m_v
        self.lam_x = lam_x
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        self.c_ratio = c_ratio
        self.tbeta = tbeta

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


    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v, totferms=len(self.f))
        return sigma

    def sigma_v(self, channel, v, totferms=1.):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        lam_f_a = self.lam_f_a
        lam_f_v = self.lam_f_v
        if self.mx > mass_f:
            gx = self.lam_x
            gfa = lam_f_a
            gfv = lam_f_v
            
            sv += nc*((4*gx**2*np.sqrt(self.mx**2*(self.mx**2 - mass_f**2))*
                   (2*self.mx**2*gfa**2 + 2*self.mx**2*gfv**2 - 2*gfa**2*mass_f**2 + gfv**2*mass_f**2)*v**2)/\
                  (27.*np.sqrt(self.mx**4)*(4*self.mx**2 - self.m_v**2)**2*Pi))

        if self.m_v < self.mx:
            gx = np.sqrt(self.c_ratio * (self.lam_f_v * self.lam_x + self.lam_x * self.lam_f_a))

            sv += (1./totferms)*((np.sqrt(self.mx**4)*gx**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                  (18*self.mx**8 - 38*self.mx**6*self.m_v**2 + 35*self.mx**4*self.m_v**4 -
                   15*self.mx**2*self.m_v**6 + 3*self.m_v**8))/\
                 (72.*self.mx**6*self.m_v**4*(2*self.mx**2 - self.m_v**2)**2*Pi) + \
                 (gx**4*(624*self.mx**14 - 1752*self.mx**12*self.m_v**2 + 1864*self.mx**10*self.m_v**4 -
                         818*self.mx**8*self.m_v**6 - 52*self.mx**6*self.m_v**8 + 211*self.mx**4*self.m_v**10 -
                         77*self.mx**2*self.m_v**12 + 9*self.m_v**14)*v**2)/\
                 (1728.*np.sqrt(self.mx**4)*self.m_v**4*np.sqrt(self.mx**2*(self.mx**2 - self.m_v**2))*
                  (2*self.mx**2 - self.m_v**2)**4*Pi))

        return sv


    def sig_therm_exact(self, temp):
        x = self.mx / temp
        #them_avg = quad(self.integrd_cs, 0., np.inf, args=x)
        them_avg = quad(self.integrd_cs, 0., 1., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def integrd_cs(self, v, x):
        #vrel = 2. * np.sqrt(eps) * np.sqrt(1. + eps) / (1. + 2. * eps)
        #return np.sqrt(eps) * np.exp(- x * eps) * self.sigma_v_all(vrel)
        eps = v**2. / (2. - 2.*v**2.+2.*np.sqrt(1. - v**2.))
        return np.sqrt(eps) * np.exp(-x * eps) * self.sigma_v_all(v) * v / (2.*(1.-v**2.)**(3./2.))

    def x_freeze_out(self, exac=True):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.1:
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
            tnew = self.mx / np.log(np.abs(0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
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

    def mediator_width(self, ferms=True, dm=True):
        if self.c_ratio == 0.:
            self.c_ratio = 1.
        width = 0.
        gx = np.sqrt(self.c_ratio * self.lam_x * (self.lam_f_a + self.lam_f_v))
        if dm:
            if 2. * self.mx < self.m_v:
                width += (gx**2*(-4*self.mx**2 + self.m_v**2)*np.sqrt(-4.*self.mx**2*self.m_v**2 + self.m_v**4))/\
                         (48.*self.mx**2*np.sqrt(self.m_v**2)*Pi)
        if ferms:
            for f in self.f:
                mass_f = get_mass(f)
                nc = color_number(f)
                gfa = np.sqrt(self.lam_f_a * self.lam_x / self.c_ratio)
                gfv = np.sqrt(self.lam_f_v * self.lam_x / self.c_ratio)
                if up_like(f):
                    gfa /= self.tbeta
                    gfv /= self.tbeta
                if 2. * mass_f < self.m_v:
                    width += nc * ((2*(-2*gfa**2 + gfv**2)*mass_f**2 + (gfa**2 + gfv**2)*self.m_v**2)*
                                   np.sqrt(-4.*mass_f**2*self.m_v**2 + self.m_v**4))/(3.*(self.m_v**2)**1.5*Pi)
        return width


class dirac_fermionic_dm_spin0_med_tchannel(object):
    """
    Lagrangian = [\bar{\chi} (\lambda_s + \lambda_p \gamma^5)f A +
    \bar{f} (\lambda_s - \lambda_p \gamma^5) \chi A^\dagger]

    ONLY FOR DIRAC FERMION! Majorana has a different formula!
    """
    def __init__(self, mx, f, m_a, lam_s, lam_p):
        self.mx = mx
        self.m_a = m_a
        self.lam_s = lam_s
        self.lam_p = lam_p
        self.f = f


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
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)

        gDM = self.lam_s
        if self.mx > mass_f:
            sv += (3*gDM**4*np.sqrt(self.mx**4)*np.sqrt(self.mx**2*(-mass_f**2 + self.mx**2)))/\
                  (32.*self.mx**2*(-mass_f**2 + self.mx**2 + self.m_a**2)**2*Pi) - \
                  (gDM**4*np.sqrt(self.mx**4)*(-2*mass_f**8 - 5*mass_f**6*self.mx**2 + 24*mass_f**4*self.mx**4 -
                                               25*mass_f**2*self.mx**6 + 8*self.mx**8 + 4*mass_f**6*self.m_a**2 -
                                               2*mass_f**4*self.mx**2*self.m_a**2 -
                                               26*mass_f**2*self.mx**4*self.m_a**2 + 24*self.mx**6*self.m_a**2 -
                                               2*mass_f**4*self.m_a**4 + 7*mass_f**2*self.mx**2*self.m_a**4 -
                                               8*self.mx**4*self.m_a**4)*v**2)/\
                  (256.*self.mx**2*np.sqrt(self.mx**2*(-mass_f**2 + self.mx**2))*
                   (-mass_f**2 + self.mx**2 + self.m_a**2)**4*Pi)
        
        return sv
    
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

    def mediator_width(self, ferms=True, dm=True):
        width = 0.
        gDM = self.lam_s
        for f in self.f:
            mass_f = get_mass(f)
            width += -(gDM**2*np.sqrt((mass_f - self.mx - self.m_a)*
                                      (mass_f + self.mx - self.m_a)*(mass_f - self.mx + self.m_a)*
                                      (mass_f + self.mx + self.m_a))*
                       (mass_f**2 + self.mx**2 - self.m_a**2))/(16.*(self.m_a**2)**1.5*Pi)
        return width


class dirac_fermionic_dm_spin1_med_tchannel(object):
    """
    Lagrangian = 1/2 [\bar{\chi} \gamma^\mu (g_\chi,s + g_\chi,p \gamma^5)f V_\mu +
    \bar{f} \gamma^\mu (g_f,s + g_f,p \gamma^5) \chi V_\mu^\dagger]

    """
    def __init__(self, mx, f, m_v, lam_v, lam_a):
        self.mx = mx
        self.m_v = m_v
        self.lam_v = lam_v
        self.lam_a = lam_a
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
        sv = 0.
        nc = color_number(channel)
        mass_f = get_mass(channel)
        gd = np.mean([self.lam_v, self.lam_a])
        if self.mx > mass_f:
            sv += (3*gd**4*np.sqrt(self.mx**2*(self.mx**2 - 1.*mass_f**2))*
                   (self.mx**6 - 2*self.mx**4*mass_f**2 + self.mx**2*mass_f**4 + 4*self.mx**4*self.m_v**2 -
                    6*self.mx**2*mass_f**2*self.m_v**2 + 2*mass_f**4*self.m_v**2 + 4*self.mx**2*self.m_v**4))\
                  /(32.*np.sqrt(self.mx**4)*self.m_v**4*(self.mx**2 - mass_f**2 + self.m_v**2)**2*Pi) - \
                  (np.sqrt(self.mx**4)*gd**4*(8*self.mx**12 - 41*self.mx**10*mass_f**2 + 82*self.mx**8*mass_f**4 -
                                              78*self.mx**6*mass_f**6 + 32*self.mx**4*mass_f**8 - self.mx**2*mass_f**10
                                              - 2*mass_f**12 + 24*self.mx**10*self.m_v**2 -
                                              98*self.mx**8*mass_f**2*self.m_v**2 +
                                              140*self.mx**6*mass_f**4*self.m_v**2 -
                                              72*self.mx**4*mass_f**6*self.m_v**2 - 4*self.mx**2*mass_f**8*self.m_v**2 +
                                              10*mass_f**10*self.m_v**2 - 8*self.mx**8*self.m_v**4 +
                                              19*self.mx**6*mass_f**2*self.m_v**4 -
                                              36*self.mx**4*mass_f**4*self.m_v**4 +
                                              47*self.mx**2*mass_f**6*self.m_v**4 - 22*mass_f**8*self.m_v**4 -
                                              32*self.mx**6*self.m_v**6 + 128*self.mx**4*mass_f**2*self.m_v**6 -
                                              118*self.mx**2*mass_f**4*self.m_v**6 + 22*mass_f**6*self.m_v**6 -
                                              32*self.mx**4*self.m_v**8 + 28*self.mx**2*mass_f**2*self.m_v**8 -
                                              8*mass_f**4*self.m_v**8)*v**2)/\
                  (256.*self.mx**2*np.sqrt(self.mx**2*(self.mx**2 - 1.*mass_f**2))*
                   self.m_v**4*(self.mx**2 - mass_f**2 + self.m_v**2)**4*Pi)
    
        return sv
    
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

    def mediator_width(self, ferms=True, dm=True):
        width = 0.
        gDM = np.sqrt(self.lam_v * self.lam_a)
        for f in self.f:
            mass_f = get_mass(f)
            width += -(gDM**2*np.sqrt((self.mx - mass_f - self.m_v)*
                                      (self.mx + mass_f - self.m_v)*(self.mx - mass_f + self.m_v)*
                                      (self.mx + mass_f + self.m_v))*((self.mx**2 - mass_f**2)**2 +
                                                                      (self.mx**2 + mass_f**2)*
                                                                      self.m_v**2 - 2*self.m_v**4))/\
                     (48.*self.m_v**4*np.sqrt(self.m_v**2)*Pi)
        return width


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
        sv = 0.
        nc = color_number(channel)
        mf = get_mass(channel)
        mx = self.mx
        mm = self.mm
        lv = self.lam_v
        la = self.lam_a
        coup = np.mean([lv, la])
        if self.mx > mf:
            sv += (np.sqrt(mx**4)*coup**4*np.sqrt(mx**2*(mx**2 - mf**2))*(8*mx**4 - 7*mx**2*mf**2 - mf**4))/\
                  (48.*mx**6*(mx**2 - mf**2 + mm**2)**2*Pi) + \
                  (coup**4*(40*mx**10 - 194*mx**8*mf**2 + 367*mx**6*mf**4 - 337*mx**4*mf**6 + 149*mx**2*mf**8 -
                            25*mf**10 + 144*mx**8*mm**2 - 572*mx**6*mf**2*mm**2 + 762*mx**4*mf**4*mm**2 -
                            384*mx**2*mf**6*mm**2 + 50*mf**8*mm**2 + 296*mx**6*mm**4 - 482*mx**4*mf**2*mm**4 +
                            211*mx**2*mf**4*mm**4 - 25*mf**6*mm**4)*v**2)/(1152.*np.sqrt(mx**4)*
                                                                           np.sqrt(mx**2*(mx**2 - mf**2))*
                                                                           (mx**2 - mf**2 + mm**2)**4*Pi)

        return sv


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

    def mediator_width(self, ferms=True, dm=True):
        width = 0.
        for f in self.f:
            mass_f = get_mass(f)
            width += self.lam_v ** 2. * ((self.mx - mass_f - self.mm) * (self.mx + mass_f - self.mm) *
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
        sv = 0.
        nc = color_number(channel)
        mf = get_mass(channel)
        mx = self.mx
        mm = self.mm
        lv = self.lam_v
        la = self.lam_a
        lamd = np.mean([lv, la])
        if self.mx > mf:
            sv += (np.sqrt(self.mx**4)*lamd**4*np.sqrt(self.mx**2*(self.mx**2 - mf**2))*
                   (8*self.mx**4 - 9*self.mx**2*mf**2 + mf**4))/\
                  (12.*self.mx**6*(self.mx**2 - mf**2 + self.mm**2)**2*Pi) - \
                  (lamd**4*(112*self.mx**10 - 428*self.mx**8*mf**2 + 629*self.mx**6*mf**4 -
                            439*self.mx**4*mf**6 + 143*self.mx**2*mf**8 - 17*mf**10 + 160*self.mx**8*self.mm**2 -
                            424*self.mx**6*mf**2*self.mm**2 + 402*self.mx**4*mf**4*self.mm**2 -
                            172*self.mx**2*mf**6*self.mm**2 + 34*mf**8*self.mm**2 - 80*self.mx**6*self.mm**4 +
                            68*self.mx**4*mf**2*self.mm**4 + 29*self.mx**2*mf**4*self.mm**4 -
                            17*mf**6*self.mm**4)*v**2)/(288.*np.sqrt(self.mx**4)*
                                                        np.sqrt(self.mx**2*(self.mx**2 - mf**2))*
                                                        (self.mx**2 - mf**2 + self.mm**2)**4*Pi)
        return sv

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

    def mediator_width(self, ferms=True, dm=True):
        width = 0.
        for f in self.f:
            mass_f = get_mass(f)
            width += self.lam_v ** 2. * ((self.mx - mass_f - self.mm) * (self.mx + mass_f - self.mm) *
                                        (self.mx - mass_f + self.mm) * (self.mx + mass_f + self.mm)) ** (1. / 2.) *\
                    (-2. * self.mx ** 4. + (mass_f**2. - self.mm**2.) ** 2. + self.mx ** 2. *
                     (mass_f**2. + self.mm**2.)) /\
                    (8. * np.pi * self.mx**2. * self.mm**3.)
        return width
