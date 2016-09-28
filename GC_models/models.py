import numpy as np
from helper import *
from scipy.integrate import quad


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
            sigma += (nc / (16. * np.pi * s * ((s - self.m_a ** 2.) ** 2. + (self.m_a * gamma) ** 2.)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                (self.lam_f_s ** 2. * (s - 4. * mass_f ** 2.) + self.lam_f_p ** 2. * s) *
                (self.lam_chi_s ** 2. * (s - 4. * self.mx ** 2.) + self.lam_chi_p ** 2. * s))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.))
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
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_chi_p ** 2. * kin_mass *
                (self.mx ** 2. * (self.lam_f_p ** 2. + self.lam_f_s ** 2.) - (mass_f * self.lam_f_s) ** 2.) /
                (2. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) +
                nc * v ** 2. / (16. * np.pi * self.mx ** 2. * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. * kin_mass) *
                (self.lam_chi_p ** 2. *
                 ((self.mx * self.lam_f_p) ** 2. *
                  (self.m_a ** 2. * (mass_f ** 2. - 2. * self.mx ** 2.) + 12. *
                  (self.mx * mass_f) ** 2. - 8. * self.mx ** 4.) + self.lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) *
                  (self.m_a ** 2. * (mass_f ** 2. + 2. * self.mx ** 2.) - 20. * (self.mx * mass_f) ** 2. + 8. * self.mx ** 4.)) -
                 2. * self.lam_chi_s ** 2. * (self.m_a ** 2. - 4. * self.mx ** 2.) * (self.mx ** 2. - mass_f ** 2.) *
                 (self.mx ** 2. * (self.lam_f_p ** 2. + self.lam_f_s ** 2.) - mass_f ** 2. * self.lam_f_s ** 2.)))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.

    def scalar_scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * red_mass ** 2. * self.lam_chi_s ** 2. / (np.pi * self.m_a ** 4.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.)
        return sigma

    def pscalar_pscalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_p * m_nuc * (-0.42 + 0.85 - 0.08)
        tp = self.lam_f_p * m_nuc * (0.43 - 0.84 - 0.50)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * 4. / 3. * (red_mass ** 2. * v ** 2. / (self.mx * m_nuc[i])) ** 2. *\
                     red_mass ** 2. * self.lam_chi_p ** 2. / (np.pi * self.m_a ** 4.) * (jn[i] + 1.) / jn[i] * \
                     (sn[i] * tn + sp[i] * tp) ** 2.)
        return sigma

    def pscalar_scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2./ 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2./ 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * red_mass ** 4. * v ** 2. * self.lam_chi_p ** 2. /
                     (2 * self.mx ** 2. * np.pi * self.m_a ** 4.) *
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.)
        return sigma

    def scalar_pscalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_p * m_nuc * (-0.42 + 0.85 - 0.08)
        tp = self.lam_f_p * m_nuc * (0.43 - 0.84 - 0.50)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_chi_s ** 2. /
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4.) * (jn[i] + 1.) / jn[i] *
                    (sn[i] * tn + sp[i] * tp) ** 2.)
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp
        them_avg = quad(lambda v: (v/2.) ** 2 * np.exp(- x * (v/2.) ** 2. / (1. - (v/2.) ** 2.)) /
                   (1. - (v/2.) ** 2.) ** (5. / 2.) * self.sigma_v_all(v), 0., 2.)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

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
    def __init__(self, mx, dm_type, f, m_v, lam_chi_v, lam_chi_a, lam_f_v, lam_f_a):
        self.mx = mx
        self.dm_type = dm_type
        self.m_v = m_v
        self.lam_chi_v = lam_chi_v
        self.lam_chi_a = lam_chi_a
        self.lam_f_v = lam_f_v
        self.lam_f_a = lam_f_a
        self.f = f
        if self.dm_type == 'majorana':
            self.lam_chi_v *= 2.
            self.lam_chi_a *= 2.


    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
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

        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * kin_mass / (2. * np.pi * self.m_v ** 4. * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2.) *
                 (self.lam_f_a ** 2. * (mass_f ** 2. * self.lam_chi_a ** 2. * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2. +
                                        2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (self.mx ** 2. - mass_f ** 2.)) +
                  self.lam_f_v ** 2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (mass_f ** 2. + 2. * self.mx ** 2.)) -
                nc * v ** 2. / (48. * np.pi * self.m_v ** 4. * self.mx ** 2. *
                                kin_mass * (4. * self.mx ** 2. - self.m_v ** 2.) ** 3.) *
                (self.lam_f_a ** 2. * (self.lam_chi_a ** 2. * (self.m_v ** 2. - 4. * self.mx ** 2.) *
                                       (mass_f ** 4. * (-72. * self.m_v ** 2. * self.mx ** 2. + 17. *
                                                        self.m_v ** 4. + 144. * self.mx ** 4.) +
                                        mass_f ** 2. * (48. * self.m_v ** 2. * self.mx ** 4. - 22. *
                                                        self.m_v ** 4. * self.mx ** 2. - 96. * self.mx ** 6.) +
                                        8. * self.m_v ** 4. * self.mx ** 4.) -
                                        2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (mass_f ** 2. - self.mx ** 2.) *
                                       (4. * self.mx ** 2. * (self.m_v ** 2. - 17. * mass_f ** 2.) + 5. *
                                        mass_f ** 2. * self.m_v ** 2. + 32. * self.mx ** 4.)) +
                 self.lam_f_v ** 2. * self.m_v ** 4. * (self.lam_chi_v ** 2. *
                                                        (8. * self.mx ** 4 * (self.m_v ** 2. - 4. * mass_f ** 2.) -
                                                         4. * (mass_f * self.mx) ** 2. * (17. * mass_f ** 2. + self.m_v ** 2.) +
                                                         5. * mass_f ** 4. * self.m_v ** 2. + 64. * self.mx ** 6.) -
                                                        4. * self.lam_chi_a ** 2. *
                                                        (mass_f ** 2. * self.mx ** 2. + mass_f ** 4. -
                                                         2. * self.mx ** 4.) * (self.m_v ** 2. - 4. * self.mx ** 2.))))
            if sv < 0:
                return 0.
            else:
                return sv
        else:
            return 0.

    def vector_vector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_f_v
        fp = 3. * self.lam_f_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_chi_v ** 2. / (np.pi * self.m_v ** 4.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma

    def avector_avector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_a * (0.84 - 0.43 - 0.09)
        tp = self.lam_f_a * (0.84 - 0.43 - 0.09)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * 4. * red_mass ** 2. * self.lam_chi_a ** 2. / (np.pi * self.m_v ** 4.) *\
                     (jn[i] + 1.) / jn[i] * (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def avector_vector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_f_v
        fp = 3. * self.lam_f_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            red_mass_n = self.mx * mn / (self.mx + mn)
            sigma += tar_frac * 2. * red_mass ** 6. * v ** 2. * self.lam_chi_a ** 2. / \
                     (self.mx ** 2. * np.pi * self.m_v ** 4. * red_mass_n ** 2.) * \
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma

    def vector_avector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_a * (0.84 - 0.43 - 0.09)
        tp = self.lam_f_a * (0.84 - 0.43 - 0.09)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            red_mass_n = self.mx * mn / (self.mx + mn)
            sigma += tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_chi_v ** 2. / \
                    (np.pi * self.m_v ** 4. * red_mass_n ** 2.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 6. * lam ** 4. * self.mx ** 2. / (np.pi * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2.) *\
                (1. + 3. * v ** 2. * (1. + 2. * self.mx ** 2. / self.m_v ** 2.) /
                 (4. * (1. - 4. * self.mx ** 2 / self.m_v ** 2.)))
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)
        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', self.dm_type)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h



class scalar_dm_spin0_med_schannel(object):
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
            sigma += (nc * self.lam_p ** 2. / (8. * np.pi * s * ((s - self.m_a ** 2) ** 2 + (self.m_a * gamma) ** 2.)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *
                     (self.lam_f_s **2. * (s - 4. * mass_f ** 2.) + s * self.lam_f_p ** 2.))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.))
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
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_p ** 2. * kin_mass * (self.lam_f_p ** 2. + self.lam_f_s ** 2. *
                                                     (1. - mass_f ** 2. / self.mx ** 2.)) /
                 (4 * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2) ** 2) -
                 nc * self.lam_p ** 2. * v ** 2. /
                 (32. * np.pi * self.mx ** 4 * (4. * self.mx ** 2 - self.m_a ** 2.) ** 3. * kin_mass) *
                 (self.lam_f_p ** 2. * self.mx ** 2. * (self.m_a ** 2. * mass_f ** 2. - 20. * mass_f ** 2. *
                                                        self.mx ** 2. + 16. * self.mx ** 4.) +
                  self.lam_f_s ** 2. * (self.mx ** 2. - mass_f ** 2.) *
                  (3. * self.m_a ** 2. * mass_f ** 2. - 28. * mass_f ** 2. * self.mx ** 2. + 16. * self.mx ** 4.)))
            if sv < 0:
                return 0.
            else:
                return sv
        else:
            return 0.


    def scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * red_mass ** 2. * self.lam_p ** 2. / (4. * np.pi * self.m_a ** 4. * self.mx ** 2.) *
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.)
        return sigma

    def pscalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_p * m_nuc * (-0.42 + 0.85 - 0.08)
        tp = self.lam_f_p * m_nuc * (0.43 - 0.84 - 0.50)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_p ** 2. /
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4. * self.mx ** 2.) * (jn[i] + 1.) / jn[i] *
                    (sn[i] * tn + sp[i] * tp) ** 2.)
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)

        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

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
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = nc * self.lam_p ** 2. * self.mx ** 2 * v ** 2 / (6. * np.pi) * kin_mass / \
            (self.m_v ** 2. - 4. * self.mx ** 2) ** 2 * (self.lam_f_a ** 2. * (1. - mass_f ** 2. / self.mx ** 2.) +
                                                         self.lam_f_v ** 2. * (1 + 0.5 * mass_f ** 2. / self.mx ** 2.))
            if sv < 0:
                return 0.
            else:
                return sv
        else:
            return 0.


    def vector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_f_v
        fp = 3. * self.lam_f_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_p ** 2. / (4. * np.pi * self.m_v ** 4.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma

    def avector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_a * (0.84 - 0.43 - 0.09)
        tp = self.lam_f_a * (0.84 - 0.43 - 0.09)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            red_mass_n = self.mx * mn / (self.mx + mn)
            sigma += tar_frac * red_mass ** 4. * v ** 2. * self.lam_p ** 2. / \
                    (2. * np.pi * self.m_v ** 4. * red_mass_n ** 2.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)

        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('scalar', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

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
            sigma += (nc * self.lam_x ** 2. / (72. * np.pi * ((s - self.m_a ** 2.) ** 2 + (self.m_a * gamma) ** 2)) *
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2 / s)) *
                     (s / self.mx ** 2 * (s / (4. * self.mx ** 2) - 1.) + 3.) *
                     (self.lam_f_s ** 2. * (1. - 4. * mass_f ** 2. / s) + self.lam_f_p ** 2.))
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += (nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.))
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
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_x ** 2. * kin_mass * (self.lam_f_p ** 2. + self.lam_f_s ** 2. * (1. - (mass_f / self.mx) ** 2.)) /
                 (12. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) +
                nc * self.lam_x ** 2. * v ** 2 / (288. * np.pi * self.mx ** 4 * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. *
                                                  kin_mass) *
                (self.lam_f_p ** 2 * self.mx ** 2 * (4. * self.mx ** 2. * (7. * mass_f ** 2 - 2 * self.m_a ** 2) + 5. *
                                                     self.m_a ** 2 * mass_f ** 2. - 16. * self.mx ** 4.) +
                 self.lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) * (self.m_a ** 2. * (mass_f ** 2 + 8. * self.mx ** 2.) -
                                                                        52. * mass_f ** 2. * self.mx ** 2. +
                                                                        16. * self.mx ** 4.)))
            if sv < 0:
                return 0.
            else:
                return sv
        else:
            return 0.

    def scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * red_mass ** 2. * self.lam_x ** 2. / (np.pi * self.m_a ** 4. * self.mx ** 2.) *
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.)
        return sigma

    def pscalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_p * m_nuc * (-0.42 + 0.85 - 0.08)
        tp = self.lam_f_p * m_nuc * (0.43 - 0.84 - 0.50)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_x ** 2. /
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4. * self.mx ** 2.) * (jn[i] + 1.) / jn[i] *
                    (sn[i] * tn + sp[i] * tp) ** 2.)
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)

        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

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
            if mass_f < self.m_v:
                width += (nc * self.m_v / (12. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_v) ** 2.) *
                         (self.lam_f_a ** 2. * (1. - 4. * (mass_f / self.m_v) ** 2.) +
                          self.lam_f_v ** 2. * (1. + 2. * (mass_f / self.m_v) ** 2.)))
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
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = (nc * self.lam_x ** 2. * v ** 2 * kin_mass / (27. * np.pi * (self.m_v ** 2. - 4. * self.mx ** 2)) ** 2 *
                 (mass_f ** 2. * (self.lam_f_v ** 2. - 2. * self.lam_f_a ** 2.) + 2. * self.mx ** 2. *
                  (self.lam_f_a ** 2. + self.lam_f_v ** 2.)))
            if sv < 0:
                return 0.
            else:
                return sv
        else:
            return 0.

    def vector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_f_v
        fp = 3. * self.lam_f_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += (tar_frac * red_mass ** 2. * self.lam_x ** 2. / (4. * np.pi * self.m_v ** 4.) *
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.)
        return sigma

    def avector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        [sn, sp, jn] = nuclide_spin(nuclide)
        tn = self.lam_f_a * (0.84 - 0.43 - 0.09)
        tp = self.lam_f_a * (0.84 - 0.43 - 0.09)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            red_mass_n = self.mx * mn / (self.mx + mn)
            sigma += (tar_frac * red_mass ** 4. * v ** 2. * self.lam_x ** 2. /
                    (2. * np.pi * self.m_v ** 4. * red_mass_n ** 2.) * (jn[i] + 1.) / jn[i] *
                    (sn[i] * tn + sp[i] * tp) ** 2.)
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)

        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('vector', self.dm_real)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

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
        self.lam_s = lam_s
        self.lam_p = lam_p
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
        sigma = 3. * lam ** 4. * self.mx ** 2. / (2. * np.pi * (self.m_a ** 2. + self.mx ** 2.) ** 2.) *\
                (1. + 3. * v ** 2. * (1. - 3. * self.mx ** 2. / self.m_a ** 2. - self.mx ** 4. / self.m_a ** 4.) /
                 (4. * (1. + self.mx ** 2 / self.m_a ** 2.)))
        return sigma

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general -- for that call sim_v_all
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = nc * kin_mass * (self.lam_p ** 2. * (self.mx - mass_f) + self.lam_s ** 2. * (mass_f + self.mx)) ** 2. / \
                 (8 * np.pi * (self.m_a ** 2. - mass_f ** 2. + self.mx ** 2.) ** 2.) - \
                nc * v ** 2. / (192. * np.pi * self.mx ** 2. * kin_mass * (self.m_a ** 2. - mass_f ** 2. + self.mx ** 2.) ** 4) * \
                (self.m_a ** 4 * (6. * mass_f ** 3 * self.mx * (self.lam_p ** 4. - self.lam_s ** 4) +
                                 mass_f ** 2 * self.mx ** 2 * (13. * self.lam_p ** 4 + 2 * self.lam_p ** 2 *
                                                               self.lam_s ** 2 + 13 * self.lam_s ** 4) +
                                 mass_f ** 4 * (-11. * self.lam_p ** 4. + 14. * self.lam_p ** 2 *
                                                self.lam_s ** 2 - 11. * self.lam_s ** 4) -
                                 8. * self.mx ** 4. * (self.lam_s ** 2 + self.lam_p ** 2.) ** 2) +
                 2. * self.m_a ** 2. * (mass_f ** 2 - self.mx ** 2) *
                 (self.lam_p ** 4. * (mass_f - self.mx) ** 2 * (8. * mass_f * self.mx + 11. * mass_f ** 2. - 12 * self.mx ** 2.) -
                  2. * self.lam_p ** 2. * self.lam_s ** 2 * (-19. * mass_f ** 2. * self.mx **2 +
                                                             7. * mass_f ** 4. + 12 * self.mx ** 4) +
                  self.lam_s ** 4. * (mass_f + self.mx) ** 2. * (-8. * mass_f * self.mx + 11. * mass_f ** 2. - 12. * self.mx ** 2)) -
                 self.lam_p ** 4. * (mass_f - self.mx) ** 4. * (mass_f + self.mx) ** 2. * (11. * mass_f ** 2. - 8. * self.mx ** 2) +
                 2. * self.lam_p ** 2. * self.lam_s ** 2. * (7. * mass_f ** 2. - 8. * self.mx ** 2.) *
                 (mass_f ** 2. - self.mx ** 2) ** 3. - self.lam_s ** 4 * (mass_f - self.mx) ** 2. *
                 (mass_f + self.mx) ** 4. * (11. * mass_f ** 2. - 8. * self.mx ** 2.))
            if sv < 0.:
                return 0.
            else:
                return sv
        else:
            return 0.


    def si_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_s
        fp = 3. * self.lam_s
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_s ** 2. / (4. * np.pi * self.m_a ** 4.) * \
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma


    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)
        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', 'dirac')
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class dirac_fermionic_dm_spin1_med_tchannel(object):
    """
    Lagrangian = [\bar{\chi} \gamma^\mu (g_\chi,s + g_\chi,p \gamma^5)f V_\mu +
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
        sigma = 3. * lam ** 4. * self.mx ** 2. / (2. * np.pi * (self.m_v ** 2 + self.mx **2) ** 2.) * \
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


    def si_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_v
        fp = 3. * self.lam_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_v ** 2. / (np.pi * self.m_v ** 4.) * \
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
        return sigma


    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)
        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', 'dirac')
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class complex_vector_dm_spin_half_med_tchannel(object):
    """
    Lagrangian = \bar{\psi} \gamma^\mu (g_v + g_a \gamma^5) f X_\mu^\dagger +
    \bar{f} \gamma^mu (g_v + g_a \gamma^5) \psi X_\mu

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
        sigma = 8. * lam ** 4. * self.mx ** 2. / (3. * np.pi * (self.mm ** 2 + self.mx **2) ** 2.) * \
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
        lv = self.lam_v
        la = self.lam_a
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

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)
        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', False)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class real_vector_dm_spin_half_med_tchannel(object):
    """
    Lagrangian = \bar{\psi} \gamma^\mu (g_v + g_a \gamma^5) f X_\mu +
    \bar{f} \gamma^mu (g_v + g_a \gamma^5) \psi X_\mu

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
        sigma = 32. * lam ** 4. * self.mx ** 2. / (3. * np.pi * (self.mm ** 2 + self.mx **2) ** 2.) * \
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
                 (-2*la**2*lv**2 * (5*mf**2 + 3*mx**2 - 12*mm**2) + la**4 * (2*mf*mm + 3*mf**2 +4*mx**2 + 3*mm**2) +
                  lv**4 * (-2*mf*mm + 3*mf**2+4*mx**2+3*mm**2)) + \
                nc * v**2 *kin_mass / (216 * np.pi * mx**2 * (-mf**2+mx**2*mm**2)**4) * \
                (3*mm**6*(la**2-lv**2)**2 *(mf**2+8*mx**2) + 6*mf*mm**5*(la**4-lv**4)*(mf**2-4*mx**2) +
                 mm**4 * (-2*la**2*lv**2 * (-46*mf**2*mx**2 + 39*mf**4-160*mx**4) +
                          5*la**4*(14*mf**2*mx**2 + mf**4) + 5*mf**2*lv**4*(mf**2 + 14*mx**2)) +
                 12*mf*mm**3 * (la**4-lv**4) * (-3*mf**2*mx**2 + mf**4 + 2*mx**4) - mm**2 *(mf**2-mx**2) *
                 (mf**2 * mx ** 2 * (474*la**2*lv**2 - 29*la**4 - 29*lv**4) + mf**4 * (-174*la**2*lv**2 + 19*la**4 + 19*lv**4)
                  -80*mx**4*(6*la**2*lv**2+la**4+lv**4)) - 2*mf*mm*(la**4-lv**4)*(3*mf**2-32*mx**2)*(mf**2-mx**2)**2 +
                 (mf**2-mx**2)**2*(8*mf**2*mx**2*(44*la**2*lv**2+la**4+lv**4) +
                                   mf**4*(-90*la**2*lv**2+11*la**4+11*lv**4) -
                                   56*mx**4*(6*la**2*lv**2+la**4+lv**4)))
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

        def integrd(v, x):
            hv = v / 2.
            return hv ** 2 * np.exp(- x * hv ** 2. / (1. - hv ** 2.)) / \
                   (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v)
        them_avg = quad(integrd, 0., 2., args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def x_freeze_out(self):
        g = dm_dof('fermion', True)
        tnew = 1.
        told = 0.
        while np.abs(told - tnew) > 0.0001:
            sv = self.sig_therm_exact(tnew)
            gstar = effective_dof(tnew)
            xf = self.mx / tnew
            told = tnew
            tnew = self.mx / np.log((0.038 * g * m_planck * self.mx * sv) / np.sqrt(gstar * xf))
            #print 'T_{i}: ', told, 'T_{i+1}:', tnew
        return self.mx / tnew

    def omega_h(self):
        xfo = self.x_freeze_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) / self.mx, 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h