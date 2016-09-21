import numpy as np
from helper import *
from scipy.integrate import quad


class fermionic_dm_spin0_med(object):
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

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += nc / (16. * np.pi * s * ((s - self.m_a ** 2.) ** 2. + (self.m_a * gamma) ** 2.)) *\
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *\
                (self.lam_f_s ** 2. * (s - 4. * mass_f ** 2.) + self.lam_f_p ** 2. * s) *\
                (self.lam_chi_s ** 2. * (s - 4. * self.mx ** 2.) + self.lam_chi_p ** 2. * s)
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *\
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.)
        return 1.

    def sigma_v_all(self, v):
        sigma = 0.
        for ferm in self.f:
            sigma += self.sigma_v(ferm, v)
        return sigma

    def sigma_v_thermal_approx(self, lam, v):
        sigma = 3. * lam ** 4. * self.mx ** 2. / (np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) *\
                (1. + 9. * v ** 2. / (8. * (1. - 4. * self.mx ** 2 / self.m_a ** 2.)))
        return sigma

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

    def sigma_v(self, channel, v):
        # Non-realtivsitic expansion to power v^2 -- NOTE: not thermally averaged!
        # This is for specific annihilation products, not general!
        nc = color_number(channel)
        mass_f = get_mass(channel)
        if self.mx > mass_f:
            kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
            sv = nc * self.lam_chi_p ** 2. * kin_mass * \
                (self.mx ** 2. * (self.lam_f_p ** 2. + self.lam_f_s ** 2.) - (mass_f * self.lam_f_s) ** 2.) /\
                (2. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) +\
                nc * v ** 2. / (16. * np.pi * self.mx ** 2. * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. * kin_mass) *\
                (self.lam_chi_p ** 2. *
                 ((self.mx * self.lam_f_p) ** 2. *
                  (self.m_a ** 2. * (mass_f ** 2. - 2. * self.mx ** 2.) + 12. *
                  (self.mx * mass_f) ** 2. - 8. * self.mx ** 4.) + self.lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) *
                  (self.m_a ** 2. * (mass_f ** 2. + 2. * self.mx ** 2.) - 20. * (self.mx * mass_f) ** 2. + 8. * self.mx ** 4.)) -
                 2. * self.lam_chi_s ** 2. * (self.m_a ** 2. - 4. * self.mx ** 2.) * (self.mx ** 2. - mass_f ** 2.) *
                 (self.mx ** 2. * (self.lam_f_p ** 2. + self.lam_f_s ** 2.) - mass_f ** 2. * self.lam_f_s ** 2.))
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
            sigma += tar_frac * red_mass ** 2. * self.lam_chi_s ** 2. / (np.pi * self.m_a ** 4.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
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
            sigma += tar_frac * 4. / 3. * (red_mass ** 2. * v ** 2. / (self.mx * m_nuc[i])) ** 2. *\
                     red_mass ** 2. * self.lam_chi_p ** 2. / (np.pi * self.m_a ** 4.) * (jn[i] + 1.) / jn[i] * \
                     (sn[i] * tn + sp[i] * tp) ** 2.
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
            sigma += tar_frac * red_mass ** 4. * v ** 2. * self.lam_chi_p ** 2. / \
                     (2 * self.mx ** 2. * np.pi * self.m_a ** 4.) * \
                     (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
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
            sigma += tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_chi_s ** 2. / \
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def x_freese_out(self):
        g = dm_dof('fermion')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class fermionic_dm_spin1_med(object):
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
            self.lam_chi_v = 0.
            self.lam_chi_a *= 0.5

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += nc / (12. * np.pi * s * ((s - self.m_v ** 2.) ** 2. + (self.m_v * gamma) ** 2.)) *\
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *\
                (self.lam_f_a ** 2. *
                 (self.lam_chi_a ** 2. *
                  (4. * self.mx ** 2. * (mass_f ** 2. * (7. - 6. * s / self.m_v ** 2. + 3. * s ** 2. / self.m_v ** 4.) - s)
                 + s * (s - 4. * mass_f ** 2.)) + self.lam_chi_v ** 2. * (s - 4. * mass_f ** 2.) * (2. * self.mx ** 2. + s))
                 + self.lam_f_v ** 2. * (2. * mass_f ** 2. + s) * (self.lam_chi_a ** 2. * (s - 4. * self.mx ** 2.) +
                                                                   self.lam_chi_v ** 2. * (2. * self.mx ** 2. + s)))
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

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

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

            sv = nc * kin_mass / (2. * np.pi * self.m_v ** 4. * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2.) *\
                 (self.lam_f_a ** 2. * (mass_f * self.lam_chi_a ** 2. * (self.m_v ** 2. - 4. * self.mx ** 2.) ** 2. +
                                        2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (self.mx ** 2. - mass_f ** 2.)) +
                  self.lam_f_v ** 2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (mass_f ** 2. + 2. * self.mx ** 2.)) -\
                nc * v ** 2. / (48. * np.pi * self.m_v ** 4. * self.mx ** 2. *
                                kin_mass * (4. * self.mx ** 2. - self.m_v ** 2.) ** 3.) *\
                (self.lam_f_a ** 2. * (self.lam_chi_a ** 2. * (self.m_v ** 2. - 4. * self.mx ** 2.) *
                                       (mass_f ** 4. * (-72. * self.m_v ** 2. * self.mx ** 2. + 17. *
                                                        self.m_v ** 4. + 144. * self.mx ** 4.) +
                                        mass_f ** 2. * (48. * self.m_v ** 2. + self.mx ** 4. - 22. *
                                                        self.m_v ** 4. * self.mx ** 2. - 96. * self.mx ** 6.) +
                                        8. * self.m_v ** 4. * self.mx ** 4.) -
                                        2. * self.lam_chi_v ** 2. * self.m_v ** 4. * (mass_f ** 2. - self.mx ** 2.) *
                                       (4. * self.mx ** 2. * (self.m_v ** 2. - 17. * mass_f ** 2.) + 5. *
                                        mass_f ** 2. * self.m_v ** 2. + 32. * self.mx ** 4.)) +
                 self.lam_f_v ** 2. * self.m_v ** 4. * (self.lam_chi_v ** 2. *
                                                        (8. * self.mx ** 4 * (self.m_v ** 2. - 4. * mass_f ** 2.) -
                                                         4. * (mass_f * self.mx) ** 2. * (17. * mass_f ** 2. + self.m_v ** 2.) +
                                                         5. * mass_f ** 2. * self.m_v ** 2. + 64. * self.mx ** 6.) -
                                                        4. * self.lam_f_a ** 2. * (mass_f ** 2. * self.mx ** 2. + mass_f ** 4. -
                                                                                   2. * self.mx ** 4.) * (self.m_v ** 2. - 4. * self.mx ** 2.)))
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


    def x_freese_out(self):
        g = dm_dof('fermion')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h



class scalar_dm_spin0_med(object):
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
            self.lam_p /= 2.

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += nc * self.lam_p ** 2. / (8. * np.pi * s * ((s - self.m_a ** 2) ** 2 + (self.m_a * gamma) ** 2.)) *\
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) *\
                     (self.lam_f_s **2. * (s - 4. * mass_f ** 2.) + s * self.lam_f_p ** 2.)
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *\
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.)
        return 1.

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

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
        kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
        sv = nc * self.lam_p ** 2. * kin_mass * (self.lam_f_p ** 2. + self.lam_f_s ** 2. *
                                                 (1. - mass_f ** 2. / self.mx ** 2.)) / \
             (4 * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2) ** 2) - \
             nc * self.lam_p ** 2. * v ** 2. / \
             (32. * np.pi * self.mx ** 4 * (4. * self.mx ** 2 - self.m_a ** 2.) ** 3. * kin_mass) * \
             (self.lam_f_p ** 2. * self.mx ** 2. * (self.m_a ** 2. * mass_f ** 2. - 20. * mass_f ** 2. *
                                                    self.mx ** 2. + 16. * self.mx ** 4.) +
              self.lam_f_s ** 2. * (self.mx ** 2. - mass_f ** 2.) *
              (3. * self.m_a ** 2. * mass_f ** 2. - 28. * mass_f ** 2. * self.mx ** 2. + 16. * self.mx ** 4.))
        return sv

    def scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_p ** 2. / (4. * np.pi * self.m_a ** 4. * self.mx ** 2.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
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
            sigma += tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_p ** 2. / \
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4. * self.mx ** 2.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def x_freese_out(self):
        g = dm_dof('scalar')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class scalar_dm_spin1_med(object):
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

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

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
        kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
        sv = nc * self.lam_p ** 2. * self.mx ** 2 * v ** 2 / (6. * np.pi) * kin_mass / \
        (self.m_v ** 2. - 4. * self.mx ** 2) ** 2 * (self.lam_f_a ** 2. * (1. - mass_f ** 2. / self.mx ** 2.) +
                                                     self.lam_f_v ** 2. * (1 + 0.5 * mass_f ** 2. / self.mx ** 2.))

        return sv

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

    def x_freese_out(self):
        g = dm_dof('scalar')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class vector_dm_spin0_med(object):
    """
    Lagrangian = [\lamba_x X^\mu X_\mu^\dag +
    \bar{f} (\lamba_{f,s} + \lambda_{f,p} i \gamma^5) f] A
    """
    def __init__(self, mx, dm_type, f, m_a, lam_x, lam_f_s, lam_f_p):
        self.mx = mx
        self.dm_type = dm_type
        self.m_a = m_a
        self.lam_x = lam_x
        self.lam_f_s = lam_f_s
        self.lam_f_p = lam_f_p
        self.f = f
        if self.dm_type:
            self.lam_x *= 0.5

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += nc * self.lam_x ** 2. / (72. * np.pi * ((s - self.m_a ** 2.) ** 2 + (self.m_a * gamma) ** 2)) * \
                np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2 / s)) * \
                     (s / self.mx ** 2 * (s / (4. * self.mx ** 2) - 1.) + 3.) * \
                     (self.lam_f_s ** 2. * (1. - 4. * mass_f ** 2. / s) + self.lam_f_p ** 2.)
        return sigma

    def mediator_width(self):
        width = 0.
        for ferm in self.f:
            sym = 2.
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            if mass_f < self.m_a:
                width += nc * self.m_a / (8. * np.pi * sym) * np.sqrt(1. - 4. * (mass_f / self.m_a) ** 2.) *\
                         (self.lam_f_s ** 2. * (1. - 4. * (mass_f / self.m_a) ** 2.) + self.lam_f_p ** 2.)
        return 1.

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

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
        kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
        sv = nc * self.lam_x ** 2. * kin_mass * (self.lam_f_p ** 2. + self.lam_f_s ** 2. * (1. - (mass_f / self.mx) ** 2.)) / \
             (12. * np.pi * (self.m_a ** 2. - 4. * self.mx ** 2.) ** 2.) + \
            nc * self.lam_x ** 2. * v ** 2 / (288. * np.pi * self.mx ** 4 * (4. * self.mx ** 2. - self.m_a ** 2.) ** 3. *
                                              kin_mass) * \
            (self.lam_f_p ** 2 * self.mx ** 2 * (4. * self.mx ** 2. * (7. * mass_f ** 2 - 2 * self.m_a ** 2) + 5. *
                                                 self.m_a ** 2 * mass_f ** 2. - 16. * self.mx ** 4.) +
             self.lam_f_s ** 2. * (mass_f ** 2. - self.mx ** 2.) * (self.m_a ** 2. * (mass_f ** 2 + 8. * self.mx ** 2.) -
                                                                    52. * mass_f ** 2. * self.mx ** 2. +
                                                                    16. * self.mx ** 4.))
        return sv

    def scalar_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        fp = self.lam_f_s * m_nuc * (7. / 9. * (0.020 + 0.026 + 0.043) + 2. / 9.)
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_x ** 2. / (np.pi * self.m_a ** 4. * self.mx ** 2.) * \
                    (z_nuc[i] * fp + (a_nuc[i] - z_nuc[i]) * fn) ** 2.
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
            sigma += tar_frac * 2. * red_mass ** 4. * v ** 2. * self.lam_x ** 2. / \
                    (4. * m_nuc[i] ** 2. * np.pi * self.m_a ** 4. * self.mx ** 2.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def x_freese_out(self):
        g = dm_dof('vector')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h


class vector_dm_spin1_med(object):
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
            self.lam_x *= 0.5

    def sigma(self, s):
        sigma = 0.
        gamma = self.mediator_width()
        for i, ferm in enumerate(self.f):
            nc = color_number(ferm)
            mass_f = get_mass(ferm)
            sigma += self.lam_x ** 2. * (s - 4 * self.mx ** 2) / (72. * np.pi * (self.mx * self.m_v) ** 4. *
                                                                  ((self.m_v * gamma) ** 2 +
                                                                   (self.m_v ** 2 - s) ** 2)) * \
            np.sqrt((1. - 4. * mass_f ** 2. / s) / (1. - 4. * self.mx ** 2. / s)) * \
            (2. * self.lam_f_v ** 2. * self.mx ** 2 * self.m_v ** 4 * (2. * mass_f ** 2 + s) +
             self.lam_f_a ** 2 * (2 * self.mx ** 2 * (s * self.m_v ** 4 - 2 * mass_f ** 2 *
                                                      (5 * self.m_v ** 4 - 6. * s * self.m_v ** 2 +
                                                       3 * s ** 2)) +
                                  3 * mass_f ** 2 * s * (self.m_v ** 2 - s) ** 2.))
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

    def sig_therm_exact(self, temp):
        x = self.mx / temp

        def integrd(v, x):
            hv = v / (2. * speed_light)
            return hv ** 2 * np.exp (- x * hv ** 2. / (1. - hv ** 2.)) / \
                (1. - hv ** 2.) ** (5. / 2.) * self.sigma_v_all(v / speed_light)
        them_avg = quad(integrd, 0., 2. * speed_light, args=x)
        return 2. / np.sqrt(np.pi) * x ** (3. / 2.) * them_avg[0]

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
        kin_mass = np.sqrt(1. - (mass_f / self.mx) ** 2.)
        sv = nc * self.lam_x ** 2. * v ** 2 * kin_mass / (27. * np.pi * (self.m_v ** 2. - 4. * self.mx ** 2)) ** 2 * \
             (mass_f ** 2. * (self.lam_f_v ** 2. - 2. * self.lam_f_a ** 2.) + 2. * self.mx ** 2. *
              (self.lam_f_a ** 2. + self.lam_f_v ** 2.))
        return sv

    def vector_cross_section(self, nuclide, v=1.):
        # Assumes one can integrate out mediator!
        [z_nuc, a_nuc, m_nuc] = nuclide_properties(nuclide)
        fn = 3. * self.lam_f_v
        fp = 3. * self.lam_f_v
        sigma = 0
        tar_frac = 1. / len(m_nuc)
        for i in range(len(m_nuc)):
            red_mass = self.mx * m_nuc[i] / (self.mx + m_nuc[i])
            sigma += tar_frac * red_mass ** 2. * self.lam_x ** 2. / (4. * np.pi * self.m_v ** 4.) * \
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
            sigma += tar_frac * red_mass ** 4. * v ** 2. * self.lam_x ** 2. / \
                    (2. * np.pi * self.m_v ** 4. * red_mass_n ** 2.) * (jn[i] + 1.) / jn[i] * \
                    (sn[i] * tn + sp[i] * tp) ** 2.
        return sigma

    def x_freese_out(self):
        g = dm_dof('vector')
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
        xfo = self.x_freese_out()
        gstar = effective_dof(self.mx / xfo)
        jterm = quad(lambda x: self.sig_therm_exact(x) * 100. / (self.mx * speed_light), 0., self.mx / xfo)
        o_h = 1.07 * 10 ** 9. / (m_planck * jterm[0] * np.sqrt(gstar))
        return o_h

