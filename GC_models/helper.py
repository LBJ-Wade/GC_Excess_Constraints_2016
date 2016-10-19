import numpy as np
import os
from scipy.interpolate import interp1d

try:
    MAIN_PATH = os.environ['GC_MODEL_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'

loadgeff = np.loadtxt(MAIN_PATH + '/Input_Data/g_eff.dat')

mn = 0.938
inv_GeV2_to_cm2 = 3.88 * 10 ** -28.  # (GeV * cm) ^ 2
m_to_cm = 100.
m2_to_barn = 10 ** 28.
femto = 10 ** -15.
speed_light = 2.998 * 10 ** 8.  # m / s
m_planck = 1.22 * 10 ** 19.  # GeV
omega_dm = [0.268, 0.268 - 0.010, 0.268 + 0.013]
hubble = 0.673

def color_number(fermion):
    quarks = ["u", "d", "c", "b", "t", "s"]
    leptons = ["e", "mu", "tau", "nu_e", "nu_mu", "nu_tau"]
    if fermion in quarks:
        return 3.
    elif fermion in leptons:
        return 1.
    else:
        print 'Not a fermion?'
        raise ValueError

def dm_dof(dm_spin, dm_type):
    if dm_spin == 'scalar':
        if dm_type:
            return 1.
        else:
            return 2.
    if dm_spin == 'fermion':
        if dm_type == 'dirac':
            return 4.
        else:
            return 2.
    if dm_spin == 'vector':
        if dm_type:
            return 3.
        else:
            return 6.

def effective_dof(temp):
    geff = interp1d(loadgeff[:, 0], loadgeff[:, 1], kind='linear', bounds_error=False, fill_value='extrapolate')
    return geff(temp)

def get_mass(particle):
    particle_table = ['e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau', 'u', 'd', 'c', 's', 't', 'b']
    mass_table = [5.11 * 10 ** -4., 0.105, 1.777, 0., 0., 0., 2.3 * 10 ** -3., 4.8 * 10 ** -3.,
                  1.275, 9.5 * 10 ** -2., 173., 4.18]
    return mass_table[particle_table.index(particle)]


def nuclide_properties(nuc):
    if nuc == 'Xe':
        return [[54., 54., 54.], [129., 131., 132.],
                [129. * mn, 131. * mn, 132. * mn]]
    if nuc == 'Si':
        return [[14.], [28.], [28. * mn]]
    if nuc == 'F':
        return [[9.], [19.], [19. * mn]]

def nuclide_spin(nuc):
    if nuc == 'Xe':
        return [[0.359, -0.227, 0.], [0.028, -0.009, 0.], [.5, 1.5, 0.]]
    if nuc == 'Si':
        return [[0.], [0.], [0.]]
    if nuc == 'F':
        return [[-0.109], [0.441], [0.5]]
    return


def interpola(val, x, y):
    try:
        f = np.zeros(len(val))
        for i, v in enumerate(val):
            if v <= x[0]:
                f[i] = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (v - x[0])
            elif v >= x[-1]:
                f[i] = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (v - x[-2])
            else:
                f[i] = interp1d(x, y, kind='cubic').__call__(v)
    except TypeError:
        if val <= x[0]:
            f = y[0] + (y[1] - y[0]) / (x[1] - x[0]) * (val - x[0])
        elif val >= x[-1]:
            f = y[-2] + (y[-1] - y[-2]) / (x[-1] - x[-2]) * (val - x[-2])
        else:
            try:
                f = interp1d(x, y, kind='cubic').__call__(val)
            except:
                f = interp1d(x, y, kind='linear').__call__(val)
    return f

