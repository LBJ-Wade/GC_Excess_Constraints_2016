"""
Created September 15th, 2016
Sam Witte
"""
import matplotlib as mpl
try:
    mpl.use('Agg')
except:
    pass
import numpy as np
from helper import *
import os
from models import *
from globals import *
from solvers import *
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import glob
from scipy.interpolate import interp1d
from scipy.optimize import fmin, minimize

#  mpl.use('pdf')
rc('font', **{'family': 'serif', 'serif': ['Times', 'Palatino']})
rc('text', usetex=True)

mpl.rcParams['xtick.major.size'] = 8
mpl.rcParams['ytick.major.size'] = 8
mpl.rcParams['xtick.labelsize'] = 16
mpl.rcParams['ytick.labelsize'] = 16

T, F = True, False

try:
    MAIN_PATH = os.environ['GC_MODEL_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'

# Interaction Information
dm_spin = ['fermion']  # scalar, fermion, vector
dm_real = [T]
dm_type = ['dirac']  # dirac, majorana
dm_mass = [25.]
mediator = ['v']  # s = scalar, v = vector, f = fermion
dm_bilinear = ['av']  # v=vector, av=axialvector, s=scalar, ps=pseudoscalar
channel = ['s']  # s or t
ferm_bilinear = ['av']

# What would you like to Calculate
direct = [T]  # Calculate direct detection bounds, fermionic coupling, target element
lhc = [F]  # Calculate LHC bounds
thermal_coups = T


candidates = len(dm_spin)
for i in range(candidates):
    if dm_mass[i] == 25.:
        ferms = ['b', 'c', 'u', 'd', 't', 's', 'e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau']
    elif dm_mass[i] == 35.:
        ferms = ['b']
    # Print Starting info:
    if dm_spin[i] == 'fermion':
        print 'Dark Matter Particle: ', dm_type[i], dm_spin[i]
    else:
        if dm_real[i]:
            real = 'Real'
        else:
            real = 'Complex'
        print 'Dark Matter Particle: ', real, dm_spin[i]
    print 'Dark Matter Mass: ', dm_mass[i]
    print channel[i] + '-channel'
    if mediator[i] == 's':
        print 'Mediator: Scalar'
    elif mediator[i] == 'v':
        print 'Mediator: Vector'
    else:
        print 'Mediator: Fermion'
    print 'DM bilinear: ', dm_bilinear[i], 'Fermion bilinear: ', ferm_bilinear[i]
    print 'Annihlation to: ', ferms

    fig_name = plot_namer(dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                          dm_bilinear[i], channel[i], ferm_bilinear[i])

    # Returns DM lambda couplings for s, ps, v, a in that order (1 or 0)
    dm_couplings = dm_couples(dm_spin[i], dm_bilinear[i])
    fm_couplings = fm_couples(ferm_bilinear[i])
    if mediator[i] == 's':
        dm_couplings = dm_couplings[:2]
        fm_couplings = fm_couplings[:2]
    elif mediator[i] == 'v':
        dm_couplings = dm_couplings[2:]
        fm_couplings = fm_couplings[2:]
        print 'DM [V: {:.0f}, AV: {:.0f}]'.format(dm_couplings[0], dm_couplings[1])

    if np.sum(np.concatenate((dm_couplings, fm_couplings))) == 0.:
        print dm_couplings
        print fm_couplings
        print 'All Couplings are 0!'
        raise ValueError

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')


    if channel[i] == 's':
        pl.ylim([10. ** -6., 10.])
        pl.xlim([1., 1000.])
    else:
        pl.ylim([10**-2., 10.])
        pl.xlim([100., 1000.])
    if mediator[i] == 's':
        pl.xlabel(r'$m_a$   [GeV]', fontsize=20)
    elif mediator[i] == 'v':
        pl.xlabel(r'$m_v$   [GeV]', fontsize=20)
    elif mediator[i] == 'f':
        pl.xlabel(r'$m_{{\psi}}$   [GeV]', fontsize=20)


    ylab = y_axis_label(dm_spin[i], dm_bilinear[i], channel[i], ferm_bilinear[i])
    pl.ylabel(ylab, fontsize=20)

    if direct[i]:
        print 'Calculating direct detection bounds...'
        dm_class = build_dm_class(channel[i], dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                                  ferms, 1., dm_couplings, fm_couplings)
        mass_med, bound = direct_detection_csec(dm_class, channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                mediator[i], dm_bilinear[i], ferm_bilinear[i], dm_mass[i])
        plt.plot(mass_med, bound, '--', lw=1, color='blue')
        arsize = len(mass_med)
        textpt = [mass_med[int(.8 * arsize)], bound[int(.8 * arsize)] + .1]

        plt.text(textpt[0], textpt[1], 'LUX', color='blue', fontsize=16,
                 rotation=np.arctan(2.) * 180. / np.pi * (3. / 4.),
                 ha='center', va='bottom')


    if lhc[i]:
        print 'Calculating LHC bounds...'
        files = glob.glob(MAIN_PATH + '/Input_Data/LHC*' + dm_bilinear[i] + '_' + ferm_bilinear[i] + '*.dat')
        print files
        for f in files:
            load = np.loadtxt(f)
            load = load[np.argsort(load[:, 0])]
            med = np.logspace(np.log10(np.min(load[:, 0])), np.log10(np.max(load[:, 0])), 100)
            plt_bnds = interpola(med, load[:, 0], load[:, 1])
            plt.plot(med, plt_bnds, '--', lw=1, color='red')

    if thermal_coups:
        mass_med_1 = np.logspace(0., np.log10(2 * dm_mass[i] * 0.75), 10)
        mass_med_3 = np.logspace(np.log10(2 * dm_mass[i] * 1.25), 3.1, 20)
        mass_med_2 = np.logspace(np.log10(2 * dm_mass[i] * 0.8), np.log10(2 * dm_mass[i] * 1.2), 20)
        mass_med = np.concatenate((mass_med_1, mass_med_2, mass_med_3))
        t_cups = np.zeros_like(mass_med)
        print 'Calculating thermal couplings...'
        for j, m_a in enumerate(mass_med):
            print 'M_a: ', m_a
            solve_c = fmin(t_coupling_omega, -1.5, args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                        dm_mass[i], mediator[i], ferms, m_a,
                                                        fm_couplings, dm_couplings), disp=False)
            t_cups[j] = np.power(10., solve_c)
        med_full = np.logspace(0., 3., 300)
        plt_therm = 10. ** interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups))
        plt.plot(med_full, plt_therm, lw=1, color='k')
    inter_label, mlab = plot_labeler(dm_spin[i], dm_real[i], dm_type[i], dm_bilinear[i], channel[i],
                                     ferm_bilinear[i], mediator[i])
    plt.text(750, 5. * 10 ** -6, inter_label, verticalalignment='bottom',
             horizontalalignment='right', fontsize=16)
    plt.text(750, 1.5 * 10 ** -6, mlab + ' = {:.0f}'.format(dm_mass[i]), verticalalignment='bottom',
             horizontalalignment='right', fontsize=16)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)



