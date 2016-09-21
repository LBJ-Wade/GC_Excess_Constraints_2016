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
import pylab as pl
import matplotlib.pyplot as plt
from matplotlib import rc
import glob

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
dm_mass = [35.]
mediator = ['v']  # s = scalar, v = vector, f = fermion
dm_bilinear = ['av']  # v=vector, av=axialvector, s=scalar, ps=pseudoscalar
channel = ['s']  # s or t
ferm_bilinear = ['av']

# What fermions does the mediator couple to, and with what relative strength
ferms = ['b', 'c', 'u', 'd', 't', 's']
# What would you like to Caclculate
direct = [F, ['LUX']]  # Calculate direct detection bounds, fermionic coupling, target element
lhc = [T]  # Calculate LHC bounds
thermal_coups = T

inter_label = [r'$\bar{{\chi}} \gamma^\mu \gamma^5 \chi, \bar{{f}} \gamma^\mu \gamma^5 f$']


mass_med = np.logspace(0., 3., 300)

candidates = len(dm_spin)
for i in range(candidates):
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
    print 'Annihlation to: ', ferms[i]

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

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    pl.xlim([1., 1000.])
    pl.ylim([10. ** -6., 10.])
    if mediator[i] == 's':
        pl.xlabel(r'$m_a$   [GeV]', fontsize=20)
    elif mediator[i] == 'v':
        pl.xlabel(r'$m_v$   [GeV]', fontsize=20)
    elif mediator[i] == 'f':
        pl.xlabel(r'$m_f$   [GeV]', fontsize=20)

    #TODO write code that returns proper ylabel dependent on channel, dm type, mediator, etc
    pl.ylabel(r'$\lambda_\chi \lambda_f$', fontsize=20)

    t_cups = np.zeros_like(mass_med)

    if direct[0]:
        print 'Calculating direct detection bounds...'
        for j, m_a in enumerate(mass_med):
            dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                                      ferms, m_a, dm_couplings, fm_couplings)
            if ferm_bilinear[i] == 's' and dm_bilinear[i] == 's':
                pass

    if lhc[0]:
        print 'Calculating LHC bounds...'
        files = glob.glob(MAIN_PATH + '/Input_Data/*' + dm_bilinear[i] + '_' + ferm_bilinear[i] + '*.dat')
        print files
        for f in files:
            load = np.loadtxt(f)
            load = load[np.argsort(load[:, 0])]
            med = np.logspace(np.log10(np.min(load[:, 0])), np.log10(np.max(load[:, 0])), 100)
            plt_bnds = interpola(med, load[:, 0], load[:, 1])
            plt.plot(med, plt_bnds, '--', lw=1, color='red')

    if thermal_coups:
        print 'Calculating thermal couplings...'
        for j, m_a in enumerate(mass_med):
            dm_class = build_dm_class(channel[i], dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                                      ferms, m_a, dm_couplings, fm_couplings)
            t_cups[j] = np.sqrt(dm_class.omega_h() / (omega_dm[0] * hubble ** 2.))

    plt.plot(mass_med, t_cups, lw=1, color='k')

    plt.text(750, 4. * 10 ** -6, inter_label[i], verticalalignment='bottom',
             horizontalalignment='right', fontsize=16)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)



