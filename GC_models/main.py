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

mpl.rcParams['xtick.major.size'] = 10
mpl.rcParams['ytick.major.size'] = 10
mpl.rcParams['xtick.minor.size'] = 5
mpl.rcParams['ytick.minor.size'] = 5
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

# What would you like to Calculate

# DD Limits implimented:
#   1.) Schannel, Fermionic DM, PS-S interaction
#   2.) Schannel, Fermionic DM, AV-AV interaction
#   3.) Schannel, Scalar/Vector DM, S-PS interaction
#   4.) Tchannel, Fermionic DM, Scalar mediator
direct = [T]  # Calculate direct detection bounds, fermionic coupling, target element

# LHC Limits implimented:
lhc = [T]  # Calculate LHC bounds
thermal_coups = T


candidates = len(dm_spin)
for i in range(candidates):
    if dm_mass[i] == 25. and channel[i] == 's':
        #ferms = ['b', 'c', 'u', 'd', 't', 's']
        ferms = ['b', 'c', 'u', 'd', 't', 's', 'e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau']
    else:
        ferms = ['b']
        #ferms = ['b', 'c', 'u', 'd', 't', 's', 'e', 'mu', 'tau', 'nu_e', 'nu_mu', 'nu_tau']
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

    file_name = file_namer(dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                          dm_bilinear[i], channel[i], ferm_bilinear[i])

    # Returns DM lambda couplings for s, ps, v, a in that order (1 or 0)
    if channel[i] == 's':
        dm_couplings = dm_couples(dm_spin[i], dm_bilinear[i])
        fm_couplings = fm_couples(ferm_bilinear[i])
        if mediator[i] == 's':
            dm_couplings = dm_couplings[:2]
            fm_couplings = fm_couplings[:2]
            if dm_spin[i] == 'fermion':
                print 'DM [S: {:.0f}, PS: {:.0f}]'.format(dm_couplings[0], dm_couplings[1])
            else:
                print 'DM [S: {:.0f}]'.format(dm_couplings[0])
            print 'FM [S: {:.0f}, PS: {:.0f}]'.format(fm_couplings[0], fm_couplings[1])
        elif mediator[i] == 'v':
            dm_couplings = dm_couplings[2:]
            fm_couplings = fm_couplings[2:]
            print 'DM [V: {:.0f}, AV: {:.0f}]'.format(dm_couplings[0], dm_couplings[1])
            print 'FM [V: {:.0f}, AV: {:.0f}]'.format(fm_couplings[0], fm_couplings[1])
        if np.sum(np.concatenate((dm_couplings, fm_couplings))) == 0.:
            print dm_couplings
            print fm_couplings
            print 'All Couplings are 0!'
            raise ValueError
    else:
        dm_couplings = np.array([1.])
        fm_couplings = np.array([0.])

    fig = plt.figure(figsize=(8., 6.))
    ax = plt.gca()
    ax.set_xscale('log')
    ax.set_yscale('log')

    if channel[i] == 's':
        pl.ylim([10. ** -6., 10.])
        pl.xlim([1., 1000.])
    else:
        pl.ylim([10 ** -2., 10.])
        pl.xlim([100., 2000.])
    if mediator[i] == 's':
        pl.xlabel(r'$m_{\rm{a}}$   [GeV]', fontsize=20)
    elif mediator[i] == 'v':
        pl.xlabel(r'$m_{\rm{v}}$   [GeV]', fontsize=20)
    elif mediator[i] == 'f':
        pl.xlabel(r'$m_{{\psi}}$   [GeV]', fontsize=20)


    ylab = y_axis_label(dm_spin[i], dm_bilinear[i], channel[i], ferm_bilinear[i])
    pl.ylabel(ylab, fontsize=20)

    if direct[i]:
        print 'Calculating direct detection bounds...'
        dm_class = build_dm_class(channel[i], dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                                  ferms, 1., dm_couplings, fm_couplings, 0.)
        mass_med, bound = direct_detection_csec(channel[i], dm_spin[i], mediator[i],
                                                dm_bilinear[i], ferm_bilinear[i], dm_mass[i])
        plt.plot(mass_med, bound, '--', lw=2, color='blue')
        arsize = len(mass_med)
        textpt = [mass_med[int(.8 * arsize)], bound[int(.8 * arsize)]]
        plt.text(textpt[0], textpt[1], 'LUX', color='blue', fontsize=20,
                 rotation=np.arctan(.66) * 180. / np.pi,
                 ha='center', va='bottom')

    if lhc[i]:
        print 'Calculating LHC bounds...'
        if channel[i] == 's':
            files = glob.glob(MAIN_PATH + '/Input_Data/LHC*' + '_' + dm_bilinear[i] + '_' +
                              ferm_bilinear[i] + '_' + '*.dat')
            print files
            for f in files:
                tag = f[len(f) - f[::-1].find('_'):f.find('.dat')]
                load = np.loadtxt(f)
                load = load[np.argsort(load[:, 0])]
                med = np.logspace(np.log10(np.min(load[:, 0])), np.log10(np.max(load[:, 0])), 100)
                plt_bnds = interpola(med, load[:, 0], load[:, 1])
                plt.plot(med, plt_bnds, '--', lw=2, color='red')
                txtpt = interpola(800, load[:, 0], load[:, 1])
                plt.text(800., 0.5*txtpt, tag, color='red', fontsize=20,
                         rotation=0, ha='right', va='top')
        else:
            if mediator[i] == 's':
                dm_med_tag = 'scalar'
            elif mediator[i] == 'v':
                dm_med_tag = 'vector'
            else:
                dm_med_tag = 'fermion'
            files = glob.glob(MAIN_PATH + '/Input_Data/LHC_tchannel*' + dm_spin[i] + '*' + dm_med_tag + '*.dat')
            print files
            for f in files:
                limit = np.loadtxt(f)
                plt.axvline(x=limit, ymin=0, ymax=1, ls='--', lw=2, color='red')

    if thermal_coups:
        try:
            med_full, plt_therm = np.loadtxt(file_name)
            plt.plot(med_full, plt_therm, lw=2, color='k')
        except:
            if channel[i] == 's':
                mass_med_1 = np.logspace(0., np.log10(2 * dm_mass[i] * 0.75), 20)
                mass_med_3 = np.logspace(np.log10(2 * dm_mass[i] * 1.25), 3.1, 20)
                mass_med_2 = np.logspace(np.log10(2 * dm_mass[i] * 0.755), np.log10(2 * dm_mass[i] * 1.23), 20)
                mass_med = np.concatenate((mass_med_1, mass_med_2, mass_med_3))
                med_full = np.logspace(0., 3., 300)
            else:
                mass_med = np.logspace(2., np.log10(2. * 10 ** 3.), 15)
                med_full = np.logspace(2., np.log10(2. * 10 ** 3.), 100)
                #plt.axhline(y=(3. / 2.) ** 2., xmin=0, xmax=1, ls='-.', color='purple')
                #plt.text(150, (3. / 2.) ** 2., 'Non-perturbative', fontsize=14, ha='left', va='bottom', color='purple')
            t_cups = np.zeros_like(mass_med)
            t_cups1 = np.zeros_like(mass_med)
            t_cups10 = np.zeros_like(mass_med)
            t_cups100 = np.zeros_like(mass_med)
            width_cups = np.zeros_like(mass_med)
            max_cups = np.zeros_like(mass_med)
            print 'Calculating thermal couplings...'
            for j, m_a in enumerate(mass_med):
                solve_c = fmin(t_coupling_omega, 0., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                            dm_mass[i], mediator[i], ferms, m_a,
                                                            fm_couplings, dm_couplings, 0.), disp=False)

                if channel[i] == 's':
                    t_cups[j] = np.power(10., solve_c)

                    solve_c1 = fmin(t_coupling_omega, 0., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                               dm_mass[i], mediator[i], ferms, m_a,
                                                               fm_couplings, dm_couplings, 1.), disp=False)
                    t_cups1[j] = np.power(10., solve_c1)
                    solve_c10 = fmin(t_coupling_omega, 0., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                               dm_mass[i], mediator[i], ferms, m_a,
                                                               fm_couplings, dm_couplings, 10.), disp=False)
                    t_cups10[j] = np.power(10., solve_c10)
                    # solve_c100 = fmin(t_coupling_omega, 0., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                    #                                              dm_mass[i], mediator[i], ferms, m_a,
                    #                                              fm_couplings, dm_couplings, 100.), disp=False)
                    # t_cups100[j] = np.power(10., solve_c100)

                else:
                    try:
                        if mediator[i] != 'f':
                            raise ValueError
                        solve_wid = fmin(narrow_width, 0., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                                   dm_mass[i], mediator[i], ferms, m_a,
                                                                   fm_couplings, dm_couplings, 0.1), disp=False)
                        width_cups[j] = np.power(10., 2. * solve_wid)

                        solve_wid2 = fmin(narrow_width, 10.**-5., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                                 dm_mass[i], mediator[i], ferms, m_a,
                                                                 fm_couplings, dm_couplings, 0.5), disp=False)
                        max_cups[j] = np.power(10., 2. * solve_wid2)
                        print width_cups[j], max_cups[j]
                    except:
                        pass
                    t_cups[j] = np.power(10., 2. * solve_c)

                print 'Mass, Sqr Coupling:', m_a, t_cups[j]

            plt_therm = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups)))
            if channel[i] == 't':
                try:
                    plt_width_lim = np.power(10, interpola(np.log10(med_full), np.log10(mass_med),
                                                           np.log10(width_cups)))
                    plt_width_lim2 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med),
                                                            np.log10(max_cups)))
                    plt.plot(med_full, plt_width_lim, '-.', lw=2, color='purple')
                    plt.text(med_full[10], plt_width_lim[10], r'$\Gamma = 0.1$', fontsize=14, ha='left',
                             va='bottom', color='purple', rotation=0)
                    plt.plot(med_full, plt_width_lim2, '-.', lw=2, color='purple')
                    plt.text(med_full[10], plt_width_lim2[10], r'$\Gamma = 0.5$', fontsize=14, ha='left',
                             va='bottom', color='purple')
                except:
                    pass
            else:
                plt_therm_c1 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups1)))
                plt_therm_c10 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups10)))
                # plt_therm_c100 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups100)))
                for jj in range(len(med_full)):
                    if plt_therm[jj] < plt_therm_c1[jj]:
                        plt_therm_c1[jj] = plt_therm[jj]
                    if plt_therm[jj] < plt_therm_c10[jj]:
                        plt_therm_c10[jj] = plt_therm[jj]
                    # if plt_therm[jj] < plt_therm_c100[jj]:
                    #     plt_therm_c100[jj] = plt_therm[jj]

                plt.plot(med_full, plt_therm_c1, lw=1, color='k')
                plt.plot(med_full, plt_therm_c10, lw=1, color='k')
                # plt.plot(med_full, plt_therm_c100, lw=1, color='k')

            plt.plot(med_full, plt_therm, lw=2, color='k')
            np.savetxt(file_name, np.column_stack((med_full, plt_therm)),fmt='%.3e')
    inter_label, mlab = plot_labeler(dm_spin[i], dm_real[i], dm_type[i], dm_bilinear[i], channel[i],
                                     ferm_bilinear[i], mediator[i])
    plt.text(750, 5. * 10 ** -6, inter_label, verticalalignment='bottom',
             horizontalalignment='right', fontsize=16)
    plt.text(750, 1.5 * 10 ** -6, mlab + ' = {:.0f} GeV'.format(dm_mass[i]), verticalalignment='bottom',
             horizontalalignment='right', fontsize=16)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)

