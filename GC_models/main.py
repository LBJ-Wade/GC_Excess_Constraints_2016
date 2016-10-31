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
dm_spin = ['vector']  # scalar, fermion, vector
dm_real = [T]
dm_type = ['dirac']  # dirac, majorana
dm_mass = [35.]   # We're using 35 GeV to annihilation to all fermions, 50 GeV if just to bb
mediator = ['s']  # s = scalar, v = vector, f = fermion
dm_bilinear = ['s']  # v=vector, av=axialvector, s=scalar, ps=pseudoscalar
ferm_bilinear = ['ps']
channel = ['s']  # s or t

# What would you like to Calculate
direct = [F]  # Calculate direct detection bounds
lhc = [F]  # Calculate LHC bounds
thermal_coups = T  # Calculate thermal couplings
csec_plots = T  # Make cross section plots


candidates = len(dm_spin)
for i in range(candidates):

    if dm_mass[i] == 35. and channel[i] == 's':
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
        pl.xlim([1., 1000.])
        if dm_spin[i] == 'scalar' or dm_spin[i] == 'vector':
            pl.ylim([10. ** -2., 1000.])
        else:
            pl.ylim([10. ** -6., 10.])

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
    #  if mediator[i] == 's' and channel[i] == 's':
    #      ylab += ' x (1 GeV / $y_f$)'
    pl.ylabel(ylab, fontsize=20)

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ DIRECT DETECTION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if direct[i]:
        print 'Calculating direct detection bounds...'
        print dm_couplings, fm_couplings
        dm_class = build_dm_class(channel[i], dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                                  ferms, 1., dm_couplings, fm_couplings, 0.)
        mass_med, bound = direct_detection_csec(channel[i], dm_spin[i], mediator[i],
                                                dm_bilinear[i], ferm_bilinear[i], dm_mass[i])
        plt.plot(mass_med, bound, '--', lw=2, color='blue', dashes=(20, 10))
        arsize = len(mass_med)
        textpt = [mass_med[int(.7 * arsize)], bound[int(.7 * arsize)]]
        plt.text(textpt[0], textpt[1], 'LUX', color='blue', fontsize=20,
                 rotation=np.arctan(.66) * 180. / np.pi,
                 ha='center', va='bottom')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ LHC ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if lhc[i]:
        print 'Calculating LHC bounds...'
        if channel[i] == 's':
            if dm_spin[i] == 'scalar':
                    ptag = 'scalar'
            elif dm_spin[i] == 'fermion':
                    ptag = 'fermion'
            else:
                    ptag = 'vector'

            files = glob.glob(MAIN_PATH + '/Input_Data/LHC_{:.0f}GeV_'.format(dm_mass[i]) + ptag +\
                              '_' + dm_bilinear[i] + '_' +\
                              ferm_bilinear[i] + '_' + '*.dat')

            print files
            for f in files:
                tag = f[len(f) - f[::-1].find('_'):f.find('.dat')]
                load = np.loadtxt(f)
                load = load[np.argsort(load[:, 0])]
                med = np.logspace(np.log10(np.min(load[:, 0])), np.log10(np.max(load[:, 0])), 100)
                plt_bnds = interpola(med, load[:, 0], load[:, 1])
                if 'dtype' in f:
                    linesty = '--'
                else:
                    linesty = '-'
                plt.plot(med, plt_bnds, linesty, lw=2, color='red', dashes=(20, 10))

            if channel[i] == 's':
                try:
                    plt.text(800., 6, tag, color='red', fontsize=20, rotation=0, ha='right', va='top')
                except:
                    pass
        else:
            if mediator[i] == 's':
                dm_med_tag = 'scalar'
                limit = np.loadtxt(MAIN_PATH + '/Input_Data/' + 'LHC_tchannel_dirac_fermion_scalar_mediator.dat')
                plt.axvline(x=limit, ymin=0, ymax=1, ls='--', lw=2, color='red', dashes=(20, 10))
                plt.text(1250, 0.4, 'Sbottom Search', color='red', fontsize=20,
                         rotation=90, ha='right', va='center')
            elif mediator[i] == 'v':
                dm_med_tag = 'vector'
                limit = np.loadtxt(MAIN_PATH + '/Input_Data/' + 'LHC_tchannel_dirac_fermion_vector_mediator.dat')
                plt.axvline(x=limit, ymin=0, ymax=1, ls='--', lw=2, color='red', dashes=(20, 10))
                plt.text(1500, 0.4, 'Sbottom Search', color='red', fontsize=20,
                         rotation=90, ha='right', va='center')
            else:
                dm_med_tag = 'fermion'
                limit = np.loadtxt(MAIN_PATH + '/Input_Data/' + 'LHC_tchannel_vectorDM_fermionMed.dat')
                plt.axvline(x=limit, ymin=0, ymax=1, ls='--', lw=2, color='red', dashes=(20, 10))
                plt.text(1500, 0.4, 'Sbottom Search', color='red', fontsize=20,
                         rotation=90, ha='right', va='center')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ WIDTH CALCULATION ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    print 'Calculating NWA...'
    mass_med = np.logspace(0., np.log10(2. * 10 ** 3.), 100)

    width_cups = np.zeros_like(mass_med)
    max_cups = np.zeros_like(mass_med)
    for j, m_a in enumerate(mass_med):

        solve_wid = fmin(narrow_width, 2., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                 dm_mass[i], mediator[i], ferms, m_a,
                                                 fm_couplings, dm_couplings, 0.1), disp=False)

        width_cups[j] = np.power(10., 2. * solve_wid)

        solve_wid2 = fmin(narrow_width, 2., args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                   dm_mass[i], mediator[i], ferms, m_a,
                                                   fm_couplings, dm_couplings, 0.5), disp=False)
        max_cups[j] = np.power(10., 2. * solve_wid2)

    print 'Width Calculation:'

    plt.plot(mass_med, width_cups, '-.', lw=2, color='purple')
    plt.text(mass_med[10], width_cups[10], r'$\Gamma / m_\psi = 0.1$', fontsize=14, ha='left',
             va='bottom', color='purple', rotation=0)
    plt.plot(mass_med, max_cups, lw=2, color='purple')
    plt.text(mass_med[10], max_cups[10], r'$\Gamma / m_\psi = 0.5$', fontsize=14, ha='left',
             va='bottom', color='purple')

    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ THERMAL COUPLINGS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
    if thermal_coups:
        print 'Calculating thermal couplings...'
        try:
            load = np.loadtxt(file_name)
            med_full, plt_therm = [load[:, 0], load[:, 1]]
            ml1 = np.logspace(np.log10(med_full[0]), np.log10(.7 * 2. * dm_mass[i]), 100)
            ml2 = np.logspace(np.log10(.73 * 2. * dm_mass[i]), np.log10(1.27 * 2. * dm_mass[i]), 100)
            ml3 = np.logspace(np.log10(1.3 * 2. * dm_mass[i]), 3, 100)
            mlarge = np.concatenate((ml1, ml2, ml3))
            pts = np.power(10, interpola(np.log10(mlarge), np.log10(med_full), np.log10(plt_therm)))
            plt.plot(med_full, plt_therm, lw=2, color='k')

            if channel[i] == 's':
                load1 = np.loadtxt(file_name[:file_name.find('.dat')] + 'c1.dat')
                med_full1, plt_therm_c1 = [load1[:, 0], load1[:, 1]]
                pts1 = np.power(10, interpola(np.log10(mlarge), np.log10(med_full1), np.log10(plt_therm_c1)))
                chold = np.argmax(pts1 > pts)
                pts1 = np.append(pts1[:chold],pts[chold])
                mplt = mlarge[:chold+1]
                plt.plot(mplt, pts1, lw=1, color='k')

                load2 = np.loadtxt(file_name[:file_name.find('.dat')] + 'c10.dat')
                med_full2, plt_therm_c10 = [load2[:, 0], load2[:, 1]]
                pts2 = np.power(10, interpola(np.log10(mlarge), np.log10(med_full2), np.log10(plt_therm_c10)))
                chold = np.argmax(pts2 > pts)
                pts2 = np.append(pts2[:chold], pts[chold])
                mplt = mlarge[:chold + 1]
                plt.plot(mplt, pts2, lw=1, color='k')

                load3 = np.loadtxt(file_name[:file_name.find('.dat')] + 'c100.dat')
                med_full3, plt_therm_c100 = [load3[:, 0], load3[:, 1]]
                pts3 = np.power(10, interpola(np.log10(mlarge), np.log10(med_full3), np.log10(plt_therm_c100)))
                chold = np.argmax(pts3 > pts)
                pts3 = np.append(pts3[:chold], pts[chold])
                mplt = mlarge[:chold + 1]
                plt.plot(mplt, pts3, lw=1, color='k')

            else:
                raise ValueError

        except:
            if channel[i] == 's':
                mass_med_1 = np.logspace(0., np.log10(2 * dm_mass[i] * 0.75), 30)
                mass_med_2 = np.logspace(np.log10(2 * dm_mass[i] * 0.755), np.log10(2 * dm_mass[i] * 1.23), 20)
                mass_med_3 = np.logspace(np.log10(2 * dm_mass[i] * 1.25), 3.1, 30)
                mass_med = np.concatenate((mass_med_1, mass_med_2, mass_med_3))
                med_full = np.logspace(0., 3., 500)
            else:
                mass_med = np.logspace(2., np.log10(2. * 10 ** 3.), 15)
                med_full = np.logspace(2., np.log10(2. * 10 ** 3.), 100)

            t_cups = np.zeros_like(mass_med)
            t_cups1 = np.zeros_like(mass_med)
            t_cups10 = np.zeros_like(mass_med)
            t_cups100 = np.zeros_like(mass_med)
            width_cups = np.zeros_like(mass_med)
            max_cups = np.zeros_like(mass_med)

            for j, m_a in enumerate(mass_med):
                guess_pt = 0.

                solve_c = fmin(t_coupling_omega, guess_pt, args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                            dm_mass[i], mediator[i], ferms, m_a,
                                                            fm_couplings, dm_couplings, 0.), disp=False)
                t_cups[j] = np.power(10., solve_c)

                if channel[i] == 's':
                    #t_cups[j] = np.power(10., solve_c)

                    solve_c1 = fmin(t_coupling_omega, guess_pt, args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                               dm_mass[i], mediator[i], ferms, m_a,
                                                               fm_couplings, dm_couplings, 1.), disp=False)
                    t_cups1[j] = np.power(10., solve_c1)
                    solve_c10 = fmin(t_coupling_omega, guess_pt, args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                               dm_mass[i], mediator[i], ferms, m_a,
                                                               fm_couplings, dm_couplings, 10.), disp=False)
                    t_cups10[j] = np.power(10., solve_c10)
                    solve_c100 = fmin(t_coupling_omega, guess_pt, args=(channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                                                 dm_mass[i], mediator[i], ferms, m_a,
                                                                 fm_couplings, dm_couplings, 100.), disp=False)
                    t_cups100[j] = np.power(10., solve_c100)

                print 'Mass, Sqr Coupling:', m_a, t_cups[j]

            plt_therm = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups)))
            if channel[i] == 't':
                try:
                    if np.all(width_cups == 0):
                        raise ValueError
                    plt_width_lim = np.power(10, interpola(np.log10(med_full), np.log10(mass_med),
                                                           np.log10(width_cups)))
                    plt_width_lim2 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med),
                                                            np.log10(max_cups)))
                    plt.plot(med_full, plt_width_lim, '-.', lw=2, color='purple')
                    plt.text(med_full[10], plt_width_lim[10], r'$\Gamma / m_\psi = 0.1$', fontsize=14, ha='left',
                             va='bottom', color='purple', rotation=0)
                    plt.plot(med_full, plt_width_lim2, '-.', lw=2, color='purple')
                    plt.text(med_full[10], plt_width_lim2[10], r'$\Gamma / m_\psi = 0.5$', fontsize=14, ha='left',
                             va='bottom', color='purple')
                except:
                    pass
            else:
                plt_therm_c1 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups1)))
                plt_therm_c10 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups10)))
                plt_therm_c100 = np.power(10, interpola(np.log10(med_full), np.log10(mass_med), np.log10(t_cups100)))
                for jj in range(len(med_full)):
                    if plt_therm[jj] < plt_therm_c1[jj] or med_full[jj] > dm_mass[i] + 5.:
                        plt_therm_c1[jj] = plt_therm[jj]
                    if plt_therm[jj] < plt_therm_c10[jj] or med_full[jj] > dm_mass[i] + 5.:
                        plt_therm_c10[jj] = plt_therm[jj]
                    if plt_therm[jj] < plt_therm_c100[jj] or med_full[jj] > dm_mass[i] + 5.:
                        plt_therm_c100[jj] = plt_therm[jj]

                plt.plot(med_full, plt_therm_c1, lw=1, color='k')
                plt.plot(med_full, plt_therm_c10, lw=1, color='k')
                plt.plot(med_full, plt_therm_c100, lw=1, color='k')

                np.savetxt(file_name[:file_name.find('.dat')] + 'c1.dat',
                           np.column_stack((mass_med, t_cups1)), fmt='%.3e')
                np.savetxt(file_name[:file_name.find('.dat')] + 'c10.dat',
                           np.column_stack((mass_med, t_cups10)), fmt='%.3e')
                np.savetxt(file_name[:file_name.find('.dat')] + 'c100.dat',
                           np.column_stack((mass_med, t_cups100)), fmt='%.3e')

            plt.plot(med_full, plt_therm, lw=2, color='k')
            np.savetxt(file_name, np.column_stack((mass_med, t_cups)), fmt='%.3e')

    inter_label, mlab = plot_labeler(dm_spin[i], dm_real[i], dm_type[i], dm_bilinear[i], channel[i],
                                     ferm_bilinear[i], mediator[i])
    if channel[i] == 's':
        plt.text(750, 5. * 10 ** -6, inter_label, verticalalignment='bottom',
                 horizontalalignment='right', fontsize=16)
        plt.text(750, 1.5 * 10 ** -6, mlab + ' = {:.0f} GeV'.format(dm_mass[i]), verticalalignment='bottom',
                 horizontalalignment='right', fontsize=16)
    else:
        plt.text(1500, 3. * 10 ** -2, inter_label, verticalalignment='bottom',
                 horizontalalignment='right', fontsize=16)
        plt.text(1500, 2 * 10 ** -2, mlab + ' = {:.0f} GeV'.format(dm_mass[i]), verticalalignment='bottom',
                 horizontalalignment='right', fontsize=16)
    fig.set_tight_layout(True)
    pl.savefig(fig_name)


    # ~~~~~~~~~~~~~~~~~~~~~~~~~~~ CROSS SECTION PLOTS ~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~

    if csec_plots:
        cross = np.zeros_like(plt_therm)
        # cross1 = np.zeros_like(plt_therm)
        # cross10 = np.zeros_like(plt_therm)
        # cross100 = np.zeros_like(plt_therm)
        for j, cup in enumerate(plt_therm):
            cross[j] = cross_section_calc(cup, channel[i], dm_spin[i], dm_real[i], dm_type[i],
                                          dm_mass[i], mediator[i], ferms, med_full[j],
                                          fm_couplings, dm_couplings, 0.)
            # cross1[j] = cross_section_calc(cup, channel[i], dm_spin[i], dm_real[i], dm_type[i],
            #                                dm_mass[i], mediator[i], ferms, med_full[j],
            #                                fm_couplings, dm_couplings, 1.)
            # cross10[j] = cross_section_calc(cup, channel[i], dm_spin[i], dm_real[i], dm_type[i],
            #                                 dm_mass[i], mediator[i], ferms, med_full[j],
            #                                 fm_couplings, dm_couplings, 10.)
            # cross100[j] = cross_section_calc(cup, channel[i], dm_spin[i], dm_real[i], dm_type[i],
            #                                  dm_mass[i], mediator[i], ferms, med_full[j],
            #                                  fm_couplings, dm_couplings, 100.)

        m_plt = med_full[cross > 0.]
        cross = cross[cross > 0.]
        # m_plt1 = med_full[cross1 > 0.]
        # cross1 = cross1[cross1 > 0.]
        # m_plt10 = med_full[cross10 > 0.]
        # cross10 = cross10[cross10 > 0.]
        # m_plt100 = med_full[cross100 > 0.]
        # cross100 = cross100[cross100 > 0.]

        fig_name = plot_namer(dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                              dm_bilinear[i], channel[i], ferm_bilinear[i], extra_tag='CROSS_SEC')
        file_name = file_namer(dm_spin[i], dm_real[i], dm_type[i], dm_mass[i], mediator[i],
                               dm_bilinear[i], channel[i], ferm_bilinear[i], extra_tag='CROSS_SEC')

        fig = plt.figure(figsize=(6., 3.))
        ax = plt.gca()
        ax.set_xscale('log')
        ax.set_yscale('log')

        plt.plot(m_plt, cross, lw=2, color='k')
        # plt.plot(m_plt1, cross1, lw=1, color='k')
        # plt.plot(m_plt10, cross10, lw=1, color='k')
        # plt.plot(m_plt100, cross100, lw=1, color='k')

        np.savetxt(file_name, np.column_stack((m_plt, cross)))
        # np.savetxt(file_name[:file_name.find('.dat')] + 'c1.dat',
        #            np.column_stack((m_plt1, cross1)), fmt='%.3e')
        # np.savetxt(file_name[:file_name.find('.dat')] + 'c10.dat',
        #            np.column_stack((m_plt10, cross10)), fmt='%.3e')
        # np.savetxt(file_name[:file_name.find('.dat')] + 'c100.dat',
        #            np.column_stack((m_plt100, cross100)), fmt='%.3e')

        if (dm_spin[i] == 'fermion' and dm_type[i] == 'dirac') or dm_real[i] == F:
            ylim1 = 8 * 10 ** -27.
            ylim2 = 3 * 10 ** -26.
        else:
            ylim1 = 2. * 8 * 10 ** -27.
            ylim2 = 2. * 3 * 10 ** -26.

        plt.axhline(y=ylim1, lw=2, ls='--', dashes=(20, 10))
        plt.axhline(y=ylim2, lw=2, ls='--', dashes=(20, 10))
        pl.ylim([10. ** -27., 10. ** -25.])
        pl.xlim([1., 1000.])

        if mediator[i] == 's':
            pl.xlabel(r'$m_{\rm{a}}$   [GeV]', fontsize=20)
        elif mediator[i] == 'v':
            pl.xlabel(r'$m_{\rm{v}}$   [GeV]', fontsize=20)
        elif mediator[i] == 'f':
            pl.xlabel(r'$m_{{\psi}}$   [GeV]', fontsize=20)

        pl.ylabel(r'$\left< \sigma \rm{v} \right>_{\rm{vis}}$   [$cm^3 s^{{-1}}$]', fontsize=20)

        # if channel[i] == 's':
        #     plt.text(750, 5. * 10 ** -26, inter_label, verticalalignment='bottom',
        #              horizontalalignment='right', fontsize=16)
        #     plt.text(750, 3. * 10 ** -26, mlab + ' = {:.0f} GeV'.format(dm_mass[i]), verticalalignment='bottom',
        #              horizontalalignment='right', fontsize=16)
        # else:
        #     plt.text(1500, 5. * 10 ** -26, inter_label, verticalalignment='bottom',
        #              horizontalalignment='right', fontsize=16)
        #     plt.text(1500, 3 * 10 ** -26, mlab + ' = {:.0f} GeV'.format(dm_mass[i]), verticalalignment='bottom',
        #              horizontalalignment='right', fontsize=16)

        fig.set_tight_layout(True)
        pl.savefig(fig_name)
