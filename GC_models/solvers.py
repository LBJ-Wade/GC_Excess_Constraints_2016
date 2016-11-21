from helper import *
from globals import *
from models import *
import numpy as np
import glob

try:
    MAIN_PATH = os.environ['GC_MODEL_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'


def t_coupling_omega(lam, channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                     ferms, m_a, fm_couplings, dm_couplings, c_ratio):
    new_coups = np.zeros_like(dm_couplings)
    new_coups[dm_couplings != 0.] = 10. ** lam

    dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                              ferms, m_a, new_coups, fm_couplings, c_ratio)
    soln = omega_dm[0] * hubble ** 2.
    if (dm_spin == 'fermion' and dm_type == 'dirac'):
        soln /= 2.
    elif (dm_spin == 'scalar' and not dm_real):
        soln /= 2.
    elif (dm_spin == 'vector' and not dm_real):
        soln /= 2.

    if np.abs(m_a - 2. * dm_mass) < 0.0:
        return np.abs(dm_class.omega_h() - soln)
    else:
        return np.abs(dm_class.omega_h_approx() - soln)


def narrow_width(lam, channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                ferms, m_a, fm_couplings, dm_couplings, wid, cratio=1., tbeta=1.):
    new_coups = np.zeros_like(dm_couplings)
    new_coups[dm_couplings != 0.] = 10. ** lam
    dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                              ferms, m_a, new_coups, fm_couplings, cratio, tbeta)
    width = dm_class.mediator_width()

    return np.abs(width / m_a - wid)


def cross_section_calc(lam, channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                       ferms, m_a, fm_couplings, dm_couplings, c_ratio):

    new_coups = np.zeros_like(dm_couplings)
    new_coups[dm_couplings != 0.] = lam

    dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                              ferms, m_a, new_coups, fm_couplings, c_ratio)
    #x_freeze = dm_class.x_freeze_out()
    #sigma = dm_class.sig_therm_exact(dm_mass / x_freeze) * inv_GeV2_to_cm2 * 2.998 * 10 ** 10.
    sigma = dm_class.sigma_v_all(0.)* inv_GeV2_to_cm2 * 2.998 * 10 ** 10.

    return sigma

def b_ratio_calc(lam, channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                 ferms, m_a, fm_couplings, dm_couplings, c_ratio):

    new_coups_dm = np.zeros_like(dm_couplings)
    new_coups_fm = np.zeros_like(fm_couplings)
    new_coups_dm[dm_couplings != 0.] = lam
    new_coups_fm[fm_couplings != 0.] = lam

    dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                              [ferms], m_a, new_coups_dm, new_coups_fm, c_ratio)
    if ferms == 'dark':
        dm_pass = True
        fm_pass = False
    else:
        dm_pass = False
        fm_pass = True
    width = dm_class.mediator_width(ferms=fm_pass, dm=dm_pass)
    return width

def direct_detection_csec(channel, dm_spin,  mediator,
                          dm_bilinear, ferm_bilinear, dm_mass):

    clim_list = []
    file = []
    print 'In DD Bounds...'
    if channel == 's':
        mass_med = np.logspace(0., 3., 300)
        print '         [DD] S-channel...'
        if dm_spin == 'fermion':
            print '         [DD] Fermionic DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if dm_bilinear == 's' and ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_s_s_*.dat'.format(dm_mass))
                elif dm_bilinear == 's' and ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_s_ps_*.dat'.format(dm_mass))
                elif dm_bilinear == 'ps' and ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_ps_s_*.dat'.format(dm_mass))
                elif dm_bilinear == 'ps' and ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_ps_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if dm_bilinear == 'v' and ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_v_v_*.dat'.format(dm_mass))
                elif dm_bilinear == 'v' and ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_v_av_*.dat'.format(dm_mass))
                elif dm_bilinear == 'av' and ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_av_v_*.dat'.format(dm_mass))
                elif dm_bilinear == 'av' and ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_fermion_av_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            else:
                print 'Not Implemented...'

        elif dm_spin == 'scalar':
            print '         [DD] Scalar DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_scalar_s_s_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_scalar_s_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_scalar_v_v_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_scalar_v_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            else:
                print 'Not Implemented...'

        elif dm_spin == 'vector':
            print '         [DD] Vector DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_vector_s_s_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_vector_s_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_vector_v_v_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_vector_v_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'

            else:
                print 'Not Implemented...'
                raise ValueError

        for f in file:
            load = np.loadtxt(f)
            bound = 10. ** interpola(np.log10(mass_med), np.log10(load[:,0]), np.log10(load[:, 1]))

    elif channel == 't':
        mass_med = np.logspace(0., 4., 300)
        if dm_spin == 'fermion':
            if mediator == 's':
                file = [MAIN_PATH +
                        '/Input_Data/DD_tchannel_{:.0f}GeV_dirac_fermion_scalar_mediator.dat'.format(dm_mass)]
            elif mediator == 'v':
                file = [MAIN_PATH +
                        '/Input_Data/DD_tchannel_{:.0f}GeV_dirac_fermion_scalar_mediator.dat'.format(dm_mass)]
            else:
                print 'Not Implemented...'

        else:
            print 'Not here yet'


        for f in file:
            load = np.loadtxt(f)
            bound = 10. ** interpola(np.log10(mass_med), np.log10(load[:,0]), np.log10(load[:, 1]))
    else:
        print 'Model may be wrong or not implemented...'



    return mass_med, bound
