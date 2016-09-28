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
                     ferms, m_a, fm_couplings, dm_couplings):
    dm_couplings[dm_couplings != 0.] = 10. ** lam
    dm_class = build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                              ferms, m_a, dm_couplings, fm_couplings)
    return np.abs(dm_class.omega_h() - omega_dm[0] * hubble ** 2.)



def direct_detection_csec(channel, dm_spin,  mediator,
                          dm_bilinear, ferm_bilinear, dm_mass):
    mass_med = np.logspace(0., 3., 300)
    clim_list = []
    file = []
    print 'In DD Bounds...'
    if channel == 's':
        print '         [DD] S-channel...'
        if dm_spin == 'fermion':
            print '         [DD] Fermionic DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if dm_bilinear == 's' and ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_s_*.dat'.format(dm_mass))
                elif dm_bilinear == 's' and ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_ps_*.dat'.format(dm_mass))
                elif dm_bilinear == 'ps' and ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_ps_s_*.dat'.format(dm_mass))
                elif dm_bilinear == 'ps' and ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_ps_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if dm_bilinear == 'v' and ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_v_*.dat'.format(dm_mass))
                elif dm_bilinear == 'v' and ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_av_*.dat'.format(dm_mass))
                elif dm_bilinear == 'av' and ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_av_v_*.dat'.format(dm_mass))
                elif dm_bilinear == 'av' and ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_av_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            else:
                print 'Not Implemented...'
                raise ValueError
        elif dm_spin == 'scalar':
            print '         [DD] Scalar DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_s_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_v_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            else:
                print 'Not Implemented...'
                raise ValueError
        elif dm_spin == 'vector':
            print '         [DD] Vector DM'
            if mediator == 's':
                print '         [DD] Scalar Mediated'
                if ferm_bilinear == 's':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_s_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'ps':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_s_ps_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            elif mediator == 'v':
                print '         [DD] Vector Mediated'
                if ferm_bilinear == 'v':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_v_*.dat'.format(dm_mass))
                elif ferm_bilinear == 'av':
                    file = glob.glob(MAIN_PATH + '/Input_Data/DD_{:.0f}GeV_v_av_*.dat'.format(dm_mass))
                else:
                    print 'Not Implemented...'
                    raise ValueError
            else:
                print 'Not Implemented...'
                raise ValueError
    elif channel == 't':
        print 'Not here yet'
        raise ValueError

    else:
        print 'Model may be wrong or not implemented...'
        raise ValueError

    for f in file:
        load = np.loadtxt(f)
        clim_list.append(np.sqrt(load / inv_GeV2_to_cm2))
    clim = np.max(clim_list)
    norm = clim
    bound = norm * np.power(mass_med, 2.)
    return mass_med, bound
