"""
Built Sept 16, 2016
Sam Witte
"""
from models import *

try:
    MAIN_PATH = os.environ['GC_MODEL_PATH']
except KeyError:
    MAIN_PATH = os.getcwd() + '/../'

def build_dm_class(channel, dm_spin, dm_real, dm_type, dm_mass, mediator,
                   f, m_a, dm_lambdas, lam_f):

    if channel == 's':
        if dm_spin == 'fermion' and mediator == 's':
            return fermionic_dm_spin0_med(dm_mass, dm_type, f, m_a, dm_lambdas[0], dm_lambdas[1], lam_f[0], lam_f[1])
        if dm_spin == 'fermion' and mediator == 'v':
            return fermionic_dm_spin1_med(dm_mass, dm_type, f, m_a, dm_lambdas[0], dm_lambdas[1], lam_f[0], lam_f[1])
        if dm_spin == 'scalar' and mediator == 's':
            return scalar_dm_spin0_med(dm_mass, dm_real, f, m_a, dm_lambdas, lam_f[0], lam_f[1])
        if dm_spin == 'scalar' and mediator == 'v':
            return scalar_dm_spin1_med(dm_mass, dm_real, f, m_a, dm_lambdas, lam_f[0], lam_f[1])


def dm_couples(dm_spin, bilinear):
    # Depending on bilinear list, sets lambda_dm couplings to 0 or 1 for
    # scalar, pseduo scalar, vector, or axial vector (in this order)
    print dm_spin, bilinear
    if dm_spin == 'fermion':
        lambda_coef = np.zeros(4)
        if bilinear == 's':
            lambda_coef[0] = 1.
        elif bilinear == 'ps':
            lambda_coef[1] = 1.
        elif bilinear == 'v':
            lambda_coef[2] = 1.
        elif bilinear == 'av':
            lambda_coef[3] = 1.
    else:
        lambda_coef = np.array([1.])
    return lambda_coef

def fm_couples(bilinear):
    # Depending on bilinear list, sets lambda_ferm couplings to 0 or 1 for
    # scalar, pseduo scalar, print dm_spin, bilinear
    lambda_coef = np.zeros(4)
    if bilinear == 's':
        lambda_coef[0] = 1.
    elif bilinear == 'ps':
        lambda_coef[1] = 1.
    elif bilinear == 'v':
        lambda_coef[2] = 1.
    elif bilinear == 'av':
        lambda_coef[3] = 1.
    return lambda_coef


def plot_namer(dm_spin, dm_real, dm_type, dm_mass, mediator, dm_bilinear,
               channel, ferm_bilinear, extra_tag=''):
    if dm_spin == 'fermion':
        spin_tag = dm_type + '_' + dm_spin
    else:
        spin_tag = dm_real + '_' + dm_spin
    mass_tag = '_Mass_{:.2f}_GeV_'.format(dm_mass)

    if mediator == 's':
        med_tag = '_Scalar_Mediator_'
    elif mediator == 'v':
        med_tag = '_Vector_Mediator_'
    else:
        med_tag = '_Fermionic_Mediator_'
    if dm_bilinear == 's':
        dmbi_tag = 'DM_bilinear_scalar_'
    elif dm_bilinear == 'ps':
        dmbi_tag = 'DM_bilinear_pseudoscalar_'
    elif dm_bilinear == 'av':
        dmbi_tag = 'DM_bilinear_axialvector_'
    else:
        dmbi_tag = 'DM_bilinear_vector_'

    if ferm_bilinear == 's':
        fbi_tag = 'Ferm_bilinear_scalar_'
    elif ferm_bilinear == 'ps':
        fbi_tag = 'Ferm_bilinear_pseudoscalar_'
    elif ferm_bilinear == 'av':
        fbi_tag = 'Ferm_bilinear_axialvector_'
    else:
        fbi_tag = 'Ferm_bilinear_vector_'

    interaction_tag = channel + '_channel' + med_tag + dmbi_tag + fbi_tag

    full_tag = MAIN_PATH + '/Plots/' + spin_tag + mass_tag + interaction_tag + extra_tag + '.pdf'

    return full_tag
