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
            return fermionic_dm_spin0_med_schannel(dm_mass, dm_type, f, m_a,
                                                   dm_lambdas[0], dm_lambdas[1],
                                                   lam_f[0], lam_f[1])
        if dm_spin == 'fermion' and mediator == 'v':
            return fermionic_dm_spin1_med_schannel(dm_mass, dm_type, f, m_a,
                                                   dm_lambdas[0], dm_lambdas[1],
                                                   lam_f[0], lam_f[1])
        if dm_spin == 'scalar' and mediator == 's':
            return scalar_dm_spin0_med_schannel(dm_mass, dm_real, f, m_a,
                                                dm_lambdas, lam_f[0], lam_f[1])
        if dm_spin == 'scalar' and mediator == 'v':
            return scalar_dm_spin1_med_schannel(dm_mass, dm_real, f, m_a,
                                                dm_lambdas, lam_f[0], lam_f[1])
        if dm_spin == 'vector' and mediator == 's':
            return vector_dm_spin0_med_schannel(dm_mass, dm_real, f, m_a,
                                                dm_lambdas, lam_f[0], lam_f[1])

        if dm_spin == 'vector' and mediator == 'v':
            return vector_dm_spin1_med_schannel(dm_mass, dm_real, f, m_a,
                                                dm_lambdas, lam_f[0], lam_f[1])
    elif channel == 't':
        if dm_spin == 'fermion' and mediator == 's' and dm_type == 'majorana':
            return dirac_fermionic_dm_spin0_med_tchannel(dm_mass, f, m_a, dm_lambdas[0], dm_lambdas[1])
        if dm_spin == 'fermion' and mediator == 'v' and dm_type == 'dirac':
            return dirac_fermionic_dm_spin1_med_tchannel(dm_mass, f, m_a, dm_lambdas[0], dm_lambdas[1])
        if dm_spin == 'vector' and mediator == 'f' and not dm_real:
            return complex_vector_dm_spin_half_med_tchannel(dm_mass, f, m_a, dm_lambdas[0], dm_lambdas[1])
        if dm_spin == 'vector' and mediator == 'f' and dm_real:
            return real_vector_dm_spin_half_med_tchannel(dm_mass, f, m_a, dm_lambdas[0], dm_lambdas[1])
        else:
            print 't-channel model not yet implemented...'
            raise ValueError
    else:
        print 'Channel not implemented...'
        raise ValueError


def dm_couples(dm_spin, bilinear):
    # Depending on bilinear list, sets lambda_dm couplings to 0 or 1 for
    # scalar, pseduo scalar, vector, or axial vector (in this order)
    #print dm_spin, bilinear
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
        if dm_real:
            real_tag = 'Real'
        else:
            real_tag = 'Complex'
        spin_tag = real_tag + '_' + dm_spin
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


def plot_labeler(dm_spin, dm_real, dm_type, dm_bilinear, channel, ferm_bilinear, mediator):
    f1 = r'$\bar{{f}}$'
    f2 = r'$f$'
    if dm_spin == 'fermion':
        mtag = r'$m_\chi$'
    elif dm_spin == 'scalar':
        mtag = r'$m_\phi$'
    else:
        mtag = r'$m_X$'
    if dm_spin == 'fermion':
        dm2 = r'$\chi$'
        if dm_type == 'dirac':
            dm1 = r'$\bar{{\chi}}$'
        else:
            dm1 = dm2
    elif dm_spin == 'scalar':
        dm2 = r'$\phi$'
        if dm_real:
            dm1 = r'$\phi$'
        else:
            dm1 = r'$\phi^\dagger$'
    else:
        dm2 = r'$X_\mu$'
        if dm_real:
            dm1 = r'$X^\mu$'
        else:
            dm1 = r'$X^{{\mu \dagger}}$'

    if dm_bilinear == 's':
        dm_bi = ''
    elif dm_bilinear == 'ps':
        dm_bi = r'$\gamma^5$'
    elif dm_bilinear == 'v':
        dm_bi = r'$\gamma^\mu$'
    else:
        dm_bi = r'$\gamma^\mu \gamma^5$'

    if ferm_bilinear == 's':
        f_bi = ''
    elif ferm_bilinear == 'ps':
        f_bi = r'$\gamma^5$'
    elif ferm_bilinear == 'v':
        f_bi = r'$\gamma^\mu$'
    else:
        f_bi = r'$\gamma^\mu \gamma^5$'

    if channel == 's':
        tag = dm1 + dm_bi + dm2 + ', ' + f1 + f_bi + f2
    else:
        if mediator == 's':
            med = r'A'
        elif mediator == 'v':
            med = r'$V_\mu$'
        else:
            med = r'\bar{{\psi}}'

        if dm_spin == 'fermion':
            tag = dm1 + dm_bi + r'(1 \pm \gamma^5)' + f1 + med
        else:
            tag = med + dm_bi + r'(1 \pm \gamma^5)' + f1 + dm1
    return tag, mtag


def y_axis_label(dm_spin, dm_bilinear, channel, ferm_bilinear):
    if channel == 's':
        if ferm_bilinear == 's':
            f_t = r'$\lambda_{{f s}}$'
        elif ferm_bilinear == 'ps':
            f_t = r'$\lambda_{{f p}}$'
        elif ferm_bilinear == 'v':
            f_t = r'$g_{{f v}}$'
        else:
            f_t = r'$g_{{f a}}$'

        if dm_spin == 'fermion':

            if dm_bilinear == 's':
                d_t = r'$\lambda_{{\chi s}}$'
            elif dm_bilinear == 'ps':
                d_t = r'$\lambda_{{\chi p}}$'
            elif dm_bilinear == 'v':
                d_t = r'$g_{{\chi v}}$'
            else:
                d_t = r'$g_{{\small \chi a}}$'
        elif dm_spin == 'scalar':

            d_t = r'$\mu_{{\phi}}$'
        else:

            d_t = r'$\mu_{{X}}$'

        return f_t + ' ' + d_t
    else:
        if dm_bilinear == 's':
            return r'$\lambda_{{s,p}}$'
        else:
            return r'$g_{{v,a}}$'
