import numpy as np
import parametric_fz as fzpar
import dust_size_dist as sizedist

def get_epsilon(psi, temp):
    """
    photoelectric heating efficiency from Bakes & Tielens 1994.
    """
    ephere = 0.049 / (1. + (psi/963.)**(0.73)) + 0.037*(temp/1.0e4)/(1.+(psi/2500.))
    return ephere

def Gamma_BT94(ntot, Geff, temp, ne):
    """
    Photoelectric heating rate Bakes & Tielens 1994.
    """
    psi     = Geff * np.sqrt(temp) / ne
    epsilon = get_epsilon(psi, temp)
    Gammahere = 1.3e-24 * epsilon * Geff * ntot
    return Gammahere


def get_Gamma_dotdot_par(asize, Gtot, Z, grain_type):
    """
        Parametric equation of the energy per photoelectron as a function of the grain size, charge, strength of the incident radiation field and charge.
        
        Input parameters:
        asize: Grain size in Angstroms
        G: Scaling of the radiation field in units of Habing field.
        Z: Charge, in units of proton charge.
        grain_type: 'silicate' or 'carbonaceous'
        
        return:
        Gamma_pe^{''} in erg s-1
        """
    
    if grain_type == "silicate":
        Gamma0 = 2.3e-20
        alpha  = 2.39
        zeta   = 2.953042 * (asize / 5.0)**(-1.03848)
    else:
        
        Gamma0 = 4.06768782673e-20
        alpha  = 2.16393168
        zeta   = 0.9935858817946079 * (asize / 5.0)**(-1.04665779)

    gamma_dotdot_pe = Gamma0 * (asize / 5.0)**(alpha) * (Gtot / 1.7) * np.exp(- zeta * Z)

    return gamma_dotdot_pe


def Cooling_par(asize, Gtot, T, ne):
    """
        Parametric cooling function.
        
        returns cooling rate in:
        erg s-1
        """
    import numpy as np
    
    Lambda0 = 5.04746112272e-22 * (asize/5.0)**(2.45132252)
    psi     = ne * np.sqrt(T) * (Gtot*np.sqrt(T)/ne)**(0.2)
    
    coolhere = Lambda0 * psi**(1.17747369697)
    
    return coolhere


def get_Gamma_dot(Gtot, T, ne, grain_size, grain_type):
    """
        Get Gamma'_{pe} for a given grain size, composition and Gtot, T and ne.
        """
    zcent = fzpar.get_zcent(Gtot, T, ne, grain_type, grain_size)
    zwidth = fzpar.get_zwidth(grain_size, grain_type, zcent)
    
    zmin = np.floor(zcent - 5*zwidth)
    zmax = np.ceil(zcent + 5*zwidth)
    
    ZZ = np.arange(zmin, zmax+1)
    
    # Assume a Gaussian distribution for the shape of the charge distribution.
    ffz = np.zeros_like(ZZ)
    ffz = 1.0 / (np.sqrt(2.*np.pi*zwidth**2)) * np.exp(-(ZZ - zcent)*(ZZ - zcent)/(2*zwidth**2))
    
    # get Gamma_dotdot
    Gamma_dotdot_a_Z = get_Gamma_dotdot_par(grain_size, Gtot, ZZ, grain_type)
    
    Gammadot = np.sum(ffz*Gamma_dotdot_a_Z)
    
    Cooldot = Cooling_par(grain_size, Gtot, T, ne)
    
    Gammanet = Gammadot - Cooldot
    
    return Gammadot


def get_Gamma_tot(Gtot, T, ne, amin=3.5, amax=2500):
    """
        Get the total heating rate per hydrogem atom.
        """
    from scipy import integrate
    
    amin, amax = 3.5, 2500
    fheat = lambda grain_size, Gtot, T, ne, grain_type: get_Gamma_dot(Gtot, T, ne, grain_size, grain_type)*sizedist.dnda(grain_size, grain_type)
    
    Gamma_pe_sil, err = integrate.quad(fheat, amin, amax, args=(Gtot, T, ne, "silicate"))
    Gamma_pe_carb, err = integrate.quad(fheat, amin, amax, args=(Gtot, T, ne, "carbonaceous"))
    
    return Gamma_pe_sil + Gamma_pe_carb
