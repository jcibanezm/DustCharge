import compute_charge_dist as fz
import numpy as np
import scipy.integrate as integrate

microntocm    = 1.0e-4
cmtomicron    = 1.0e4
AAtocm        = 1.0e-8
cmtoAA        = 1.0e8
microntoAA    = 1.0e4
AAtomicron    = 1.0e-4
ergtoeV       = 6.242e11
eVtoerg       = 1.602e-12

hplanck       = 4.135667662e-15 # eV s
clight        = 2.99792458e10   # cm s-1


def pe_energy_dist(Ehere, Elow, Ehigh, y2):
    """
    Photoelectron energy distribution of the escaping electrons. 
    Fraction of attempting electrons with energy between E and E+dE, fE0(E)dE, equation 10 WD01.
    The attemptng electrons assume a parabolic energy distribution.

    syntax: 
            pe_energy_dist(E, Elow, Ehigh, y2)
            
    input:
            Ehere : calculate at this energy, eV
            Elow  : low end of the energy of the attempting electrons, eV
            Ehigh : high end of the energy of the attempting electrons, eV
            y2    : fraction of electrons that escape to infinity.
    
    output:
            fE(E).
    """
    
    fE0 = 6.*(Ehere - Elow)*(Ehigh - Ehere) / (Ehigh - Elow)**3
    
    fE = fE0 / y2
    
    return fE


def totalE_escapingE(hnu, asize, Zhere, grain_type):
    """
    Compute the integral of the photoelectron energy distribution of escaping electrons.
    
    returns:
            energy in units eV
    """
    elow = fz.get_Elow(asize, Zhere)
    ehigh = fz.get_Ehigh(hnu, asize, Zhere, grain_type)
    y2 = fz.get_y2(hnu, asize, Zhere, grain_type)

    emin = fz.get_Emin(asize, Zhere)
    emax = hnu - fz.hplanck*fz.get_nu_pet(asize,Zhere, grain_type) + emin

    ffEE = integrate.quad(pe_energy_dist, emin, emax, args=(elow, ehigh, y2))[0]

    return ffEE


def get_YQcu_hnu_fe(nu, asize, Z, grain_type, Ntot, Qabs, G0=1.7):
    """
    Get the product inside the integral.

    return: in units: eV / AA^2
    """
    import numpy as np
    from scipy.interpolate import interp1d
    import os as sys
    
    hnu = nu*hplanck

    Yield = fz.get_Yield(hnu, asize, Z, grain_type)

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    u_nu  = fz.get_ISRF(hnu, Ntot, G0)

    fe          = totalE_escapingE(hnu, asize, Z, grain_type)
    YQcu_hnu_fe = Yield * Qabs_nu * clight * u_nu / (1.0*hnu) * ergtoeV / cmtoAA**2 * fe
    #YQcu_hnu = Yield * Qabs * clight * u_nu / (1.0*hnu) * ergtoeV / cmtoAA**2

    return YQcu_hnu_fe

def get_pdt_heat(nu, asize, Z, grain_type, Ntot, G0=1.7):
    """
    get the photodetachment heating rate.

    Return:
        Gamma_pdt in units eV s-1.
    """
    import numpy as np

    hnu = nu*hplanck

    u_nu  = fz.get_ISRF(hnu, Ntot, G0)

    sigma_pdt  = fz.get_sigma_pdt(hnu, asize, Z, grain_type)

    nupdt = fz.get_nu_pdt(asize, Z, grain_type)

    Emin = fz.get_Emin(asize, Z)

    pdt_factor = sigma_pdt * clight * u_nu * ergtoeV / (1.0*hnu) * (hplanck*nu - hplanck*nupdt + Emin)  

    return pdt_factor


def get_Gamma_pe_dotdot(asize, Z, grain_type, Ntot, Qabs, G0=1.7):
    """
    Calculate the heating rate per grain due to photoemission.
    Equations 38 and 39 in Weingartner and Draine 2001,134.
    
    Parameters:
        asize:     dust grain size in Amstrongs.
        Z:         grain charge
        grain_type: carbonaceous or silicate dust.
        Ndust:      Total dust column density (actually I think I need to give Av in magnitudes. Check!!!)
        Qabs :      Absorption efficiency table.
        G0:         Scaling of the interstellar radiation field. Default = 1.7 Gnot.

    returns:
        Gamma_pe:  photoelectric heating rate in erg s-1.
    """
    import math
    import numpy as np
    from scipy import integrate
    
    pia2    = math.pi*asize**2

    nu_low  = fz.get_nu_pet(asize, Z, grain_type)
    nu_up   = 13.6 / hplanck

    hnu_low = nu_low * hplanck
    hnu_up  = 13.6

    if hnu_low > hnu_up:
        Gamma_pe = 0.
    else:
        # Integrate the photoemission from dust grains.
        Gamma_pe = integrate.quad(get_YQcu_hnu_fe, nu_low, nu_up, args=(asize, Z, grain_type, Ntot, Qabs, G0))[0]

    # Run the  photodetachment rate.
    nu_pdt_low  = fz.get_nu_pdt(asize, Z, grain_type)
    hnu_pdt_low = nu_pdt_low * hplanck

    if Z >= 0:
        Gamma_pdt = 0
    else:
        if hnu_pdt_low > hnu_up:
            Gamma_pdt = 0
        else:
            Gamma_pdt = integrate.quad(get_pdt_heat, nu_pdt_low, nu_up, args=(asize, Z, grain_type, Ntot, G0))[0]

    Gamma_heat = (pia2 * Gamma_pe + Gamma_pdt)*eVtoerg

    return Gamma_heat


################################################################################################

def Gamma_per_grain(ZZall, Gamma_a_Z, ZZ_fz, fz, GG):
    """
    Computes the heating rate per grain. Equation 38 in Weingartner and Draine 2001.
    This function requires the charge distribution function of the grain. 
    Internally computes the heating rate of the grain at each charge in the distribution.
    Then sums over the product of the charge PDF times heating(Z).
    
    input:
        asize: grain size in Angstroms.
        grain_type: carbonaceous or silicate
        ZZ: charge array
        fz: charge distribution function.
        GG: scaling of the radiation field
        
    return:
        Gamma_pe_a: Heating rate per grain, in units erg/s
    """

    # index in the ZZall array for the charges in ZZ_fz
    zi_down = np.where(ZZall == ZZ_fz[0])[0][0]# find the index of the ZZ_fz[0] in ZZall 
    zi_up   = np.where(ZZall == ZZ_fz[-1])[0][0]# find the index of the ZZ_fz[-1] in ZZall
    
    Gamma_pe_a = np.sum(fz*Gamma_a_Z[zi_down:zi_up+1])
    
    return Gamma_pe_a


