#import matplotlib.pyplot as plt
import numpy as np
import scipy.integrate as integrate
#import math
import gc
#from memory_profiler import profile


print("Loading the library to compute the charge distribution of dust grains.")

kb            = 1.38e-16        # erg K-1
hplanck       = 4.135667662e-15 # eV s
echarge       = 4.80320425e-10  # esu [cm3/2 g1/2 s-1]
clight        = 2.99792458e10   # cm s-1


microntocm    = 1.0e-4
cmtomicron    = 1.0e4
AAtocm        = 1.0e-8
cmtoAA        = 1.0e8
microntoAA    = 1.0e4
AAtomicron    = 1.0e-4
ergtoeV       = 6.242e11
eVtoerg       = 1.602e-12

mH            = 1.6733e-24  # g
me            = 9.109e-28   # g
mC            = 12.011 * mH # g

def Jtilde_0(tau, nu):
    """
    Calculate the reduced collisional charching rate for neutral grains. Equation (3.3) in Drain and Sutin 1987.
    tau = a*kb*T/q^2    is the "reduced temperature"
    nu  = Ze/q          ratio of grain-to-projectile charge.
    """
    import numpy as np
    import math

    J = 1.0 + np.sqrt(math.pi / (2.0*tau))

    return J

def Jtilde_neg(tau, nu):
    """
    Calculate the reduced collisional charching rate for negative grains. Equation (3.4) in Drain and Sutin 1987.
    tau = a*kb*T/q^2    is the "reduced temperature"
    nu  = Ze/q          ratio of grain-to-projectile charge.
    """
    import numpy as np
    import math

    J = (1.0 - nu/tau) * (1.0 + np.sqrt(2.0 / (tau - 2.0*nu)))

    return J

def Jtilde_pos(tau, nu):
    """
    Calculate the reduced collisional charching rate for positive grains. Equation (3.5) in Drain and Sutin 1987.
    tau = a*kb*T/q^2    is the "reduced temperature"
    nu  = Ze/q          ratio of grain-to-projectile charge.
    """
    import numpy as np
    import math

    # Dimensionless measure of the value of the potential maximum.
    if nu <= 0:
        theta = 0.0
    else:
        theta = nu / (1.0 + nu**(-0.5))

    J = (1.0 + np.sqrt(1.0 / (4.0*tau + 3.0*nu)))**2 * np.exp(-theta/tau)

    del theta
    #gc.collect(generation=2)

    return J

def Jtilde(tau, nu):
    if nu < 0:
        J = Jtilde_neg(tau, nu)
    elif nu == 0:
        J = Jtilde_0(tau, nu)
    elif nu > 0:
        J = Jtilde_pos(tau,nu)

    return J

def get_stickCoef(Z, asize, grain_type):
    """
    An electron colliding with a dust grain has some probability of scattering.
    As the grain acquires more electrons the electron afinity decreases.
    Electrons impinging neutral grains may undergo inelastic scattering and transfer its energy onto the grain lattice.
    Positively charge grains are expected te recombine.
    We adopt a maximum scattering probability of Pes = 0.5, then the maximmum sticking coef. is (1-Pes)=0.5 (as in WD01)

    Parameters:
        Z:          Dust grain charge. In units of proton charge.
        asize:      Dust size in Angstroms
        grain_type: "carbonaceous" or "silicate"

    returns:
        stick_coef: electrons sticking coefficient. Value between 0-0.5 (dimensionless).
    """
    import math
    zmin = get_Zmin(asize, grain_type)

    # Number of carbon atoms in the grain.
    Nc   = 468. *(asize/10.0)**3

    # le electron escape length. (le=10 AA from WD01)
    le = 10.0

    #print("I m in get_stick coef:    asize = %.2f     le = "%(asize), le, ",    ratio = %.2f"%(asize/le))

    if Z < 0 :
        if Z <= zmin :
            stick_coef = 0.0
        else :
            stick_coef = 0.5 * (1.0 - math.exp(-asize/le))*1.0/(1.+math.exp(20.0 - Nc))
    elif Z == 0 :
        stick_coef = 0.5 * (1.0 - math.exp(-asize/le))*1.0/(1.+math.exp(20.0 - Nc))
    else :
        stick_coef = 0.5*(1.0 - math.exp(-asize/le))

    del zmin, Nc, le
    #gc.collect(generation=2)

    return stick_coef

#@profile
def Jrate(Z, numdens, ionFrac, Temp, grain_size, partner, grain_type, Jmin=1.0e-200):
    """
    Calculate the rate at which charged particles with number density n, charge q, mass m and sticking coefficient s,
    arrive at the grain surface.
    Syntax: Jrate(Z= <#>, numdens=<#>, Temp=<#>, mass=<#>, stick_coeff=<#>, grain_size=<#>, partner=<string>)

        Z           = Charge of the dust grain
        numdens     = gas number density of the environment, in units cm-3
        Temp        = gas temperature of the environment, in Kelvin
        mass        = mass of the collisional partner, in units gram
        grain_size  = Size of dust grain, in units Angstroms.
        partner     = type of collisional partner, avail 'ion', 'electron'
        grain_type  = either "carbonaceous" or "silicate"
    """
    import numpy as np
    import math

    #print("I'm in Jrate.   \n\tasize = %.2f, \n\tcoll. partner = %s, \n\tn=%.2f, \n\tT=%.2f, \n\txe=%.4f"%(grain_size, partner, numdens, Temp, xe))

    tau = grain_size * AAtocm * kb * Temp / echarge**2

    if partner == "electron":
        nu = -1*Z
        stick_coef  = get_stickCoef(Z, grain_size, grain_type)
        charge_frac = 1.0
        mass        = me
        # number density of electrons.
    else :
        # ion.
        nu         = Z
        stick_coef = 1.0

    if partner == "hydrogen":
        charge_frac = ionFrac
        mass        = mH
    elif partner == "carbon":
        charge_frac = ionFrac
        #charge_frac = np.min(nC, xe)
        mass        = mC
    #elif partner == "magnesium":
    #    charge_frac = min(nMg, xe)
    #    mass        = 24.305 * mH
    #elif partner == "silicon":
    #    charge_frac = min(nSi, xe)
    #    mass        = 28.086 * mH
    #elif partner == "sulfur":
    #    charge_frac = min(nS, xe)
    #    mass        = 32.060 * mH

    Jtilde = 1.0

    if nu < 0:
        Jtilde = Jtilde_neg(tau, nu)
    elif nu == 0:
        Jtilde = Jtilde_0(tau, nu)
    else:
        Jtilde = Jtilde_pos(tau, nu)

    J = numdens * charge_frac * stick_coef * np.sqrt(8.0 * kb * Temp / (math.pi * mass)) * math.pi * (grain_size*AAtocm)**2 * Jtilde

    if J < Jmin:
        J = Jmin

    del tau, nu, stick_coef, charge_frac, mass, Jtilde
    #gc.collect()
    #gc.collect(generation=2)

    return J


def get_Emin(asize, Z):
    """
    calculate the energy shift given a tunneling probability of 10^-3 for negatively charged grains.
    Note: Updated Emin calculation from Weingartner, Draine & Barr 2006. eq. (3).
    Is it really? or is it commented.

    Parameters:
        asize in Angstroms.
        Z in units of proton charge.

    returns: Emin in units of eV.
    """
    import numpy as np

    #print "calculating Emin"
    emin  = -(Z + 1.0)*echarge**2 / (asize*AAtocm) * (1. + (27./asize)**(0.75))**(-1) * ergtoeV

    #if Z >= -1:
    #    emin = 0.0
    #else:
    #    zprime = abs(Z + 1)
    #    nu     = 1.0 * zprime
    #    theta  = nu / (1.0 + nu**(-0.5))
    #    emin   = theta*(1.0 - 0.3*(asize/10.)**(-0.45)*abs(Z+1.)**(-0.26))

    return emin

def get_IPv(asize, Z, grain_type):
    """
    Calculate the ionization potential of a grain of a given size a, and charge Z.
    Parameters:
        asize in Angstroms
        Z charge in units of proton charge.

    returns: ionizatin potential in units of eV.
    """
    import numpy as np


    if grain_type == "carbonaceous" or grain_type == "Carbonaceous":
        W = 4.4 # eV. Work function for carbonaceous grains.
    elif grain_type == "silicate" or grain_type == "Silicate":
        W = 8.0

    # electron charge given in esu. 1esu = g1/2 cm3/2 s-1.
    # echarge^2 / asize -> convert size from Angstrom to cm 1AA = 1.0e-8 cm.
    # echarge^2/asize is now in ergs. convert that to eV -> 1 erg = 6.242e11 eV.

    if Z >= 0 :
        IPv = W + (Z + 0.5)*echarge**(2)/(asize*AAtocm)*ergtoeV + (Z + 2.)*echarge**(2)/(asize*AAtocm)*0.3/asize*ergtoeV
    else:
        IPv = get_EA(asize, Z+1, grain_type)

    del W
    #gc.collect(generation=2)

    return IPv

def get_EA(asize, Z, grain_type):
    """
    Calculates the electron affinity EA(Z) for negatively charged dust grains.
    The electron affinity is the differece between infinity and the "lowest occupied molecular orbital" (LUMO) for the grain of charge Z.
    Parameters:
        asize in Amstrongs
        Z in proton charges.
        grain_type is either carbonaceous or silicate.

    returns: Electron Afinity in eV.
    """
    import numpy as np

    if grain_type == "carbonaceous" or grain_type == "Carbonaceous":
        # equation 4 in Weingartner & Draine 2001.
        W  = 4.4
        EA = W + (Z-0.5)*echarge**2/(asize*AAtocm)*ergtoeV - echarge**(2)/(asize*AAtocm)*4.0/(asize+7.0)*ergtoeV
    else:
        # Equation 5 in WD01.
        W   = 8.0
        Ebg = 5.
        EA  = W - Ebg + (Z-0.5)*echarge**2/(asize*AAtocm)*ergtoeV

    del W
    #gc.collect(generation=2)

    return EA

def get_nu_pet(asize, Z, grain_type):
    """
    Calculate the photon frequency threshold for the ionization of a grain with a given size a, and charge Z.
    Parameters:
        asize in Angstroms.
        Z charge in units of proton charge.
        grain_type can be either carbonaceous or silicate.

    returns: photon frequency photoelectron threshold in units of 1/s.
    """
    import numpy as np

    if Z >= -1:
        nu_pet = 1.0/hplanck * get_IPv(asize,Z, grain_type)
    else:
        nu_pet = 1.0/hplanck *( get_IPv(asize,Z, grain_type) + get_Emin(asize, Z) )

    return nu_pet

def get_Theta(hnu, asize, Z, grain_type):
    """
    calculate the parameter Theta for the parametrized photoelectric yield. eq. (9) WD01.
    Parameters:
        hnu in eV
        asize in Amstrong
        Z in proton charge
        grain_type either carbonaceous or silicates.

    returns Theta (dimensionless)
    """
    import math
    import numpy as np

    # electron charge given in esu. 1esu = g1/2 cm3/2 s-1.
    # echarge^2 / asize -> convert size from Angstrom to cm 1AA = 1.0e-8 cm.
    # echarge^2/asize is now in ergs. convert that to eV -> 1 erg = 6.242e11 eV.

    if Z >=0:
        Theta = hnu - hplanck * get_nu_pet(asize, Z, grain_type) + (Z + 1.)*echarge**2 / (asize*AAtocm)*ergtoeV
    else:
        Theta = hnu - hplanck * get_nu_pet(asize, Z, grain_type)

    if Theta < 0:
        Theta = 0.

    return Theta

def get_Ehigh(hnu, asize, Z, grain_type):
    """
    high end of the energy o fthe electrons attempting to escape the grain.
    Parameters:
        hnu in eV
        asize in Angstrom
        Z charge in proton charges.

    returns Ehigh in eV.
    """
    import numpy as np

    if Z < 0:
        Ehigh = get_Emin(asize, Z) + hnu - hplanck*get_nu_pet(asize, Z, grain_type)
    else:
        Ehigh = hnu - hplanck*get_nu_pet(asize, Z, grain_type)

    #Ehigh = max(0, Ehigh)

    return Ehigh

def get_Elow(asize, Z):
    """
    low end of the energy of the electrons attempting to escape the grain
    Parameters:
        hnu in eV
        asize in Angstrom.
        Z charge in proton charges

    returns: Elow in eV.
    """
    import numpy as np

    if Z < 0:
        Elow = get_Emin(asize, Z)
    else:
        Elow = -(Z+1.)*echarge**2 / (asize*AAtocm)*ergtoeV

    #Elow = max(0, Elow)

    return Elow

def get_y2(hnu, asize, Z, grain_type):
    """
    y2 is the fraction of attempting electrons that escape to infinity.
    Parameters:
        hnu: incident photon energy in eV.
        asize: grain size in Angstroms.
        Z: grain charge in proton charge units.
        grain_type: either carbonaceous or silicate

    returns: y2 (dimensionless.)
    """
    import numpy as np

    if Z >= 0:
        Ehigh = get_Ehigh(hnu, asize, Z, grain_type)
        Elow  = get_Elow(asize, Z)

        if Ehigh <= Elow:# or Ehigh < 0.0:
            y2 = 0.0
        else:
            if Ehigh < 0:
                Enorm = abs(Ehigh)
                Elow  = Elow + Enorm
                Ehigh = Ehigh + Enorm
                y2 = Ehigh**2 * (Ehigh - 3.0*Elow) / (Ehigh - Elow)**3
            else:
                y2 = Ehigh**2 * (Ehigh - 3.0*Elow) / (Ehigh - Elow)**3
    else:
        y2 = 1.0

    return y2

def get_y1(hnu, asize, grain_type):
    """
    estimate of the escape lengths to photon attenuation length of the grain.
    Parameters:
        hnu in eV
        asize in Angstrom.
        grain_type: carbonaceous or silicates

    return: y1 in dimensionless.
    """
    import numpy as np
    import math

    # le, electron escape lengths. We use 10 AA, same value used by Bakes & Tielens 1994 and Weingarten and Draine 2001.
    # Value motivated by data from Hino, Sato & Inokutchi 1976.
    le = 10.0

    #la is the photon attenuation length, given by the inverse of the complex refractive index of the material.

    wavelength = clight / ( hnu / hplanck ) * cmtoAA

    if grain_type == "carbonaceous" or grain_type == "Carbonaceous":

        #ref_index_perp, ref_index_par = get_Im_RefIndex_Carbonaceous(hnu)
        #la_inv  = 4.0 * math.pi / wavelength * (2./3. * ref_index_perp + 1./3.*ref_index_par)
        #la      = 1. / la_inv
        la = 100.0

    elif grain_type == "silicate" or grain_type == "Silicate":

        #ref_index = get_Im_RefIndex_Silicates(hnu)
        #la        =  wavelength / (4.0*math.pi*ref_index)
        # Bakes and Tielens 1994 use a constant factor of 100 AA, for the photoatenuation length, motivated from data by
        # Pope & Swenberg 1982.
        la = 100.0


    alpha = asize/la + asize/le
    beta  = asize/la

    try:
        exp_alpha = math.exp(-alpha)
    except OverflowError:
        exp_alpha = 0

    try:
        exp_beta = math.exp(-beta)
    except OverflowError:
        exp_beta = 0

    y1 = (beta/alpha)**2 *(alpha**2. - 2.*alpha + 2. - 2.*exp_alpha)/(beta**2. - 2.*beta + 2. - 2.*exp_beta)

    del wavelength, alpha, beta, exp_alpha, exp_beta
    #gc.collect()
    #gc.collect(generation=2)

    return y1

def get_y0(hnu, asize, Z, grain_type):
    """
    y0.
    Parameters:
        hnu in eV
        asize in Angstrom
        Z in proton charge.
        grain_type either carbonaceous or silicate.

    returns: y0 in dimensionless.
    """
    import numpy as np

    Theta = get_Theta(hnu, asize, Z, grain_type)

    if grain_type == "carbonaceous" or grain_type == "Carbonaceous":
        W     = 4.4 # eV
        TW5   = (Theta/W)**5
        y0    = 9.0e-3 * TW5 / (1. + 3.7e-2*TW5)
    elif grain_type == "silicate" or grain_type == "Silicate":
        W     = 8.0
        TW    = Theta/W

        num   = 0.5*TW
        den   = (1.0 + 5.0*TW)

        if num*den > 0:
            y0 = num/den
        else:
            y0 = 0.0

        #y0    = 0.5*TW/(1.0 + 5.0*TW)

    del Theta, W
    #gc.collect(generation=2)

    return y0

def get_Yield(hnu, asize, Z, grain_type):
    """
    Calculate the photoelectric yield for an incident photon with energy hnu, onto a dust grain of size a, and charge Z.
    Parameters:
        hnu in eV
        asize in Angstrom
        Z charge in proton charges.

    returns: Yield in dimensionless units.
    """
    import numpy as np

    y0 = get_y0(hnu, asize, Z, grain_type)
    y1 = get_y1(hnu, asize, grain_type)
    y2 = get_y2(hnu, asize, Z, grain_type)

    Yield = y2 * min(y0*y1, 1.0)

    del y0, y1, y2
    #gc.collect(generation=2)

    return Yield

def get_BB_spec(hnu, T):
    """
    Get the energy density for a black body spectrum at a given energy (hnu), and temperature (T).
    wfac is the dilution factor given by Mathis et al 1983.
    Parameters:
        hnu in eV
        T in K
        wfac is dimensionless

    returns: BB in  erg s-1 cm-2 s
    basically, energy rate, per unit frequency per solid angle.
    """
    import math
    import numpy as np

    nu_here = hnu / hplanck

    BB = 2. * hplanck * nu_here**3 / (clight**2) * 1.0/(math.exp(hnu/(kb*T*ergtoeV))-1.) * eVtoerg

    del nu_here
    #gc.collect(generation=2)

    return BB

def get_u_BB(hnu, T):
    """
    returns the spectral energy density, u, at energy hnu for a black body spectrum with temperature T.
    Parameters:
        hnu in eV
        T in Kelvin.

    Returns: erg cm-3 s.
    """
    import math

    BB = get_BB_spec(hnu, T)

    u = 4.0 * math.pi / clight * BB

    del BB
    #gc.collect(generation=2)

    return u

def get_ISRF(hnu, Ntot, G0=1.0):
    """
    Get the radiation field energy density at a given energy.  Spectrum estimated by Mezger, Mathis & Panagia (1982) and Matiz, Mezger & Panagia (1983).
    Parameters:
        hnu in eV
        Ntot: total hydrogen column density, Ntot = NH+ + NH + 2NH2

    returns: u_nu in erg cm-3 s.
    """
    import math
    import numpy as np

    nu_here = hnu / hplanck

    Av = Ntot / (1.87e21) #Relation of Av to the total hydrogen column density N H,tot (Bohlin, Savage & Drake 1978)

    if hnu > 13.6:
        unu = 0.
    elif 11.2 < hnu <= 13.6:
        unu = 3.328e-9 * hnu**(-4.4172)
    elif 9.26 < hnu <= 11.2:
        unu = 8.463e-13 * hnu**(-1)
    elif 5.04 < hnu <= 9.26:
        unu = 2.055e-14 * hnu**(0.6678)
    elif hnu <= 5.04:
        unu_1 = 1.0e-14 * get_u_BB(hnu, 7500)
        unu_2 = 1.65e-13* get_u_BB(hnu, 4000)
        unu_3 = 4.0e-13 * get_u_BB(hnu, 3000)

        unu =  nu_here * (unu_1 + unu_2 + unu_3)

    if unu != unu:
        unu = 0.

    if unu == 0:
        u_field_edens = 0
    else:
        if nu_here == 0 :
            u_field_edens = 0.
        else:
            u_field_edens = unu / nu_here

    # The constant 0.8868078539 is because this ISRF gives a G0=1.12 so I have to scale it to get a G0=1.0
    u_field_edens = u_field_edens*G0*0.8868078539

    #Av  = Ndust*100.0 / 1.8e21
    tau = Av / 1.086
    # tau = -2.5*Av # In Walch et al 2015.

    u_field_edens = u_field_edens * np.exp(-tau)

    return u_field_edens


def ISRF_nu(nu_here, Ntot, G0=1.0):
    """
    Get the radiation field energy density at a given energy.  Spectrum estimated by Mezger, Mathis & Panagia (1982) and Matiz, Mezger & Panagia (1983).
    Parameters:
        nu in frequency
        Ntot: total hydrogen column density.
        G0: Scaling of the radiation field.

    returns: u_nu in erg cm-3 s.
    """
    import math
    import numpy as np

    hnu = nu_here*hplanck
    Av = Ntot / (1.87e21) #Relation of Av to the total hydrogen column density N H,tot (Bohlin, Savage & Drake 1978)

    if hnu > 13.6:
        unu = 0.
    elif 11.2 < hnu <= 13.6:
        unu = 3.328e-9 * hnu**(-4.4172)
    elif 9.26 < hnu <= 11.2:
        unu = 8.463e-13 * hnu**(-1)
    elif 5.04 < hnu <= 9.26:
        unu = 2.055e-14 * hnu**(0.6678)
    elif hnu <= 5.04:
        unu_1 = 1.0e-14 * get_u_BB(hnu, 7500)
        unu_2 = 1.65e-13* get_u_BB(hnu, 4000)
        unu_3 = 4.0e-13 * get_u_BB(hnu, 3000)
        unu   =  nu_here * (unu_1 + unu_2 + unu_3)
    if unu != unu:
        unu = 0.
    if unu == 0:
        u_field_edens = 0
    else:
        if nu_here == 0 :
            u_field_edens = 0.
        else:
            u_field_edens = unu / nu_here
    # The constant 0.8868078539 is because this ISRF gives a G0=1.12 so I have to scale it to get a G0=1.0
    u_field_edens = u_field_edens*G0*0.8868078539
    tau           = Av / 1.086
    # tau = -2.5*Av # Dust absorption (van Dishoeck & Black 1988)

    u_field_edens = u_field_edens * np.exp(-tau)

    return u_field_edens

def get_G0(G0=1.0):
    """
    Calculate the ratiation field in units of Habing fields.
    Parameters:
        G0 : how I want to scale Mathis et al (1983) ISRF

    Returns:
        G in units of Habing field.
    """
    import scipy.integrate as integrate

    nu_min = 6.0  / hplanck
    nu_max = 13.6 / hplanck

    u_here = integrate.quad(ISRF_nu, nu_min, nu_max, args=(0.0, G0))
    #print(u_here)
    # ergs cm-3
    u_Hab = 5.33e-14

    # Calculate the interstellar radiation field in units of Habing fields.
    G_here = u_here[0] / u_Hab

    del nu_min, nu_max, u_here, u_Hab
    return G_here

def get_G(Ntot, G0=1.0):
    """
    Calculate the ratiation field in units of Habing fields.
    For a given G0 and Extinction.

    Parameters:
        G0 : how I want to scale Mathis et al (1983) ISRF
        Ntot : total hydrogen column density.

    Returns:
        G in units of Habing field.
    """
    import scipy.integrate as integrate

    nu_min = 6.0 /hplanck
    nu_max = 13.6/hplanck

    U_here = integrate.quad(ISRF_nu, nu_min, nu_max, args=(Ntot, G0))

    # ergs cm-3
    u_Hab = 5.33e-14

    # Calculate the interstellar radiation field in units of Habing fields.
    G_here = U_here[0] / u_Hab

    del nu_min, nu_max, U_here, u_Hab
    return G_here

def get_nu_pdt(asize, Z, grain_type):
    """
    When Z<0, the -Z attached electrons occupy energy levels above the valence band.
    The photodetachment threshold energy is given by equation 18 (WD01).
    """

    EA   = get_EA(asize, Z+1, grain_type)
    Emin = get_Emin(asize, Z)

    hnu_pdt = EA + Emin

    nu_pdt = hnu_pdt / hplanck

    del EA, Emin, hnu_pdt
    #gc.collect(generation=2)

    return nu_pdt

def get_sigma_pdt(hnu, asize, Z, grain_type):
    """
    Photodetachment cross section.
    """

    DeltaE  = 3.0

    hnu_pdt = hplanck * get_nu_pdt(asize, Z, grain_type)

    x = (hnu - hnu_pdt) / DeltaE

    sigma = 1.2e-17 * abs(Z) * x / (1.0 + x**2/3.0)**2

    del DeltaE, hnu_pdt, x
    #gc.collect(generation=2)

    return sigma


def get_YQcu_hnu(nu, asize, Z, grain_type, Ntot, Qabs, G0=1.0):
    """
    Get the product inside the integral.

    return: in units: AA^-2
    """
    import numpy as np
    from scipy.interpolate import interp1d
    import os as sys

    # Reading the table here every time I call this function is a waste of time.
    # Giving the Qabs table is somehow better, I guess.
    # Let's see what I get.

    hnu = nu*hplanck

    Yield = get_Yield(hnu, asize, Z, grain_type)

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    #Qabs_basic  = get_Qabs(asize, grain_type)
    # get Qabs, should get, asize, nu (in microns), type. and interpolate the table.

    #print("This is Qabs= %.3f, at %.3f microns"%(Qabs_nu, lambda_here))
    #print(Qabs)

    #print("Qabs = %.3f, Qabs_nu = %.3f"%(Qabs_basic, Qabs_nu))

    u_nu  = get_ISRF(hnu, Ntot, G0)

    YQcu_hnu = Yield * Qabs_nu * clight * u_nu / (1.0*hnu) * ergtoeV / cmtoAA**2
    #YQcu_hnu = Yield * Qabs * clight * u_nu / (1.0*hnu) * ergtoeV / cmtoAA**2

    del hnu, Yield, Qabs, u_nu
    #gc.collect(generation=2)

    return YQcu_hnu


def get_YQnu(nu, asize, Z, grain_type, Qabs):
    """
    Get the product inside the integral.

    return: in units: AA^-2
    """
    import numpy as np
    from scipy.interpolate import interp1d
    import os as sys

    # Reading the table here every time I call this function is a waste of time.
    # Giving the Qabs table is somehow better, I guess.
    # Let's see what I get.

    hnu = nu*hplanck

    Yield = get_Yield(hnu, asize, Z, grain_type)

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    YQnu = Yield * Qabs_nu

    return YQnu


def get_pdt_factor(nu, asize, Z, grain_type, Ntot, G0=1.0):
    """
    get the photodetachment rate factor inside the integral.
    """
    import numpy as np

    hnu = nu*hplanck

    u_nu  = get_ISRF(hnu, Ntot, G0)

    sigma_pdt  = get_sigma_pdt(hnu, asize, Z, grain_type)

    pdt_factor = sigma_pdt * clight * u_nu / (1.0*hnu) * ergtoeV

    del hnu, u_nu, sigma_pdt
    #gc.collect(generation=2)

    return pdt_factor

def basic_integral(func, nu_low, nu_up, asize, Z, grain_type, Ntot, G0, N=100):
    """
    Calculate the integral.
    """
    import numpy as np

    nu   = np.linspace(nu_low, nu_up, N)
    fx   = np.array(np.zeros_like(nu))

    for ii in range(N):
        fx[ii] = func(nu[ii], asize, Z, grain_type, Ntot, G0)
        #fx   = f(nu, asize, Z, grain_type, Ndust, G0)

    area = np.sum(fx)*(nu_up-nu_low)/N

    del nu, fx
    return area

#@profile
def Jrate_pe_test(asize, Z, grain_type, Ntot, Qabs, G0=1.0):
    """
    Calculate the photo emission rate given,
    Parameters:
        asize:     dust grain size in Amstrongs.
        Z:         grain charge
        grain_type: carbonaceous or silicate dust.

    returns:
        Jpe:  Rate at which electrons are detached from the grain surface by photons.
    """
    import math
    import numpy as np
    from scipy import integrate

    pia2 = math.pi*asize**2

    hnu_low = get_nu_pet(asize, Z, grain_type) * hplanck
    hnu_up  = 13.6

    nu_low = get_nu_pet(asize, Z, grain_type)
    nu_up  = 13.6 / hplanck

    if hnu_low > hnu_up:
        Jpe_pet = 0.
    else:
        for i in range(1):
            # Integrate the photoemission from dust grains.
            Jpe_pet = integrate.quad(get_YQcu_hnu, nu_low, nu_up, args=(asize, Z, grain_type, Ntot, Qabs, G0))[0]
            #Jpe_pet = basic_integral(get_YQcu_hnu, nu_low, nu_up, asize, Z, grain_type, Ndust, G0)

        # Run the  photodetachment rate.
        nu_pdt_low  = get_nu_pdt(asize, Z, grain_type)
        hnu_pdt_low = nu_pdt_low * hplanck

        if Z >= 0:
            Jpe_pdt = 0
        else:
            if hnu_pdt_low > hnu_up:
                Jpe_pdt = 0
            else:
                #for i in range(1000):
                Jpe_pdt = integrate.quad(get_pdt_factor, nu_pdt_low, nu_up, args=(asize, Z, grain_type, Ntot, G0))[0]
                #Jpe_pdt = basic_integral(get_pdt_factor, nu_pdt_low, nu_up, asize, Z, grain_type, Ndust, G0)

                Jpe = pia2 * Jpe_pet + Jpe_pdt

    del pia2, hnu_low, hnu_up, nu_low, nu_up, Jpe_pet, nu_pdt_low, hnu_pdt_low, Jpe_pdt
    #gc.collect()
    #gc.collect(generation=2)

    return Jpe


def Jrate_pe(asize, Z, grain_type, Ntot, Qabs, G0=1.0):
    """
    Calculate the photo emission rate given,
    Parameters:
        asize:     dust grain size in Amstrongs.
        Z:         grain charge
        grain_type: carbonaceous or silicate dust.
        Ntot: Total hydrogen column density
        Qabs: Absorption efficiency table for the appropriate grain size and type.
        G0: Scaling of the Mathis, Mezger radiation field.

    returns:
        Jpe:  Rate at which electrons are detached from the grain surface by photons.
    """
    import math
    import numpy as np
    from scipy import integrate

    pia2 = math.pi*asize**2

    hnu_low = get_nu_pet(asize, Z, grain_type) * hplanck
    hnu_up  = 13.6

    nu_low = get_nu_pet(asize, Z, grain_type)
    nu_up  = 13.6 / hplanck

    #print("nu_low frequency", nu_low)
    #print("nu_up  frequency", nu_up )

    if hnu_low > hnu_up:
        Jpe_pet = 0.
    else:
        # Integrate the photoemission from dust grains.
        Jpe_pet = integrate.quad(get_YQcu_hnu, nu_low, nu_up, args=(asize, Z, grain_type, Ntot, Qabs, G0))[0]
        #Jpe_pet = basic_integral(get_YQcu_hnu, nu_low, nu_up, asize, Z, grain_type, Ndust, G0)

    # Run the  photodetachment rate.
    nu_pdt_low  = get_nu_pdt(asize, Z, grain_type)
    hnu_pdt_low = nu_pdt_low * hplanck

    if Z >= 0:
        Jpe_pdt = 0
    else:
        if hnu_pdt_low > hnu_up:
            Jpe_pdt = 0
        else:
            Jpe_pdt = integrate.quad(get_pdt_factor, nu_pdt_low, nu_up, args=(asize, Z, grain_type, Ntot, G0))[0]
            #Jpe_pdt = basic_integral(get_pdt_factor, nu_pdt_low, nu_up, asize, Z, grain_type, Ndust, G0)

    Jpe = pia2 * Jpe_pet + Jpe_pdt

    del pia2, hnu_low, hnu_up, nu_low, nu_up, Jpe_pet, nu_pdt_low, hnu_pdt_low, Jpe_pdt

    return Jpe

def Jrate_pe_pdt(asize, Z, grain_type, Ntot, G0=1.0):
    """
    Calculate the photo emission rate given,
    Parameters:
        asize:     dust grain size in Amstrongs.
        Z:         grain charge
        grain_type: carbonaceous or silicate dust.
        Ntot: Total hydrogen column density
        G0: scaling of the radiation field, units of Habing field.

    returns:
        Jpe:  Rate at which electrons are detached from the grain surface by photons.
    """
    import math
    import numpy as np
    from scipy import integrate

    pia2 = math.pi*asize**2

    hnu_up = 13.6
    nu_up  = 13.6 / hplanck

    # Run a photodetachment rate.
    nu_pdt_low  = get_nu_pdt(asize, Z, grain_type)
    hnu_pdt_low = nu_pdt_low * hplanck

    if Z >= 0:
        Jpe_pdt = 0
    else:
        if hnu_pdt_low > hnu_up:
            Jpe_pdt = 0
        else:
            Jpe_pdt = integrate.quad(get_pdt_factor, nu_pdt_low, nu_up, args=(asize, Z, grain_type, Ntot, G0))[0]
            #Jpe_pdt = basic_integral(get_pdt_factor, nu_pdt_low, nu_up, asize, Z, grain_type, Ndust, G0)

    Jpe = Jpe_pdt

    del pia2, hnu_up, nu_up, nu_pdt_low, hnu_pdt_low, Jpe_pdt
    gc.collect(generation=2)

    return Jpe



def get_Zmin(asize, grain_type):
    """
    Given the dust size and grain type, compute the minimum, negative, charge possible for the dust.
    The minimum charge is the most negative charge before the dust starts spitting electrons due to the strong Coulomb repulsion.
    Also known as electron field emission.

    asize in Amstrongs.
    grain_type: carbonaceous or silicate

    returns:
        Zmin: units of proton charge.
    """
    import math
    import numpy as np

    if grain_type == "carbonaceous" or  grain_type == "Carbonaceous":
        Uait_V = -1.*(3.9 + 0.12*asize + 2.0/asize)
    else:
        Uait_V = -1.*(2.5 + 0.07*asize + 8.0/asize)

    Zmin = math.floor(Uait_V / 14.4 * asize) + 1

    del Uait_V
    #gc.collect(generation=2)

    return Zmin

# Compute Zmax
def get_Zmax(asize, grain_type, hnu_max=13.6):
    """
    Get the most positive charge that can be aquired by a grain given the local strength of the ISRF.
    asize in Amstrongs.
    grain type: either carbonaceous or silicate.
    hnu_max = maximum energy of the interstellar radiation field. Default 13.6 eV for HI regions.

    return:
        Zmax: units of proton charge.
    """
    import math
    import numpy as np

    if grain_type == "carbonaceous" or  grain_type == "Carbonaceous":
        W = 4.4
    else:
        W = 8.0

    Zmax = math.floor(((hnu_max - W)/14.4*asize + 0.5 - 0.3/asize)/(1+0.3/asize))

    del W
    #gc.collect(generation=2)

    return Zmax

def CR_xe(nH2, zeta=1.0e-17):
    """
    Given a molecular hydrogen number density, return the density of secondary electrons
    from the Caselli et al 2002 model 3. (Also equation 2 in Ivlev 2015)
    """

    if nH2 > 1.0e2:
        xe = 6.7e-6*(nH2)**(-0.56)*np.sqrt(zeta/1.0e-17)
    else:
        xe = 0.0

    return xe

def CRparticles_spectra(E, particle="electron"):
    """
    Energy spectrum of particles (electrons/protons) resulting from CR ionizations.
    """

    E0Mev = 500 #MeV
    E0 = E0Mev * 1.0e6

    if (particle == "electron"):
        alpha = -1.5
        beta  = 1.7
        C     = 2.1e18
    else:
        alpha = -0.8   # 0.1
        beta  = 1.9    # 2.8
        C     = 2.4e15 # 2.4e15

    # Equation 1 in Ivlev et al 2015
    j = C * E**(alpha) / (E + E0)**((beta))

    return j

def sec_e_yield(E):
    """
    secondary emission yield of electrons, obtained by averaging over the velocity distriubution
    of the emitted electrons, believed to be a non-Maxwellian distribution, decaying as E^-1.
    Following Ivlev et al 2015, we employ the Sternglass formyla (Horanyi et al 1988).
    """

    # I think I should convert the energy into eV!

    deltae_max = 2.0 # typically between 1.5 and 2.5 Draine & Salpeter 1979.
    Emax       = 0.3 * 1.0e3 # typically between 0.2 and 0.4 keV (Draine & Salpeter 1979)

    eYield = deltae_max * E / Emax * np.exp(2. - 2 * np.sqrt(E / Emax))

    return eYield

def Jrate_CR(nH2, asize, Z, grain_type, zeta=1.0e-17):
    """
    Calculate the current of electrons/protos resulting from cosmic ray ionization.

    This particles have non-maxwellian velocity distribution functions.

    Following Ivlev et al 2015
    """

    import math
    import numpy as np
    from scipy import integrate

    pia2 = math.pi*asize**2

    xe = CR_xe(nH2, zeta=zeta)

    hnu_up = 13.6
    nu_up  = 13.6 / hplanck

    Elow = 1.5e-2 # eV
    Eup  = 1.0e10 # infinity.

    stick = get_stickCoef(Z, asize, grain_type)

    def CR_current(E, stick):
        current = 4. * math.pi * CRparticles_spectra(E)*(stick - sec_e_yield(E))
        return current

    #J_CR = ne* pia2 * integrate.quad(CR_current, Elow, Eup, args=(stick))[0]
    J_CR = 0.

    return J_CR

def Jrate_pe_CR(zeta, asize, Z, grain_type, Qabs):
    """
    Calculate the photoelectric heating rate prduce by phosphorecense of
    molecular hydrogen due to secondary electros resuting from cosmic ray
    ionization of hydrogen.

    Parameters:
        zeta: CR ionization rate
        asize: grain size
        Z: grain charges
        grain_type: carbonaceous or silicate
        Qabs: absorption efficiency table.

    Returns:
        Charging rate by CR-induced photons. Units = s^-1.
    """

    import math

    pia2 = math.pi*asize**2

    omega = 0.5       # dust albedo
    Rv    = 3.1       # Slope of the extinction curve
    #NH2_mag = 1.87e21 # Typical dust to extinction ratio.(Bohlin, Savage & Drake 1978)
    NH2_mag = 1.0e21 # Typical dust to extinction ratio.(Bohlin, Savage & Drake 1978)

    FUV = 960. * (1. / (1.-omega))* (NH2_mag / 1.0e21) * (Rv/3.2)**1.5 * (zeta / 1.0e-17)

    yyQabs = get_avgYieldQabs(Qabs, asize, Z, grain_type)
    #YnuQabs = 0.04

    #Jpe_CR = pia2 * FUV * YnuQabs
    Jpe_CR = pia2 * FUV * yyQabs * AAtocm**2
    #Jpe_CR = 0

    return Jpe_CR


def get_zeq(numdens, xp, T, asize, Ntot, grain_type, Qabs, G0=1.0):
    """
    Compute the equilibrium charge.
    """
    import math
    import numpy as np

    zlow = get_Zmin(asize, grain_type)
    zup  = get_Zmax(asize, grain_type)

    zlow = max(-200, zlow)
    zup  = min(200, zup)

    gotZeq = False

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]

    ne = nH*xHp + nC*xCp

    while gotZeq == False:

        # Compute the currents at the lower charge.
        JPE  = Jrate_pe(asize, zlow, grain_type, Ntot, Qabs, G0=G0)
        JE   = Jrate   (zlow, ne, 1.0, T, asize,'electron', grain_type)

        JH   = Jrate   (zlow, nH, xHp, T, asize,'hydrogen', grain_type)
        JC   = Jrate   (zlow, nC, xCp, T, asize,'carbon',   grain_type)

        JION = JH + JC

        # Compare the currents, positive and negative.
        flow = JPE + JION - JE

        # Compute the currents at the upper end charge.
        JPE  = Jrate_pe(asize, zup, grain_type, Ntot, Qabs, G0=G0)

        JE   = Jrate   (zup, ne, 1.0, T, asize,'electron', grain_type)

        JH   = Jrate   (zup, nH, xHp, T, asize,'hydrogen', grain_type)
        JC   = Jrate   (zup, nC, xCp, T, asize,'carbon',   grain_type)

        JION = JH + JC

        # Compare the currents, positive and negative.
        fup = JPE + JION - JE

        # Positive currents should dominate at flow and negative at fup.
        # Therefore flow and fup should have different signs.
        if flow*fup > 0.:
            print("there is an problem with the minimum and maximum grain charges!!")
            print(" zlow = ", zlow, "zup = ", zup)
            print(" flow = ", flow, "fup = ", fup)

            print("Setting the minimum and maximum charge by hand, to come out of zeq!!!")
            zup  = 0.02
            zlow = 0.01
            gotZeq = True


        # take a charge right in the middle of zup and zlow.
        zmiddle = 0.5 * (zup + zlow)

        # Compute the currents at the upper end charge.
        JPE  = Jrate_pe(asize, zmiddle, grain_type, Ntot, Qabs, G0=G0)

        JE   = Jrate   (zmiddle, ne, 1.0, T, asize,'electron', grain_type)

        JH   = Jrate   (zmiddle, nH, xHp, T, asize,'hydrogen', grain_type)
        JC   = Jrate   (zmiddle, nC, xCp, T, asize,'carbon',   grain_type)

        JION = JH + JC

        fmiddle =  JPE + JION - JE

        # move the upper or lower charges depending on the sign of the current.
        if flow*fmiddle < 0:
            zup = zmiddle
            fup = fmiddle
        else:
            zlow = zmiddle
            flow = fmiddle

        if abs(zup-zlow) <= 0.01:
            zeq = zmiddle
            gotZeq = True
            #print("Equilibrium charge", zeq)

    del zlow, zup, gotZeq, nH, nC, xHp, xCp, ne, JPE, JE, JH, JC, flow, JION, fup, zmiddle, fmiddle
    gc.collect(generation=2)

    return zeq


def get_zeq_vec(Jpe, Je, Jh, Jc, ZZ, asize, grain_type):
    """
    Compute the equilibrium charge.
    """
    import math
    import numpy as np

    zlow = get_Zmin(asize, grain_type)
    zup  = get_Zmax(asize, grain_type)

    gotZeq = False

    while gotZeq == False:

        ilow = np.argwhere(ZZ == np.floor(zlow))[0][0]  # find the index of zhere.
        #print("!!!!!!! zlow = ", np.floor(zlow), " ilow = ", ilow)

        # Compare the currents, positive and negative.
        flow = Jpe[ilow] + Jh[ilow] + Jc[ilow] - Je[ilow]

        iup = np.argwhere(ZZ == np.floor(zup))[0][0]  # find the index of zhere.
        #print("!!!!!!! zup = ", np.floor(zup), " iup = ", iup)

        # Compare the currents, positive and negative.
        fup = Jpe[iup] + Jh[iup] + Jc[iup] - Je[iup]

        # Positive currents should dominate at flow and negative at fup.
        # Therefore flow and fup should have different signs.
        if flow*fup > 0.:
            print("there is an problem with the minimum and maximum grain charges!!")
            print(" zlow = ", zlow, "zup = ", zup)
            print(" flow = ", flow, "fup = ", fup)

            print("Setting the minimum and maximum charge by hand, to come out of zeq!!!")
            zup  = 0.02
            zlow = 0.01
            gotZeq = True


        # take a charge right in the middle of zup and zlow.
        zmiddle = 0.5 * (zup + zlow)

        imiddle = np.argwhere(ZZ == np.floor(zmiddle))[0][0]  # find the index of zhere.
        # Compare the currents, positive and negative.
        fmiddle = Jpe[imiddle] + Jh[imiddle] + Jc[imiddle] - Je[imiddle]

        # move the upper or lower charges depending on the sign of the current.
        if flow*fmiddle < 0:
            zup = zmiddle
            fup = fmiddle
        else:
            zlow = zmiddle
            flow = fmiddle

        if abs(zup-zlow) <= 1.0:
            zeq = zmiddle
            gotZeq = True
            #print("Equilibrium charge", zeq)

    #del zlow, zup, gotZeq, nH, nC, xHp, xCp, ne, JPE, JE, JH, JC, flow, JION, fup, zmiddle, fmiddle
    #gc.collect(generation=2)

    #print("\n------------------------- \n zeq = %f \n------------------------\n"%zeq)

    return zeq


def compute_currents(numdens, xp, xH2, T, zeta, asize, Ntot, grain_type, Qabs, mu=1.0, zmin='default', zmax='default', G0=1.0, includeCR=False):
    """
    Compute the currents from photoelectric ejection of electrons, and sticking collisions from ions and electrons.
    Return arrays of Jpe, Jion, Je for all charges between zmin and zmax.
    """

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]


    if isinstance(nH, list):
        #print("I'm running the list")
        for i in range(len(nH)):
            # Electrons coming from ionization of Carbon and Hydroen by ISRF.
            ne  = nH[i]*xHp[i] + nC[i]*xCp[i]
            # Secondary electrons from CR.
            nH2  = nH[i]*xH2[i]
            if nH > 1.0e3:
                xeCR = CR_xe(nH2, zeta=zeta[i])
                neCR = nH[i]*xeCR
                # Take the largest electron density of the two.
                ne = max(ne, neCR)
    else:
        #print("I'm running the individual parameters")
        ne  = nH*xHp + nC*xCp
        nH2  = nH*xH2
        #print("H2 number density", nH2)
        if nH > 1.0e3:
            xeCR = CR_xe(nH2, zeta=zeta)
            neCR = nH*xeCR
            #print("Cosmic ray electrons = ", neCR)
            # Take the largest electron density of the two.
            ne = max(ne, neCR)

    if includeCR == False:
        ne  = nH*xHp + nC*xCp

    #print("I'm about to compute the current of electrons")
    #print("Number density H       nH  = %.1f \t cm-3" %nH)
    #print("Number density H2      nH2 = %.1f \t cm-3" %nH2)
    #print("Electron Fraction H    xHp = %.2g "%xHp)
    #print("Electron Fraction C   xCp  = %.2g "%xCp)
    #print("Ambient temperature     T  = %.2f \t K" %T)
    #print("Cosmic ray flux      zeta  = %.2g \t s-1"%zeta)
    #print("Electorn density = %.4f"%ne)
    #print("CR Electorns  = %.4f"%neCR)

    if zmin == 'default':
        zmin = int(get_Zmin(asize, grain_type))
    elif zmin > 0:
        zmin = 0

    if zmax == 'default':
        zmax = int(get_Zmax(asize, grain_type))
    elif zmax < 0:
        zmax = 0

    zitts = int(abs(zmax) + abs(zmin)+1)

    JPE    = np.zeros(zitts)
    JE     = np.zeros(zitts)
    JH     = np.zeros(zitts)
    JC     = np.zeros(zitts)
    ZZ     = np.zeros(zitts)

    for ii in range(zitts):
        zhere = zmin + ii
        JPE[ii]  = Jrate_pe(asize, zhere, grain_type, Ntot, Qabs, G0=G0)
        JE[ii]   = Jrate   (zhere, ne, 1.0, T, asize,'electron', grain_type)
        JH[ii]   = Jrate   (zhere, nH, xHp, T, asize,'hydrogen', grain_type)
        JC[ii]   = Jrate   (zhere, nC, xCp, T, asize,'carbon',   grain_type)
        ZZ[ii]   = zhere

    return JPE, JE, JH, JC, ZZ

def get_f2shield(x):
    f2s = 0.965 / (1. + x / 5e14)**2 + 0.35 / np.sqrt(1. + x / 5e14)*np.exp(-8.5e-4*np.sqrt(1.+x / 5e14))
    return f2s

def compute_CR_currents(numdens, zeta, asize, grain_type, Qabs, zmin="default", zmax="default"):
    """
    Compute the charging rates by CR induced effects.
    This one computes the CR-induced photoelectric charging.
    Collisions with electrons is already taken into account in the normal currents.
    """

    if zmin == 'default':
        zmin = int(get_Zmin(asize, grain_type))
    elif zmin > 0:
        zmin = 0

    if zmax == 'default':
        zmax = int(get_Zmax(asize, grain_type))
    elif zmax < 0:
        zmax = 0

    zitts = int(abs(zmax) + abs(zmin)+1)

    JCRe    = np.zeros(zitts)
    JCRp    = np.zeros(zitts)
    JCRpe   = np.zeros(zitts)
    ZZ      = np.zeros(zitts)

    for ii in range(zitts):
        zhere = zmin + ii
        JCRe[ii]    = Jrate_CR   (numdens, asize, zhere, grain_type)
        JCRpe[ii]   = Jrate_pe_CR(zeta, asize, zhere, grain_type, Qabs)
        ZZ[ii]      = zhere

    return JCRe, JCRpe, ZZ

def compute_fhere(numdens, xp, T, asize, Ntot, grain_type, Z, Qabs, zeta, G0=1.0):
    """
    Compute the distribution function for a given charge Z.
    """
    import math
    import numpy as np

    #print "Entering compute fhere function"
    #print "Going to calculate f(%i)"%Z

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]

    ne = nH*xHp + nC*xCp

    fz0 = 1.0

    if Z == 0 :
        fz = fz0
    if Z > 0:
        fz = fz0
        #if Z > zmax - 1:
        #    print("Computing fhere. Warning looking for f(Z) for Z > zmax.")
        #    fz = 1.0e-200
        #else:
        for itt in range(int(math.floor(Z))):
            zprime = itt + 1
            JPE  = Jrate_pe(asize, zprime-1, grain_type, Ntot, Qabs, G0=G0)
            #JION = Jrate(zprime-1, n, xe, T, mu*mp, asize,'ion',      grain_type)
            JE   = Jrate   (zprime, ne, 1.0, T, asize,'electron', grain_type)

            JH   = Jrate   (zprime-1, nH, xHp, T, asize,'hydrogen', grain_type)
            JC   = Jrate   (zprime-1, nC, xCp, T, asize,'carbon',   grain_type)
            #JMg  = Jrate   (zprime-1, n, xe, T, asize,'magnesium',grain_type)
            #JSi  = Jrate   (zprime-1, n, xe, T, asize,'silicon',  grain_type)
            #JS   = Jrate   (zprime-1, n, xe, T, asize,'sulfur',   grain_type)

            JPE_CR = Jrate_pe_CR(zeta, asize, zprime-1, grain_type, Qabs)

            #JION = JH + JC + JMg + JSi + JS
            JION = JH + JC

            if(JE == 0.0):
                print ("!! Warning: dividing by 0 while computing f_here !! (Z>0)")
            if((JPE + JION) == 0.0):
                print ("!! Warning: Jpe + Jion = 0 while computing f_here (Z>0)")

            fhere = (JPE + JION + JPE_CR) / JE
            fz = fz*fhere

    elif Z < 0:
        fz = fz0
        for itt in range(int(abs(math.floor(Z)))):
            zprime = Z + itt
            #print("running negative itteration, Zprime=%i"%zprime)

            JPE  = Jrate_pe(asize, zprime, grain_type, Ntot, Qabs, G0=G0)

            JE   = Jrate   (zprime, ne, 1.0, T, asize,'electron', grain_type)

            JH   = Jrate   (zprime, nH, xHp, T, asize,'hydrogen', grain_type)
            JC   = Jrate   (zprime, nC, xCp, T, asize,'carbon',   grain_type)

            JPE_CR = Jrate_pe_CR(zeta, asize, zprime, grain_type, Qabs)

            #JION = Jrate(zprime,   n, xe, T, mu*mp,  asize,'ion',      grain_type)
            #JE   = Jrate   (zprime+1, n, xe, T, asize,'electron', grain_type)

            #JH   = Jrate   (zprime, n, xe, T, asize,'hydrogen', grain_type)
            #JC   = Jrate   (zprime, n, xe, T, asize,'carbon',   grain_type)
            #JMg  = Jrate   (zprime, n, xe, T, asize,'magnesium',grain_type)
            #JSi  = Jrate   (zprime, n, xe, T, asize,'silicon',  grain_type)
            #JS   = Jrate   (zprime, n, xe, T, asize,'sulfur',   grain_type)

            #JION = JH + JC + JMg + JSi + JS
            JION = JH + JC

            if(JE == 0.0):
                print ("!! Warning: Jpe + Jion = 0 while computing f_here (Z<0)")
            if((JPE + JION) == 0.0):
                print ("!! Warning: dividing by 0 while computing f_here !! (Z<0)")

            fhere = JE / (JPE + JION + JPE_CR)
            fz = fz*fhere

    if Z!=0:
        del zprime, JPE, JE, JH, JC, JION, fhere
    del nH, nC, xHp, xCp, ne, fz0
    #gc.collect(generation=2)

    return fz


def fhere_vec(Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZ, zhere, includeCR=False):
    """
    Compute the distribution function given the currents.
    """
    import math
    import numpy as np

    #print "Entering compute fhere function"
    #print "Going to calculate f(%i)"%Z

    fz0 = 1.0

    #print("Charge here =", zhere)
    #print("charge array:", ZZ)

    if zhere ==0 or zhere > np.max(ZZ) or zhere < np.min(ZZ):
        fz = fz0
    elif zhere > 0 and zhere <= np.max(ZZ):
        i00   = np.argwhere(ZZ == 0)[0][0]      # find the index of ZZ == 0
        ihere = np.argwhere(ZZ == zhere)[0][0]  # find the index of zhere.

        nbins = ihere - i00
        #print("Number of bins = ", nbins)
        Jup    = np.zeros(nbins)
        Jdown  = np.zeros(nbins)

        Jup[:]   = Jpe[i00:ihere-1+1]+Jh[i00:ihere-1+1]+Jc[i00:ihere-1+1]
        Jdown[:] = Je[i00+1:ihere+1]

        if includeCR == True:
            Jup[:]   += JCRpe[i00:ihere-1+1]
            Jdown[:] += JCRe[i00+1:ihere+1]

        # Is there anywhere in Jdown == 0
        if(0.0 in Jdown):
            ihere = np.argwhere(Jdown==0)
            print ("!! Warning: Prod Je = 0. Dividing by 0 while computing fhere_vec (Z>0)")
            print ("zhere = %i, Jdown=0 at index="%(zhere, ihere))
            #exit()
        if(0.0 in Jup):
            print ("!! Warning: Prod Jpe + Jion = 0 while computing f_here (Z>0)")
            ihere = np.argwhere(Jup==0)
            print ("zhere = %i, Jup=0 at index="%(zhere, ihere))

        Jratio = np.prod(Jup / Jdown)
        fz = fz0 * Jratio

    elif zhere < 0 and zhere >= np.min(ZZ):
        i00   = np.argwhere(ZZ == 0)[0][0]      # find the index of ZZ == 0
        ihere = np.argwhere(ZZ == zhere)[0][0]   # find the index of zhere.

        #print("ihere and i00", ihere, i00)
        nbins = abs(ihere - i00)
        #print("Number of bins = ", nbins)
        Jup    = np.zeros(nbins)
        Jdown  = np.zeros(nbins)

        Jup[:]   = Jpe[ihere:i00-1+1]+Jh[ihere:i00-1+1]+Jc[ihere:i00-1+1]
        Jdown[:] = Je[ihere+1:i00+1]

        if includeCR == True:
            Jup[:]   += JCRpe[ihere:i00-1+1]
            Jdown[:] += JCRe[ihere+1:i00+1]


        if(0.0 in Jup):
            print ("!! Warning: Prod Jpe + Jion = 0. Dividing by 0 while computing fhere_vecv (Z<0)")
            ihere = np.argwhere(Jup==0)
            print ("zhere = %i, Jup=0 at index="%(zhere, ihere))

        if(0.0 in Jdown):
            print ("!! Warning: prod Je = 0 while computing f_here (Z<0)")
            ihere = np.argwhere(Jdown==0)
            print ("zhere = %i, Jdown=0 at index="%(zhere, ihere))

        Jratio = np.prod(Jdown / Jup)
        fz = fz0 * Jratio
        #fz = fz0 * Jdown/Jup

    return fz


def get_new_zmin_zmax(numdens, xp, T, asize, Ntot, grain_type, Qabs, zeta, G0=1.0, zeq=False, includeCR=False):
    """
    Something Zmin and Zmax are crazy large. And computing the charge distribution function has to loop over the (|zmin| + |zmax|)^2.
    I need a way to speed things up.
    """
    import math
    import numpy as np

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]

    ne = nH*xHp + nC*xCp

    # get the equilibrium charge for this case.
    if zeq == False:
        zeq = get_zeq([nH, nC], [xHp, xCp], T, asize, Ntot, grain_type, Qabs, G0=G0)

    #print("Equilibrium charge <Z> = %.1f" %zeq)
    zeq = int(math.floor(zeq))

    # What is the charge probability distribution for the equilibrium charge.
    f_mean = compute_fhere([nH, nC], [xHp, xCp], T, asize, Ntot, grain_type, zeq, Qabs, zeta, G0=G0)

    # Compute the old minimum and maximum charges possible using equations 22 and 24 from WD01.
    zmin_old = int(get_Zmin(asize, grain_type))
    zmax_old = int(get_Zmax(asize, grain_type))

    zmin_old = max(-200, zmin_old)
    zmax_old = min(400, zmax_old)

    #print(zmin_old, zmax_old)

    # if the maximum charge is very large, this will slow down the calculation of the charge distribution function.
    # systematically look at higher charges, when the charge probability has decreased by 10^8.
    # Go up by 5 charges.
    ff   = 1.0
    zpp  = 5
    zmax_test = zeq + zpp

    if zmax_old > zmax_test:

        counter = 0

        while ff > 1.0e-3:
        #while zpp <= 5:
            f_up        = compute_fhere([nH, nC], [xHp, xCp], T, asize, Ntot, grain_type, zeq+zpp, Qabs, zeta, G0=G0)
            ff          = f_up / f_mean
            zmax_test   = zeq + zpp
            zpp        += 5
            counter    +=1
            if counter%50 == 0 :
                print("Calculating the new MAXIMUM charge is taking a bit long. Currently at counter %i"%counter)
                #print("Current testing charge %i"%(zeq+zpp))
                #print("f_mean        = %.4g"%f_mean)
                #print("f_up          = %.4g"%f_up)
                print("f_up / f_mean = %.4g"%ff)
                if counter >= 100:
                    zmax = zmax_test
                    break
                    ff = 1.0e-99

        #  Set the maximum charge to the new test maximuym charge where the charge probability is lower than the
        # equilibrium charge probability by 10^8.
        zmax = zmax_test
    else:
        # Otherwise leave the maximum charge to the one given by equation 22 in WD 01.
        zmax = zmax_old


    # Do the same for the negative charges.
    ff   = 1.0
    zmm  = -5
    zmin_test = zeq + zmm

    # If the minimum charge given by equation 24 WD01
    if zmin_old < zmin_test:

        counter = 0

        while ff > 1.0e-3:
            f_down    = compute_fhere([nH, nC], [xHp, xCp], T, asize, Ntot, grain_type, zeq + zmm, Qabs, zeta, G0=G0)
            ff        = f_down / f_mean
            zmm      += -1
            zmin_test = zeq + zmm
            counter  +=1
            if counter%50 == 0 :
                print("Calculating the new MINIMUM charge is taking a bit long. Currently at counter %i"%counter)
                #print("Current testing charge %i"%(zeq+zmm))
                #print("f_mean", f_mean)
                #print("f_down", f_down)
                print("ff    ", ff    )
                #print("f_mean        = %.4g"%f_mean)
                #print("f_down        = %.4g"%f_down)
                #print("f_down/f_mean = %.4g"%ff)
                if counter >= 100:
                    zmin = zmin_test
                    break
                    ff = 1.0e-99

        zmin = zmin_test
    else:
        zmin = zmin_old

    #print("New minimum and maximum charge to compute the distribution:")
    #print("Zmin = %i,  Zmax = %i"%(zmin, zmax))
    #if zmin_old < zmin_test:
    #    del counter, f_down, ff, zmm, zmin_test
    #del nH, nC, xHp, xCp, ne, zeq, f_mean, zmin_old, zmax_old, zpp, zmax_test, f_up

    #gc.collect(generation=2)

    #if zmin > 0:
    #    zmin = 0
    #if zmax < 0:
    #    zmax = 0

    # This should not happen, but it's happening!!
    if zmax > zmax_old: zmax=zmax_old
    if zmin < zmin_old: zmin=zmin_old

    if includeCR:
        zmax +=10
        zmax = min(zmax, zmax_old)


    return zmin, zmax


def new_zmin_zmax_vec(Jpe, Je, Jh, Jc, ZZ, zeq):
    """
    Given the currents, compute the new zmin and zmax to constrain the charge distribution function.
    """

    zeq = int(np.floor(zeq))

    # What is the charge probability distribution for the equilibrium charge.
    f_mean = fhere_vec(Jpe, Je, Jh, Jc, ZZ, zeq)

    # Compute the old minimum and maximum charges possible using equations 22 and 24 from WD01.
    zmin_old = ZZ[0]
    zmax_old = ZZ[-1]

    # if the maximum charge is very large, this will slow down the calculation of the charge distribution function.
    # systematically look at higher charges, when the charge probability has decreased by 10^8.
    # Go up by 5 charges.
    ff   = 1.0
    zpp  = 5
    zmax_test = zeq + zpp

    counter = 0
    while zmax_old > zmax_test or ff > 1.0e-3:
        f_up        = fhere_vec(Jpe, Je, Jh, Jc, ZZ, zmax_test)
        ff          = f_up / f_mean
        zmax_test  += zpp
        counter    +=1
        if counter%50 == 0 :
            print("Calculating the new MAXIMUM charge is taking a bit long. Currently at counter %i"%counter)
            #print("Current testing charge %i"%(zeq+zpp))
            #print("f_mean        = %.4g"%f_mean)
            #print("f_up          = %.4g"%f_up)
            print("f_up / f_mean = %.4g"%ff)
            if counter >= 100:
                zmax = zmax_old
                break
                ff = 1.0e-99

    if zmax_old < zmax_test:
        # Otherwise leave the maximum charge to the one given by equation 22 in WD 01.
        zmax = zmax_old


    # Do the same for the negative charges.
    ff   = 1.0
    zmm  = -5
    zmin_test = zeq + zmm

    # If the minimum charge given by equation 24 WD01
    counter = 0
    while zmin_old < zmin_test or ff > 1.0e-3:
        f_down     = fhere_vec(Jpe, Je, Jh, Jc, ZZ, zmin_test)
        ff         = f_down / f_mean
        zmin_test += zmm
        counter   +=1

        if counter%50 == 0 :
            print("Calculating the new MINIMUM charge is taking a bit long. Currently at counter %i"%counter)
            #print("Current testing charge %i"%(zeq+zmm))
            #print("f_mean", f_mean)
            #print("f_down", f_down)
            print("ff    ", ff    )
            #print("f_mean        = %.4g"%f_mean)
            #print("f_down        = %.4g"%f_down)
            #print("f_down/f_mean = %.4g"%ff)
            if counter >= 100:
                zmin = zmin_test
                break
                ff = 1.0e-99

    if zmin_old > zmin_test:
        # Otherwise leave the maximum charge to the one given by equation 22 in WD 01.
        zmin = zmin_old

    #Make new arrays.
    nbins = int(abs(zmax)+abs(zmin)+1)
    Jpe_new, Je_new, Jh_new, Jc_new, ZZ_new = np.zeros(nbins), np.zeros(nbins), np.zeros(nbins), np.zeros(nbins), np.zeros(nbins)

    ilow = np.argwhere(ZZ == zmin)[0][0]
    iup  = np.argwhere(ZZ == zmax)[0][0]

    Jpe_new = Jpe[ilow:iup+1]
    Je_new  = Je[ilow:iup+1]
    Jh_new  = Jh[ilow:iup+1]
    Jc_new  = Jc[ilow:iup+1]
    ZZ_new  = ZZ[ilow:iup+1]

    return Jpe_new, Je_new, Jh_new, Jc_new, ZZ_new, zmin, zmax

def compute_fz_speed(numdens, xp, T, asize, Ntot, grain_type, mu=1.0, zmin='default', zmax='default', G0=1.0):
    """
    Compute the distribution function of grain charges given the local density,
    temperature, dust size, electron fraction and grain type.
    """
    import numpy as np

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]

    ne = nH*xHp + nC*xCp

    if zmin == 'default':
        zmin = int(get_Zmin(asize, grain_type))
    if zmax == 'default':
        zmax = int(get_Zmax(asize, grain_type))

    #print("Computing the charge distribution function.")
    #print("Between Zmin = %i and Zmax = %i"%(zmin, zmax))

    f0 = 0

    zitts = int(abs(zmax) + abs(zmin)+1)

    fz     = np.zeros(zitts)
    charge = np.zeros_like(fz)

    for i in range(zitts):
        zhere     = zmin + i
        charge[i] = zhere
        fhere     = compute_fhere([nH, nC], [xHp, xCp], T, asize, Ntot, grain_type, zhere, G0=G0)
        fz[i]     = fhere
        f0        = f0 + fhere

    fz = fz/f0

    del nH, nC, xHp, xCp, ne, zmin, zmax, f0, zitts, zhere, fhere
    gc.collect(generation=2)

    return fz, charge

def vector_fz(Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZ, zmin, zmax, includeCR=False):
    """
    Compute the distribution function of grain charges given the currents.
    """
    import numpy as np

    #zitts = len(ZZ)
    zitts = int(zmax - zmin + 1 )
    fz     = np.zeros(zitts)
    newZZ  = np.zeros(zitts)

    # I could do the loop here over the reduced version of zmin and zmax.
    # But have the currents and charge array for longer values.

    for i in range(zitts):
        #zhere     = ZZ[i]
        zhere     = zmin + i
        fhere     = fhere_vec(Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZ, zhere, includeCR=includeCR)
        fz[i]     = fhere
        newZZ[i]  = zhere
        #f0        = f0 + fhere

    f0 = np.sum(fz)
    fz = fz/f0

    return fz, newZZ

def get_charge_centroid(charge_dist, charge):
    """
    Given an array of charges and the charge distribution, compute the steady state charge centroid.
    """
    import numpy as np

    Zavg = np.average(charge, weights=charge_dist)

    return Zavg

def weighted_avg_and_std(values, weights):
    """
    Return the weighted average and standard deviation.

    values, weights -- Numpy ndarrays with the same shape.
    """
    import numpy as np
    import math

    average  = np.average(values, weights=weights)
    variance = np.average((values-average)**2, weights=weights)  # Fast and numerically precise

    return (average, math.sqrt(variance))

def get_Zmode(charges, charge_dist):
    """
    returns the mode of the charge distribution
    """
    import numpy as np

    index = np.argmax(charge_dist)
    Zmode = charges[index]

    return Zmode

def get_tauz(asize, grain_type, numdens, xp, T, Ntot, charge, charge_dist, G0=1.68):
    """
    compute the timescale of charge fluctuations.
    """

    nH = numdens[0]
    nC = numdens[1]

    xHp = xp[0]
    xCp = xp[1]

    ne = nH*xHp + nC*xCp

    avg, std = weighted_avg_and_std(charge, charge_dist)

    fz_Jtot_sum = 0

    for ii in range(len(charge)):
        zhere = charge[ii]
        JPE  = Jrate_pe(asize, zhere, grain_type, Ntot, G0=G0)
        JE   = Jrate   (zhere, ne, 1.0, T, asize,'electron', grain_type)
        JH   = Jrate   (zhere, nH, xHp, T, asize,'hydrogen', grain_type)
        JC   = Jrate   (zhere, nC, xCp, T, asize,'carbon',   grain_type)
        JTOT = JPE + JE + JH + JC

        fz_Jtot_sum += charge_dist[ii]*JTOT

    tauz = std**2 / fz_Jtot_sum

    del nH, nC, xHp, xCp, avg, std, zhere, JPE, JE, JH, JC, JTOT, fz_Jtot_sum
    gc.collect(generation=2)

    return tauz

def tauz_vec(Jpe, Je, Jh, Jc, ZZ, ffz, zmin, zmax):
#asize, grain_type, numdens, xp, T, Ndust, charge, charge_dist):
    """
    compute the timescale of charge fluctuations.
    """

    ilow = np.argwhere(ZZ == np.floor(zmin))[0][0]  # find the index of zhere.
    iup  = np.argwhere(ZZ == np.floor(zmax))[0][0]  # find the index of zhere.

    avg, std = weighted_avg_and_std(ZZ[ilow:iup+1], ffz)

    fz_Jtot = 0
    fz_Jtot = np.sum(ffz*(Jpe[ilow:iup+1]+Je[ilow:iup+1]+Jh[ilow:iup+1]+Jc[ilow:iup+1]))

    tauz = std**2 / fz_Jtot

    return tauz

def get_QabsTable(grain_type, grain_size, num_sizes=21, dirtables='default'):
    """
    Given tha grain type, and the number of dust grains in the table (21 or 81).
    Read that tabulated data of Absorption efficiency as a function of size and photon energy.
    Return the interpolated Qabs table, for the given grain size. (In development.)
    """

    from scipy.interpolate import interp2d
    import numpy as np
    import sys

    if grain_type == "carbonaceous":
        filename = "Gra_%.2i"%num_sizes
    else:
        filename = "Sil_%.2i"%num_sizes

    # Array of sizes available in the file.
    sizelist = np.logspace(1, 5, num_sizes)
    index_up = False

    # Get the index of the size in the array.
    for i,v in enumerate(sizelist):
        if v >= grain_size:
            index_up = i
            break

    if index_up == 0:
        index_up +=1

    # If I am asking for a dust grain outside the tabulated data.
    if grain_size > 1.0e5:
        # Do something
        print("I'm looking for a grain size larger than 10 microns. Outside the tabulated data!!!")
        print("Extrapolating the absorption efficiencies linearly.  !!!! Dangerous territory !!!!")

    if grain_size < 1.0e1:
        # Do something else.
        print("I'm looking for a grain size smaller than 10 Angstrom. Outside the tabulated data!!!")
        print("Using the absorption efficiency for a 10 AA grain")

    index_low = index_up - 1
    # Now, get the absorption efficiency for the grains smaller and larger than the one I'm interested in.
    asize_low, nu,  Qabs_low = Read_oneBlock(filename, index_low, dirtables)
    asize_up,  nu,  Qabs_up  = Read_oneBlock(filename, index_up, dirtables)

    #print("Minimum and maximum grain size in the table.")
    #print(asize_low, asize_up)

    # Make large arrays to pass to the interpolation
    Asize = np.array([asize_low, asize_up])*1.0e4
    Qabs  = np.transpose([Qabs_low, Qabs_up])

    # 2D interpolation of the array.
    # Kind of overkill here, but whatever.
    f2d = interp2d(Asize, nu, Qabs, bounds_error=False)

    Qabsnew = np.array(f2d(grain_size, nu))
    Qabsnew = np.ndarray.flatten(Qabsnew)

    Qabsnew = np.flipud(Qabsnew)

    return Qabsnew


def Read_oneBlock(filename, index_size, dirtables='default'):
    """
    Given the filename and the index of the grain size. Read the block of data of absorption efficiencies
    as a function of photon energy, for the given size.
    """
    import numpy as np

    if dirtables == "default":
        dirtables = "/home/jcibanezm/codes/DustAnalysis/Charge/Tables"

    filename = "%s/%s"%(dirtables, filename)
    f = open(filename, "r")

    offset = 268 + 9748*index_size

    f.seek(offset)

    line0  = f.readline().split()
    asize  = float(line0[0])

    f.readline()

    nu   = np.zeros(241)
    Qabs = np.zeros_like(nu)

    for i in range(241):
        data    = f.readline().split()
        nu  [i] = float(data[0])
        Qabs[i] = float(data[1])

    f.close()

    return asize, nu, Qabs

def get_Qabs_unu(nu, Qabs, G0):
    """
    Get Qabs * u_nu. For the average absorption efficiency calculation.
    Given:

    frequency nu
    Absorption efficiency table Qabs
    Scaling of the radiation field, e.g. 1.68 for Draine field.
    """
    from scipy import integrate
    from scipy.interpolate import interp1d

    u_nu = ISRF_nu(nu, 0.0, G0=G0)

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    return Qabs_nu*u_nu

def get_Qabsnu(nu, Qabs):
    """
    Get Qabs For the average absorption efficiency calculation.
    Given:

    frequency nu
    Absorption efficiency table Qabs
    """
    from scipy import integrate
    from scipy.interpolate import interp1d

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    return Qabs_nu

def get_avgQabs(Qabs, G0):
    """
    calculate the averaged value of Qabs*u_nu.
    """
    from scipy import integrate

    nu_low = 6.0  / hplanck
    nu_up  = 13.6 / hplanck

    Quint = integrate.quad(get_Qabs_unu, nu_low, nu_up, args=(Qabs, G0))[0]
    uint  = integrate.quad(ISRF_nu, nu_low, nu_up, args=(0, G0))[0]

    return Quint / uint

def get_YieldQabs_nu(hnu, Qabs, asize, Z, grain_type):
    """
    get the value of Qabs(nu, a, gr_type)*Yield(nu, a, Z, gr_type)
    """
    from scipy import integrate
    from scipy.interpolate import interp1d

    nu = hnu / hplanck

    # This is the range of wavelengths in microns.  I get photon energies in frequency.
    lambda_array = np.logspace(3, -3, 241)
    f1d          = interp1d(lambda_array, Qabs, fill_value="extrapolate")

    # What is the corresponding wavelength of the photon in microns.
    lambda_here = clight / nu * cmtomicron
    Qabs_nu     = f1d(lambda_here)

    Yield_nu = get_Yield(hnu, asize, Z, grain_type)

    YieldQabs_nu = Qabs_nu*Yield_nu

    return YieldQabs_nu

def get_avgYieldQabs(Qabs, asize, Z, grain_type):
    from scipy import integrate

    hnu_low = 11.2
    hnu_up  = 13.6

    YieldQabs = integrate.quad(get_YieldQabs_nu, hnu_low, hnu_up, args=(Qabs, asize, Z, grain_type))[0]
    #print("Calculating average Yield*Qabs = ", YieldQabs)
    #Yield = integrate.quad(get_Yield, hnu_low, hnu_up, args=(asize, 0, 'carbonaceous'))[0]
    #YieldQabs = get_avgQabs(Qabs, G0=1.0)*Yield

    return YieldQabs

def get_zeta(Ntot, model="high"):
    """
    Get the local cosmic ray flux from the H2 column density and the CR flux models
    Polynomial fit in Appendix F in Padovani et al 2018.

    input:
        fH2shield = Flash value of H2 shielding factor.
        model     = Cosmic Ray model, 'low' or 'high' (Check Padovani et al 2018 appendix F for information about this.)

    """

    kL = [-3.331056497233e6, 1.207744586503e6, -1.913914106234e5, 1.731822350618e4, -9.790557206178e2, 3.543830893824e1, -8.034869454520e-1, 1.048808593086e-2, -6.188760100997e-5, 3.122820990797e-8]
    kH = [ 1.001098610761e7, -4.231294690194e6, 7.921914432011e5, -8.623677095423e4, 6.015889127529e3, -2.789238383353e2, 8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3, -9.937499546711e-6]

    #zeta = np.zeros_like(NH2)
    zeta = 0

    if model == "high":
        K = kH
    elif model == "low":
        K = kL
    else:
        print("Cosmic Ray ionization rate models available are 'high' and 'low'")

    for kk in range(10):
        zeta  += K[kk]*np.power(np.log10(Ntot), kk)

    zeta = np.power(10, zeta)

    if Ntot < 1.0e18:
        zeta = 0.0

    return zeta


def get_G_CR(Ntot, model="high"):
    """
    Compute the energy density of the cosmic ray induced ultraviolet radiation field
    in units of the Habing Field.

    input:
        Ntot      = Total column density
        model     = Cosmic Ray model, 'low' or 'high' (Check Padovani et al 2018 appendix F for information about this.)

    Returns:
        Cosmic Ray induced radiation field in units of Habing Field.
    """

    zeta = get_zeta(Ntot, model=model)

    FUV_H2 = np.zeros_like(zeta)

    omega   = 0.5       # dust albedo
    Rv      = 3.1       # Slope of the extinction curve
    NH2_mag = 1.87e21 # Typical dust to extinction ratio.(Bohlin, Savage & Drake 1978)

    FUV_H2 = 960. * (1. / (1.-omega))* (NH2_mag / 1.0e21) * (Rv/3.2)**1.5 * (zeta / 1.0e-17) * 12.4*eVtoerg / clight

    U_Hab = 5.33e-14

    G_CR = FUV_H2 / U_Hab

    return G_CR
