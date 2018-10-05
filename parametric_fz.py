import numpy as np
from scipy.interpolate import interp1d

eVtoerg       = 1.602e-12
ergtoeV       = 6.242e11


def get_zcent(Gtot, T, ne, grain_type, grain_size):
    """
    Compute the charge centroid given Gtot, T, ne, grain type and grain size.
    Equation XX and Tables YY and ZZ in Ibanez-Mejia et al 2018.
    """
    size_table = [3.5, 5.0, 10., 50., 100., 500., 1000.]

    # Select the table of parameters depending on the grain type.
    if grain_type == "silicate":
        k          = [0.0050, 0.0219, 0.0328, 0.1678, 0.8650, 3.2283, 4.6588]
        b          = [-0.0839, -0.2390, -0.3906, -0.1707, 0.0745, 0.9703, 1.6579]
        alpha      = [0.4318, 0.3573, 0.3857, 0.4081, 0.3210, 0.3390, 0.3635]
        h          = [71., 117., 110., 490., 572., 438., 486.]
    else:
        print("I have copied the table for silicates!!! Update and put the parameters for carbonaceous grains here!!!")
        k          = [0.0045, 0.0139, 0.0143, 0.1868, 2.1251, 5.7066, 5.9382]
        b          = [-0.0794, -0.2490, -0.4209, -0.2701, -0.0450, 0.6004, 1.2954]
        alpha      = [0.5270, 0.4642, 0.5257, 0.4610, 0.2937, 0.3419, 0.3988]
        h          = [95.3779, 155.3420, 125.4317, 839.5276, 1231.4187, 1136.3664, 1149.3435]

    # Interpolate the tables and find the values for the given size.
    k_int1d          = interp1d(size_table, k, fill_value="extrapolate")
    khere            = k_int1d(grain_size)

    b_int1d          = interp1d(size_table, b, fill_value="extrapolate")
    bhere            = b_int1d(grain_size)

    alpha_int1d      = interp1d(size_table, alpha, fill_value="extrapolate")
    alphahere        = alpha_int1d(grain_size)

    h_int1d          = interp1d(size_table, h, fill_value="extrapolate")
    hhere            = h_int1d(grain_size)

    GTn = Gtot*np.sqrt(T)/ne

    centroid = khere*(1.-np.exp(-GTn/hhere))*np.poe(GTn, alphahere) + bhere

    return centroid

def get_zwidth(grain_size, grain_type, zcent):
    """
    Return the width of the charge distribution given the charge centroid, grain type and size.
    """
    # Add the values of the tables.
    
    size_table = [3.5, 5.0, 10., 50., 100., 500., 1000.]

    if grain_type == "silicate":
        if zcent >= 0:
            c        = [0.4246, 0.3237, 0.4746, 1.1309, 1.7014, 3.7183,  5.3094]
            eta      = [0.2308, 0.3386, 0.7528, 1.6041, 2.6212, 12.9426, 26.6146]
        else:
            c        = [0.5864, 0.5383, 0.1553, 0.0323,  1.0e-99, 1.0e-99, 1.0e-99]
            eta      = [0.4415, 0.9440, 0.4803, 1.0e-99, 1.0e-99, 1.0e-99, 1.0e-99]

        d = [0.1669, 0.2961, 0.4085, 0.5086, 0.5385, 1.0217, 1.4119]
    else:
        print("I have copied the table for silicates!!! Update and put the parameters for carbonaceous grains here!!!")
        if zcent >= 0:
            c        = [0.3308, 0.3987, 0.6954, 1.7705, 2.5930, 5.7972,  8.2829]
            eta      = [0.2270, 0.5453, 1.0163, 2.4686, 4.1751, 19.5302, 40.5363]
        else:
            c        = [0.1531, 0.2801, 0.0107, 0.0260,  0.0260,  1.0e-99, 1.0e-99]
            eta      = [0.1642, 1.001, 1.0e-99, 1.0e-99, 1.0e-99, 1.0e-99, 1.0e-99]
        
        d = [0.2216, 0.3827, 0.4901, 0.5573, 0.5848, 1.0066, 1.3812]

    # Interpolate the tables and find the values for the given size.
    c_int1d          = interp1d(size_table, c, fill_value="extrapolate")
    chere            = c_int1d(grain_size)
    
    eta_int1d        = interp1d(size_table, eta, fill_value="extrapolate")
    etahere          = eta_int1d(grain_size)
    
    d_int1d          = interp1d(size_table, d, fill_value="extrapolate")
    dhere            = d_int1d(grain_size)

    width = chere * (1.0 - np.exp(-np.abs(zcent)/etahere)) + dhere
    return width

def get_fz(ntot=1.0, T=1.0, xe=1.0, Ntot=1.0, NH2=1.0, grain_type="silicate", grain_size=10, xH2="default", G0=1.7, correct_edens=False, CR_model="high"):
    """
    Compute the dust charge distribution as a function of the ISM parameters.
    Use the parametric equations in Ibañez-Mejia et al 2018.
    
    input:
        nH   = total hydrogen volume density, nHI + nHII + 2*nH2
        T    = temperature
        xe   = ionization fraction
        Ntot = total hydrogen column density, NHI + NHII + 2*NH2
        NH2  = molecula hydrogen column density.
        xH2  = molecular hydrogen fraction
        G    = Scaling of the background interstellar radiation field. Default set to 1.7
        
        grain_type = type of grain: silicate or carbonaceous
        grain_size = grain size, in Angstroms.
        
        correct_edens = boolean. Do I need to correct the electron density at volume densities above 1.0e3 cm-3? See Appendix XX in Ibanez-Mejia et al 2018. for a discussion.
        
        CR_model = cosmic ray protons spectrum. 'low' or 'high'. See Discussion in Ivlev et al 2015 and Padovanni et al 2018. Default 'high'.
    
    return:
        Charge array, PDF
    """

    # Get the effective strength of the radiation field.
    Av   = Ntot / 1.87e22
    Geff = G0 * np.exp(-2.5*Av)
    
    # Get the intensity of the H2 phosphoresence in units of the Habing Field.
    # Zeta - Use the equation by Padovani et al 2018 to get CR flux given Ntot.
    
    zeta  = get_zeta(NH2)
    
    omega = 0.5     # dust albedo
    Rv    = 3.1     # Slope of the extinction curve
    NH2_mag = 1.8e21# Typical dust to extinction ratio.
    
    # FUV in units of eV s-1 cm-3
    FUV_CR = 960. * (1. / (1.-omega))* (NH2_mag / 1.0e21) * (Rv/3.2)**1.5 * (zeta / 1.0e-17) * 12.0

    # energy density of a Habing field.
    u_Hab = 5.33e-14 * ergtoeV

    # convert the CR-induced radiation field energy to Habing field units
    G_CR   = FUV_CR / u_Hab

    Gtot = Geff + G_CR
    
    # Compute the electron density.
    # Ask wether I have a chemical network that gives me the information of xe, or I want to use the tabulated one.
    if correct_edens == True:
        if xH2 == "default": print("You need to give the molecular hydrogen fraction!")
        ne, xe = compute_new_xe(ntot, xe, xH2, zeta)
    else:
        ne = ntot*xe

    zcent = get_zcent(Gtot, T, ne, grain_type, grain_size)
    
    zmin = int(zcent - 3*zwidth)
    zmax = int(zcent + 3*zwidth)
    
    ZZ = np.araynge(zmin, zmax+1)
    
    # Assume a Gaussian distribution for the shape of the charge distribution.
    ffz = np.zeros_like(ZZ)
    ffz = 1.0 * np.exp(-(ZZ - zcent)*(ZZ - zcent)/(2*zwidth))
    ffz = ffz / np.sum(ffz) # Normalize the resulting distribution.
    
    return ZZ, ffz

# Add function to compute the cosmic ray flux.
def get_zeta(NH2, model="high"):
    """
        Get the local cosmic ray flux from the H2 column density and the CR flux models
        Polynomial fit in Appendix F in Padovani et al 2018.
        
        input:
        NH2 = H2 column density.
        model     = Cosmic Ray model, 'low' or 'high' (Check Padovani et al 2018 appendix F for information about this.)
        
        """
    
    # Set a ceiling for the CR ionization rate. Too small column densities result in crazy high zeta values.
    NH2 = max(NH2, 1.0e19)
    
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
        zeta  += K[kk]*np.power(np.log10(NH), kk)
    
    zeta = np.power(10, zeta)
    
    # For very low H2 column densities,
    #if NH2 < 1.0e19:
    #    zeta = 0.0

    return zeta

def compute_new_xe(numdens, xe, xH2, zeta):
    """
    Update the electron density and electron fraction for densities above 1000 cm-3.
    """
    ne   = numdens*xe
    
    #print("H2 number density", nH2)
    if nH > 1.0e3:
        nH2 = numdens*xH2
        # Equation from Caselli et al 2002 model 3.
        xeCR = 6.7e-6*(nH2)**(-0.56)*np.sqrt(zeta/1.0e-17)
        neCR = nH*xeCR
        ne = max(ne, neCR)
        xe = ne / numdens
    
    return ne, xe
