#include <iostream>

using namespace std;

double get_fz(){
 
  return ffz;
}


double copute_new_xe(double numdens, double xe, double xH2, double zeta){
    
    double ne, nH2, xeCR, neCR;
    
    ne   = numdens*xe;
    
    //print("H2 number density", nH2)
    if (numdens > 1.0e3){
        nH2 = numdens*xH2;
        // Equation from Caselli et al 2002 model 3.
        xeCR = 6.7e-6*(nH2)**(-0.56)*np.sqrt(zeta/1.0e-17);
        neCR = nH*xeCR;
        ne   = max(ne, neCR);
        xe   = ne / numdens;
    }

    return ne
}


double get_zeta(double NH2, bool model="high"){
    /*
    Get the local cosmic ray flux from the H2 column density and the CR flux models
    Polynomial fit in Appendix F in Padovani et al 2018.

    input:
        NH2 = H2 column density.
        model     = Cosmic Ray model, 'low' or 'high' (Check Padovani et al 2018 appendix F for information about this.)

    */
    
    double zeta;
    double kl[10], kH[10];
    

    // Forgot how to initialize an array in CPP
kL = [-3.331056497233e6, 1.207744586503e6, -1.913914106234e5, 1.731822350618e4, -9.790557206178e2, 3.543830893824e1, -8.034869454520e-1, 1.048808593086e-2, -6.188760100997e-5, 3.122820990797e-8]
kH = [ 1.001098610761e7, -4.231294690194e6, 7.921914432011e5, -8.623677095423e4, 6.015889127529e3, -2.789238383353e2, 8.595814402406e0, -1.698029737474e-1, 1.951179287567e-3, -9.937499546711e-6]

    double zeta = 0

    if (model == "high"){
        K = kH;
    else if(model == "low"){
        K = kL;
    }

    for (kk=0:kk+1:10){
        zeta  += K[kk]*np.power(np.log10(NH2), kk);
    }

    zeta = power(10., zeta);

    // For very low H2 column densities set a minimum CR ionization rate.
    if (NH2 < 1.0e19){
        zeta = 1.0e-17;
    }

    return zeta;
}
