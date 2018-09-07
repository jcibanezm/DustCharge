import matplotlib.pyplot as plt
import numpy as np
import compute_charge_dist as fz
from pynverse import inversefunc

temp       = np.array( [7000, 100, 10])
nH         = np.array( [0.9, 52, 1.0e5])
xe         = np.array( [0.012, 0.00018, 4.2e-10])
xH2        = np.array( [4.6e-5, 0.032, 0.9986])
Av         = np.array( [0.046, 0.079, 10.48])
fH2shield  = np.array( [0.11, 0.00057, 7.0e-8])
NH2        = np.array( [5.27e+15, 9.12e+19, 2.91e+22])

Ntot = Av * 1.87e21

# fH2shield is a value from Flash simulations related to the nH2 column density.
# I should use a normal NH2 input for the code release.

# Carbon abundance with respect to Hydrogen.
xC = 2.95e-4
nC = nH*xC

GG = [fz.get_G(Ntot[0], G0=1.7), fz.get_G(Ntot[1], G0=1.7), fz.get_G(Ntot[2], G0=1.7)]

print("---------------------------------------------------------------------------")
print(" Local environment properties:")
print("       WNM          CNM          CMM ")
print("nH:    %.1g          %.2g        %.4g"%(nH[0], nH[1], nH[2]))
print("T:     %.1g        %.1g        %.3g" %(temp[0], temp[1], temp[2]))
print("G:     %.2f         %.2f         %.2g"%(GG[0], GG[1], GG[2]))
print("xe:    %.2g        %.2g      %.1g" %(xe[0], xe[1], xe[2]))
print("xH2:   %.2g      %.2g        %.4g" %(xH2[0], xH2[1], xH2[2]))
print("Av:    %.2g        %.2g      %.1g" %(Av[0], Av[1], Av[2]))
print("Ntot:   %.2g        %.2g      %.1g" %(Ntot[0], Ntot[1], Ntot[2]))

grain_type = "carbonaceous"
grain_size = [5, 50, 100]
G0         = 1.7

# Dust grain absorption efficiency tables Qabs need to be downloaded from:
# Function fz.get_QabsTable(... dirtables=<directory where the tables are located>)
Qabs5    = fz.get_QabsTable(grain_type, 5)
Qabs50   = fz.get_QabsTable(grain_type, 50)
Qabs100  = fz.get_QabsTable(grain_type, 100)

Qabs = [Qabs5, Qabs50, Qabs100]

zmean, zmode, zstd, tauz  =  np.zeros(9), np.zeros(9), np.zeros(9), np.zeros(9)
zminmax = np.array(np.zeros(2*9))
fdistCR   = []

phases = ["Warm Neutral Medium", "Cold Neutral Medium", "Cold Molecular Medium"]

# loop over grain sizes
for kk in range(3):
    # loop over ISM phases.
    for ii in range(3):

        print("Running %s grain with size %i in phase %s"%(grain_type, grain_size[kk], phases[ii]))

        # Get the CR ionization rate given the Column density.
        # Update to the total column density instead of the weird Flash function.
        zeta = fz.get_zeta(NH2[ii])

        ############################################################################################
        # Run the charge distribution calculation!!!
        Jpe, Je, Jh, Jc, ZZall = fz.compute_currents ([nH[ii], nC[ii]], [xe[ii], 0.0], xH2[ii], temp[ii], zeta, grain_size[kk], Ntot[ii], grain_type, Qabs[kk], G0=G0)
        JCRe, JCRpe, ZZnew     = fz.compute_CR_currents(nH[ii], zeta, grain_size[kk], grain_type, Qabs[kk])
        zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size[kk], grain_type)
        new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH[ii], nC[ii]], [xe[ii], 0.0], temp[ii], grain_size[kk], Ntot[ii], grain_type, Qabs[kk], zeta, zeq=zeq, G0=G0)

        #new_zmax +=5

        ffzCR, ZZ              = fz.vector_fz        (Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZall, new_zmin, new_zmax, includeCR=True)

        # Charging timescale.
        tauz[ii]               = fz.get_tauz         (grain_size[kk], grain_type, [nH[ii], nC[ii]], [xe[ii], 0.0], temp[ii], Ntot[ii], ZZ, ffzCR, xH2[ii], zeta, Qabs[kk], G0=G0, includeCR=True)

        Zm        = fz.get_Zmode(ZZ, ffzCR)
        zmode[ii+kk*3] = Zm

        avg, std  = fz.weighted_avg_and_std(ZZ, ffzCR)
        zmean[ii+kk*3] = avg
        zstd[ii+kk*3]  = std

        zminmax[ii*2+  2*kk*3]  = new_zmin
        zminmax[ii*2+1+2*kk*3]  = new_zmax

        for jj in range(len(ffzCR)):
            fdistCR.append(ffzCR[jj])


grain_type = "silicate"
grain_size = [5, 50, 100]
G0         = 1.7

# Dust grain absorption efficiency tables Qabs need to be downloaded from:
# Function fz.get_QabsTable(... dirtables=<directory where the tables are located>)
Qabs5    = fz.get_QabsTable(grain_type, 5)
Qabs50   = fz.get_QabsTable(grain_type, 50)
Qabs100  = fz.get_QabsTable(grain_type, 100)

Qabs = [Qabs5, Qabs50, Qabs100]

zmean_sil, zmode_sil, zstd_sil  =  np.zeros(9), np.zeros(9), np.zeros(9)
zminmax_sil   = np.array(np.zeros(2*9))
fdistCR_sil   = []

tauz_sil = np.zeros_like(zmean_sil)

phases = ["Warm Neutral Medium", "Cold Neutral Medium", "Cold Molecular Medium"]

# loop over grain sizes
for kk in range(3):
    # loop over ISM phases.
    for ii in range(3):

        print("Running %s grain with size %i in phase %s"%(grain_type, grain_size[kk], phases[ii]))

        # Get the CR ionization rate given the Column density.
        # Update to the total column density instead of the weird Flash function.
        #zeta = fz.get_zeta(Ntot[ii])
        zeta = fz.get_zeta(NH2[ii])

        ############################################################################################
        # Run the charge distribution calculation!!!
        Jpe, Je, Jh, Jc, ZZall = fz.compute_currents ([nH[ii], nC[ii]], [xe[ii], 0.0], xH2[ii], temp[ii], zeta, grain_size[kk], Ntot[ii], grain_type, Qabs[kk], G0=G0)
        JCRe, JCRpe, ZZnew     = fz.compute_CR_currents(nH[ii], zeta, grain_size[kk], grain_type, Qabs[kk])
        zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size[kk], grain_type)
        new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH[ii], nC[ii]], [xe[ii], 0.0], temp[ii], grain_size[kk], Ntot[ii], grain_type, Qabs[kk], zeta, zeq=zeq, G0=G0)

        ffzCR, ZZ              = fz.vector_fz        (Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZall, new_zmin, new_zmax, includeCR=True)

        # Charging timescale.
        tauz_sil[ii]               = fz.get_tauz         (grain_size[kk], grain_type, [nH[ii], nC[ii]], [xe[ii], 0.0], temp[ii], Ntot[ii], ZZ, ffzCR, xH2[ii], zeta, Qabs[kk], G0=G0, includeCR=True)

        Zm                 = fz.get_Zmode(ZZ, ffzCR)
        zmode_sil[ii+kk*3] = Zm

        avg, std           = fz.weighted_avg_and_std(ZZ, ffzCR)
        zmean_sil[ii+kk*3] = avg
        zstd_sil[ii+kk*3]  = std

        zminmax_sil[ii*2+  2*kk*3]  = new_zmin
        zminmax_sil[ii*2+1+2*kk*3]  = new_zmax

        for jj in range(len(ffzCR)):
            fdistCR_sil.append(ffzCR[jj])


############################################################################################################
#                                                Plotting
############################################################################################################

xsize = 14
ysize = 11

nfigs_x = 3
nfigs_y = 3

fig = plt.figure(figsize=(xsize, ysize))

hpad = 0.05
wpad = 0.054

xs_panel = 0.83 / nfigs_x
ys_panel = 0.88 / nfigs_y

############################################################################################################
############################               5 Angstroms             #######################################
############################################################################################################

ax = fig.add_axes([0, 0, 1, 1])

#ax.plot([0,1], [0,1], visible=False)
ax.set_xlim(0,1)
ax.set_ylim(0,1)

ax.text(0.18, 0.955, "5 $\\AA$", fontsize=20, horizontalalignment='center')
ax.text(0.48, 0.955, "50 $\\AA$", fontsize=20, horizontalalignment='center')
ax.text(0.74, 0.955,"100 $\\AA$", fontsize=20, horizontalalignment='center')

ax.text(0.94, 0.15, "Warm\nNeutral\nMedium", fontsize=20, horizontalalignment='center')
ax.text(0.94, 0.45, "Cold\nNeutral\nMedium", fontsize=20, horizontalalignment='center')
ax.text(0.94, 0.75,"Cold\nMolecular\nMedium", fontsize=20, horizontalalignment='center')

ax.plot([0.855, 0.875], [0.978, 0.978], "-k", linewidth=2)
ax.text(0.88, 0.97, "Silicates", fontsize=16, horizontalalignment='left')

ax.plot([0.855, 0.875], [0.958, 0.958], "-r", linewidth=2)
ax.text(0.88, 0.95, "Carbonaceous", fontsize=16, horizontalalignment='left')

plt.axis('off')

#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 0
jj = 0

# Start the cumulative count.
cum     = 0
cum_sil = 0

########## 5AA WNM
##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

##############################
cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz, "-r", linewidth=2, drawstyle='steps', alpha=1.0)

# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil, "-k", linewidth=2, drawstyle='steps', alpha=1.0)

#carbon
ax.plot(ZZ+0.5, ffzCR, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.95*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.95*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

ax.text(0.95*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.95*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')


ax.set_ylim(0, 0.99)

ax.tick_params(
    axis='x',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on') # labels along the bottom edge are off

ax.set_ylabel("f(z)", fontsize=20)
ax.set_xlabel("Z", fontsize=16)

ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 0
jj = 1

########## 5AA CNM

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################


ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz, "-r", linewidth=2, drawstyle='steps', alpha=1.0)

# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR, "--r", linewidth=2, drawstyle='steps', alpha=1.0)

ax.text(0.95*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.95*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

ax.text(0.95*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.95*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')



#ax.set_xlim(np.min(Z_a_3_100)+0.1, np.max(Z_a_3_100)-0.1)
ax.set_ylim(0, 0.99)

ax.tick_params(axis='x', which='both', bottom='on', labelbottom='on')
ax.set_ylabel("f(z)", fontsize=20)
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 0
jj = 2

########## 5AA CMM

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################



ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR, "--r", linewidth=2, drawstyle='steps', alpha=1.0)

#ax.plot(ZZ+0.5, ffz, "-r", linewidth=2, drawstyle='steps', alpha=1.0)

ax.text(0.95*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.95*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

ax.text(0.95*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.95*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')

ax.set_ylim(0, 0.99)

ax.tick_params(axis='x', which='both', bottom='on', labelbottom='on')
ax.set_ylabel("f(z)", fontsize=18)
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------


############################################################################################################
############################                50 Angstroms            #######################################
############################################################################################################

#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 1
jj = 0

########## 100AA WNM

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz*5, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil*2, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR*2, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.85*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.85*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')
ax.text(0.85*charges + zmin, 0.6, "fz$\\times$2", fontsize=14, color='red')

ax.text(0.85*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.85*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.85*charges + zmin, 0.3, "fz$\\times$2", fontsize=14, )

#ax.text(0.7*charges + zmin, 0.6, "fz$\\times$5", fontsize=14, color='red')

#ax.set_xlim(np.min(Z_a_3_100)+0.1, np.max(Z_a_3_100)-0.1)
ax.set_ylim(0, 0.99)
ax.set_xlim(min(ZZ[0], ZZ_sil[0]), max(ZZ[-1], ZZ_sil[-1])+10)

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on', labelleft='off') # labels along the bottom edge are off


ax.set_xlabel("Z", fontsize=18)

ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 1
jj = 1

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz*5, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil*2, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR*2, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.7*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.7*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')
ax.text(0.7*charges + zmin, 0.6, "fz$\\times$2", fontsize=14, color='red')

ax.text(0.7*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.7*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')
ax.text(0.7*charges + zmin, 0.3, "fz$\\times$2", fontsize=14)

ax.set_ylim(0, 0.99)

ax.tick_params(axis='both', which='both', bottom='on', labelbottom='on', labelleft='off')
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 1
jj = 2

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(4, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(4, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

ax.text(4, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14, color='black')
ax.text(4, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14, color='black')


ax.set_ylim(0, 0.99)
ax.set_xlim(-2.5, 8)

ax.tick_params(axis='both', which='both', bottom='on', labelbottom='on', labelleft='off')
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


############################################################################################################
############################               100 Angstroms            #######################################
############################################################################################################

#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 2
jj = 0

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz*10, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil*2, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR*2, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.95*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.95*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')
ax.text(0.95*charges + zmin, 0.6, "fz$\\times$2", fontsize=14, color='red')

ax.text(0.95*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14)
ax.text(0.95*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14)
ax.text(0.95*charges + zmin, 0.3, "fz$\\times$2", fontsize=14)

ax.set_ylim(0, 0.99)
ax.set_xlim(min(ZZ[0], ZZ_sil[0]), max(ZZ[-1], ZZ_sil[-1])+20)

ax.tick_params(
    axis='both',          # changes apply to the x-axis
    which='both',      # both major and minor ticks are affected
    bottom='on',      # ticks along the bottom edge are off
    top='off',         # ticks along the top edge are off
    labelbottom='on', labelleft='off') # labels along the bottom edge are off


ax.set_xlabel("Z", fontsize=16)

ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)

#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 2
jj = 1

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz*10, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil*2, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR*2, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.6*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.6*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')
ax.text(0.6*charges + zmin, 0.6, "fz$\\times$2", fontsize=14, color='red')

ax.text(0.6*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14)
ax.text(0.6*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14)
ax.text(0.6*charges + zmin, 0.3, "fz$\\times$2", fontsize=14)

ax.set_ylim(0, 0.99)

ax.tick_params(axis='both', which='both', bottom='on', labelbottom='on', labelleft='off')
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)


#------------------------------------------------------------------------------------------------
# From bottom left, to upper right.
ii = 2
jj = 2

##############################
zmin = zminmax[jj*2 + 0 + 2*ii*3]
zmax = zminmax[jj*2 + 1 + 2*ii*3]
charges = int(zmax-zmin)
zmin, zmax = zmin-1, zmax+1
ZZ = np.arange(zmin, zmax+1)
ffzCR = np.array(fdistCR[cum:cum+charges+1])
ffzCR = np.concatenate([[0.0],ffzCR,[0.0]])
##############################

##############################
zmin_sil = zminmax_sil[jj*2 + 0 + 2*ii*3]
zmax_sil = zminmax_sil[jj*2 + 1 + 2*ii*3]
charges_sil = int(zmax_sil-zmin_sil)
zmin_sil, zmax_sil = zmin_sil-1, zmax_sil+1
ZZ_sil = np.arange(zmin_sil, zmax_sil+1)
ffzCR_sil = np.array(fdistCR_sil[cum_sil:cum_sil+charges_sil+1])
ffzCR_sil = np.concatenate([[0.0],ffzCR_sil,[0.0]])
##############################

cum    += int(charges)+1
cum_sil+= int(charges_sil)+1
##############################

ax = fig.add_axes([wpad + ii*xs_panel, hpad + jj*ys_panel, xs_panel, ys_panel])

#ax.plot(ZZ+0.5, ffz, "-r", linewidth=2, drawstyle='steps', alpha=1.0)
# silicates
ax.plot(ZZ_sil+0.5, ffzCR_sil, "-k", linewidth=2, drawstyle='steps', alpha=1.0)
# carbon
ax.plot(ZZ+0.5, ffzCR, "--r", linewidth=2, drawstyle='steps', alpha=1.0)


ax.text(0.25*charges + zmin, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
ax.text(0.25*charges + zmin, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

ax.text(0.25*charges + zmin, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14)
ax.text(0.25*charges + zmin, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14)

#print(1.*charges + zmin)
#ax.text(6, 0.8, "$\\langle$Z$\\rangle =%.2f$"%zmean[jj + ii*3], fontsize=14, color='red')
#ax.text(6, 0.7, "$\\sigma_{Z}\'=%.2f$"%zstd[jj + ii*3], fontsize=14, color='red')

#ax.text(6, 0.5, "$\\langle$Z$\\rangle =%.2f$"%zmean_sil[jj + ii*3], fontsize=14)
#ax.text(6, 0.4, "$\\sigma_{Z}\'=%.2f$"%zstd_sil[jj + ii*3], fontsize=14)


ax.set_ylim(0, 0.99)

ax.tick_params(axis='both', which='both', bottom='on', labelbottom='on', labelleft='off')
ax.tick_params(axis='both', which='major', length=4, width=2, labelsize=12)

ax.set_xlim(-1.8, 11)

#fig.show()

fig.savefig("./Test/fZ_5AA_50AA_100AA_WNM_CNM_CMM.pdf", format="pdf")
