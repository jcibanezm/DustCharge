#!/home/jcibanezm/codes/libs/miniconda3/bin python
#from memory_profiler import profile
import gc
import sys

def print_to_logfile(text, grain_size=0, grain_type=0., filename=0., first_call=False):
    """
    At some point this function will open and save information into a log file.
    """
    #if first_call == True:
    #    log_dir="/home/jcibanezm/codes/DustAnalysis/Charge/LogFiles"
    #    f = open('%s/logfile_%s_%sGrains_%iAA'%(log_dir, filename, grain_type, grain_size), 'w')
    #    f.write("-------------------------------------------------------------------------------------")
    #    f.write("Juan C. Ibanez-Mejia                                              Cologne, June 2017.")
    #    f.write("Charge distribution calculation of %s grains with size %i."%(grain_type, grain_size))
    #    f.write('%s\n'%text)

    print (text)


# Define the charge distribution function calculation to run in parallel
#@profile
def charge_dist(filename, pos, num_procs, grain_size, grain_type, test=False):
    """ Calculates the charge distribution """
    import compute_charge_dist as fz
    import yt
    import numpy as np
    import sys

    mH = 1.6733e-24 # g
    mC = 12.011*mH

    print_to_logfile("Reading Flash file. \t proc = %i"%pos, grain_size=grain_size, grain_type=grain_type, filename=filename, first_call=True)

    pf = yt.load("%s"%(filename))

    ppc = 3.085677e18
    le = [ -15*ppc,  -4.93696000e+19,  -4.93696000e+19]
    re = [ +15*ppc,  +4.93696000e+19,  +4.93696000e+19]

    dd = pf.box(le,re)
    if pos == 0:
        print("Total number of cells = %i"%len(dd["dens"]))
    #dd = pf.all_data()

    # Read file and calculate the number of cells.
    if test == True:
        num_cells = 5*num_procs
    else:
        num_cells = len(dd["dens"])

    # calculate the number of cells to be distributed per processor. [has to be an interger number]
    # Calculate the offset in the array to
    cells_per_proc = num_cells // num_procs

    # Find the offset in the array where the information is coming from.
    offset = cells_per_proc*pos

    # The missing cells are handled by the last processor
    if pos == (num_procs-1): cells_per_proc += num_cells % num_procs

    dens = np.array(np.zeros(cells_per_proc))
    temp, Av = np.zeros_like(dens), np.zeros_like(dens)
    ih2, iha, ihp, ico, icp, dx, xx, yy, zz = np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens),np.zeros_like(dens)

    zmean, zmode, zstd  =  np.zeros_like(dens), np.zeros_like(dens), np.zeros_like(dens)

    nH, nC, xHp, xCp, ne, xe = np.zeros_like(dens), np.zeros_like(dens), np.zeros_like(dens), np.zeros_like(dens), np.zeros_like(dens), np.zeros_like(dens)

    zminmax = np.array(np.zeros(2*cells_per_proc))

    tauz  = np.zeros_like(dens)

    fdist   = []
    #fdist = np.zeros_like(dens)

    dictionary = {"info":"charge distribution calculated in processor %i"%pos}

    ############################################################################
    # I should read the Qabs table here, and pass it to the Jrate, calculation.
    ############################################################################
    Qabs = fz.get_QabsTable(grain_type, grain_size)

    #########################################################################################################
    #                          Charge distribution loop over all cells.
    #########################################################################################################
    print_to_logfile("Looping over %i cells in proc = %i"%(cells_per_proc, pos), grain_size=grain_size, grain_type=grain_type, filename=filename)

    for ii in range(cells_per_proc):

        index = offset + ii

        if ii%1.0e4 == 0: print("I'm proc %i, running cell %i"%(pos, index))

        dens[ii]  = dd["dens"][index].value
        temp[ii]  = dd["temp"][index].value
        Av[ii]    = dd["cdto"][index].value

        ih2[ii] = dd["ih2 "][index].value
        iha[ii] = dd["iha "][index].value
        ihp[ii] = dd["ihp "][index].value
        ico[ii] = dd["ico "][index].value
        icp[ii] = dd["icp "][index].value
        dx[ii]  = dd["dx"][index].value
        xx[ii]  = dd["x"][index].value
        yy[ii]  = dd["y"][index].value
        zz[ii]  = dd["z"][index].value

        # Number density of hydrogen atoms.
        nH[ii]    = dd["dens"][index].value*(dd["ihp "][index].value+dd["iha "][index].value + dd["ih2 "][index].value)/(1.4*mH)
        nH2       = dd["dens"][index].value*(dd["ih2 "][index].value)/(1.4*mH)

        # Number density of carbon atoms.
        nC[ii]    = dd["dens"][index].value*(dd["icp "][index].value + dd["ico "][index].value)/(1.4*mC)

        # fraction of ionized hydrogen and carbon
        xHp[ii] = dd["dens"][index].value*dd["ihp "][index].value/(1.4*mH) / nH[ii]
        xCp[ii] = dd["dens"][index].value*dd["icp "][index].value/(1.4*mC) / nC[ii]

        # Number density of electrons
        ne[ii]    = dd["dens"][index].value*(dd["ihp "][index].value/(1.4*mH) + dd["icp "][index].value/(1.4*mC))

        # electron fraction.
        xe[ii]    = ne[ii] / (nH[ii]+nC[ii])

        ############################################################################################
        # Run the charge distribution calculation!!!
        Jpe, Je, Jh, Jc, ZZall = fz.compute_currents ([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type, Qabs)
        # Compute CR currents
        JCRe, JCRpe, ZZnew     = fz.compute_CR_currents(nH2, grain_size, grain_type)

        zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size, grain_type)
        new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type, Qabs, zeq=zeq)

        ffz, ZZ                = fz.vector_fz        (Jpe, Je, Jh, Jc, JCRe, JCRpe, ZZall, new_zmin, new_zmax, includeCR=True)

        # This should be updated accordingly!!!
        tauz[ii]               = fz.tauz_vec         (Jpe, Je, Jh, Jc, ZZall, ffz, new_zmin, new_zmax)

######## Commented on Feb 13 2018 #### Before including CR currents.
#        zeq                    = fz.get_zeq_vec      (Jpe, Je, Jh, Jc, ZZall, grain_size, grain_type)
#        new_zmin, new_zmax     = fz.get_new_zmin_zmax([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type, Qabs, zeq=zeq)
#        ffz, ZZ                = fz.vector_fz        (Jpe, Je, Jh, Jc, ZZall, new_zmin, new_zmax)
#        tauz[ii]               = fz.tauz_vec         (Jpe, Je, Jh, Jc, ZZall, ffz, new_zmin, new_zmax)


        #zeq                 = fz.get_zeq          ([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type)
        #new_zmin, new_zmax  = fz.get_new_zmin_zmax([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type, zeq=zeq)
        #ffz, ZZ             = fz.compute_fz_speed ([nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], grain_size, Av[ii], grain_type, zmin=new_zmin, zmax=new_zmax)
        #tauz[ii] = fz.get_tauz(grain_size, grain_type, [nH[ii], nC[ii]], [xHp[ii], xCp[ii]], temp[ii], Av[ii], ZZ, ffz)

        Zm        = fz.get_Zmode(ZZ, ffz)
        zmode[ii] = Zm

        avg, std  = fz.weighted_avg_and_std(ZZ, ffz)
        zmean[ii] = avg
        zstd[ii]  = std

        zminmax[ii*2]  = new_zmin
        zminmax[ii*2+1]= new_zmax

        #fdist[ii]   = offset + ii
        for jj in range(len(ffz)):
            fdist.append(ffz[jj])


    print_to_logfile("Finished loop over all cells in this processor. \t proc=%i"%pos, grain_size=grain_size, grain_type=grain_type, filename=filename)

    dictionary["dens"]     = dens
    dictionary["temp"]     = temp
    dictionary["Av"]       = Av
    dictionary["zmean"]    = zmean
    dictionary["zmode"]    = zmode
    dictionary["zstd"]     = zstd
    dictionary["tauz"]     = tauz
    dictionary["fdist"]    = np.array(fdist)
    dictionary["zminmax"]  = np.array(zminmax)
    dictionary["nH"]       = nH
    dictionary["nC"]       = nC
    dictionary["xHp"]      = xHp
    dictionary["xCp"]      = xCp
    dictionary["ne"]       = ne
    dictionary["xe"]       = xe

    dictionary["iha"]     = iha
    dictionary["ihp"]     = ihp
    dictionary["ih2"]     = ih2
    dictionary["ico"]     = ico
    dictionary["icp"]     = icp
    dictionary["dx"]      = dx
    dictionary["xx"]      = xx
    dictionary["yy"]      = yy
    dictionary["zz"]      = zz

    del dens, temp, Av, zmean, zmode, zstd, fdist, zminmax, nH, nC, ne, xe, xHp, xCp, tauz
    del pf, dd, iha, ihp, ih2, ico, icp, dx, xx, yy, zz

    gc.collect()

    return dictionary

#@profile
def gather_all_data(cdist, comm):
    """
    gather the information stored in the dictionaries accross the different processors.
    """
    import numpy as np

    rank = comm.Get_rank()
    size = comm.Get_size()

    # Gather the results of the charge distribution from all processors
    results = comm.gather(cdist, root=0)
    #end = time.clock()

    # Concatenate the arrays into a final dictionary.
    if rank == 0:
        cdist = dict(results[0])

        cdist["MPI"] = "MPI calculation of the charge distribution in %i procs" %(size)

        fields  = ['dens', 'temp', 'Av', 'zmean', 'zmode', 'zstd', 'fdist', 'zminmax', 'nH', 'nC', 'ne', 'xe', 'xHp', 'xCp', 'nH2', 'fH2shield', 'G', 'tauz']
        # Loop over all procesors to concatenate the data.
        for proc in range(size-1):
            # Loop over al fields.
            for field in fields:
                # In the new dictionary, append the data of a given field from another processor.
                cdist[field] = np.append(cdist[field], results[proc+1][field])

    del results

    return cdist


def save_charge_distribution(cdist, grain_size, grain_type, comm, filename, out_dir='default', start=0.0, end=0.0):
    """
    Save the dictionary containing the information of the charge distribution in a pickle file.
    """
    import pickle

    rank = comm.Get_rank()
    size = comm.Get_size()

    total_time = end-start

    cdist["info"] = "Charge distribution of %s grains with size %i AA, from the simulation %s"%(grain_type, grain_size, filename)
    cdist["time"] = "Time taken to run the charge distribution calculation t = %.1f s" %(total_time)

    # Save the data into a pickle file.
    if rank == 0:
        print (cdist["info"])
        print (cdist["time"])

        if out_dir == 'default':
            out_dir = "/home/jcibanezm/codes/run/Silcc/CF_Prabesh/ChargeDist"
        else:
            out_dir = out_dir

        outname = "ChargeDist_CF_%sgrains_%iAA.pkl"%(grain_type, grain_size)

        print_to_logfile("Saving charge distribution to %s/%s"%(out_dir, outname), grain_size=grain_size, grain_type=grain_type, filename=filename)
        outfile = open('%s/%s'%(out_dir, outname), 'wb')
        pickle.dump(cdist, outfile)
        outfile.close()


def save_fz_proc(cdist, grain_size, grain_type, comm, filename, out_dir="default"):
    """
    Save the dictionary containing the information of the charge distribution in a pickle file.
    """
    import pickle

    rank = comm.Get_rank()
    size = comm.Get_size()

    cdist["info"] = "Charge distribution of %s grains with size %i AA, from the simulation %s"%(grain_type, grain_size, filename)
    cdist["proc"] = rank
    cdist["size"] = size

    # Save the data into a pickle file.
    print(cdist["info"])
    print("Saving the distribution within proc %i, out of %i"%(rank, size))

    if out_dir == 'default':
        out_dir = "/home/hpc/pr62su/di36yav/ParallelCharge/Output/HiRes/tmp"
    else:
        out_dir = out_dir

    outname = "ChargeDist_CF_%sgrains_%iAA_proc%.3i.pkl"%(grain_type, grain_size,rank)

    print_to_logfile("Saving charge distribution to %s/%s"%(out_dir, outname), grain_size=grain_size, grain_type=grain_type, filename=filename)
    outfile = open('%s/%s'%(out_dir, outname), 'wb')
    pickle.dump(cdist, outfile)
    outfile.close()
