#Michael Weger
#weger@tropos.de
#Permoserstrasse 15
#04318 Leipzig                   
#Germany

# load external python packages
import numpy as np
from netCDF4 import Dataset

# load model specific *py files
import domain_decomp as ddcp


def get_emissions(comm, param_dict):
    '''
    Reads in the emission fields for 
    passive-tracer dispersion.

    comm... communicator
    param_dict... parameter dictionary 
    '''

    ng = int(param_dict['n_ghost'])

    mpicomm = comm.mpicomm
    pids = comm.pids
    nri = comm.nri
    ncj = comm.ncj
  
    rank = mpicomm.Get_rank()

    ind_p = pids.index(rank)

    ind_pr = ind_p / (len(ncj) - 1)
    ind_pc = ind_p - ind_pr * (len(ncj) - 1)   

    emiss_full = []
    
    ny = nri[ind_pr + 1] - nri[ind_pr]
    nx = ncj[ind_pc + 1] - ncj[ind_pc]   
    nz = comm.nz


    # if no tracers are computed, return 3 empty lists
    if param_dict['with_emissfile'] == False:
       return [], []

    emiss_file = Dataset('INPUT/' + param_dict['simulation_name'] + '_emiss.nc', 'r')
    var_names =  [str(name) for name in emiss_file.variables.keys() if name not in ['x', 'y', 'ufl']]    

    if rank == 0:
        for name in var_names:
            var = emiss_file.variables[name]
            shape = var[:].shape
            emiss_full.append(var[:, :, :])
    else:
        for name in var_names:
            emiss_full.append(None)

    emiss_flds = []
    for n, name in enumerate(var_names):        
        fld = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
        ddcp.distribute_data(mpicomm, emiss_full[n], fld, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
        emiss_flds.append(fld)
   
    emiss_file.close()
    
    return emiss_flds, var_names
