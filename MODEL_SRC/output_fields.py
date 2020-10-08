# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# load external python packages
import numpy as np
from netCDF4 import Dataset

# load model specific *py files
import domain_decomp as ddcp
from multigrid import grad_p
from explicit_schemes import vorticity


def organize_outfields(all_output_fields, param_dict, emiss_names):
    """
    Organizes the list of output fields at simulation initialization.
    
    all_output_fields... full list of potential output fields
    param_dict... parameter dictionary
    emiss_names... Names of transported tracers
    """
    
    global output_fields
    global output_types
    global output_names
    global output_longnames
    global output_units

    global calc_dp, calc_vort

    
    t_units = param_dict['t_units']
    l_units = param_dict['l_units']

    u_units = l_units + t_units + '^-1'
    p_units = 'kg' + l_units + '^-1' + t_units + '^-2'
    dp_units = 'kg' + l_units + '^-2' + t_units + '^-2'
    vort_units = t_units + '^-1'
    emiss_units = 'u'
   
    if len(emiss_names):
        conc_units = emiss_units + l_units + '^{-3}'    


    all_output_names = [
                           'U', 'V', 'W', 'Pp', 'THETAv', 'QV', 'DPX', 'DPY', 'DPZ', 'VORTX', 'VORTY', 'VORTZ',
                           'U_SGS', 'V_SGS', 'W_SGS'
                       ]

    all_output_names.extend(emiss_names)


    all_output_longnames = [
                               'Velocity component u', 'Velocity component v', 'Velocity component w', 'Perturbation pressure', 
                               'Virtual potential temperature', 'Specific humidity', 'Pressure gradient component x', 'Pressure gradient component y',
                               'Pressure gradient component z', 'Vorticity component x', 'Vorticity component y', 'Vorticity component z',
                               'Subscale turbulent intensity component u', 'Subscale turbulent intensity component v', 'Subscale turbulent intensity component w'
                           ]

    all_output_longnames.extend(['Concentration of tracer ' + name for name in emiss_names])
    

    all_output_types = ['u', 'v', 'w','c', 'c', 'c', 'u', 'v', 'w', 'c', 'c', 'c', 'u', 'v', 'w'] 
  
    all_output_types.extend(['c' for name in emiss_names])

   
    all_output_units = [
                           u_units, u_units, u_units, p_units, 'K', '', dp_units, dp_units, dp_units, vort_units, 
                           vort_units, vort_units, u_units, u_units, u_units
                       ]

    all_output_units.extend([conc_units for name in emiss_names])


    output_names = list(param_dict['output_fields'])
    output_names.extend(emiss_names) 

    output_fields = []
    output_longnames = []
    output_units = []
    output_types = []

    

    for name in output_names:

         index = all_output_names.index(name)

         output_fields.append(all_output_fields[index])
         output_types.append(all_output_types[index])
         output_longnames.append(all_output_longnames[index])
         output_units.append(all_output_units[index])


    if any(('DPX' in output_names, 'DPY' in output_names, 'DPZ' in output_names)):
        calc_dp = True
    else:
        calc_dp = False

    if any(('VORTX' in output_names, 'VORTY' in output_names, 'VORTZ' in output_names)):
        calc_vort = True
    else:
        calc_vort = False

    
def write_output_fields(comm, all_output_fields, time, out_step, param_dict, int_ops):
    """
    This routine outputs a variable list of prognostic fields
    at simulation time "time" in the form of a new netcdf.
    Before creating the file, all subdomain field parts need to be gathered
    on root process.
   
    all_output_fields... full list of potential output fields
    comm... the more general communicator object which contains the MPI-communicator
    time... simulation time in seconds
    out_step... number of output file
    param_dict... dictionary that contains auxiliary information
    int_ops... interpolation operators needed to optionally map to z=const
    """

   
    global output_fields
    global output_types
    global output_names
    global output_longnames
    global output_units    

    global calc_dp, calc_vort

    ng = param_dict['n_ghost']

    int_zs_z, int_z_zs = int_ops[:]

    mpicomm = comm.mpicomm
    ncj = comm.ncj
    nri = comm.nri
    nz = comm.nz  
    pids = comm.pids

    rank = mpicomm.Get_rank()

    xcoord = param_dict['xcoord']
    x2coord = param_dict['x2coord']
    ycoord = param_dict['ycoord']
    y2coord = param_dict['y2coord'] 
    zcoord = param_dict['zcoord']
    z2coord = param_dict['z2coord']
    output_z = param_dict['conv_output_z']

    flds_full = []
    nx = ncj[-1]
    ny = nri[-1]


    if calc_dp:
        dp_x, dp_y, dp_z = grad_p(all_output_fields[3])
        all_output_fields[6][:] = dp_x
        all_output_fields[7][:] = dp_y
        all_output_fields[8][:] = dp_z

    if calc_vort:
        vort_x, vort_y, vort_z = vorticity(all_output_fields[0], all_output_fields[1], all_output_fields[2])
        all_output_fields[9][:] = vort_x
        all_output_fields[10][:] = vort_y
        all_output_fields[11][:] = vort_z


    if output_z:
        flds_out = []
        zcoord = param_dict['zscoord']
        nz = zcoord.size
        for n, fld, in enumerate(output_fields):
            if output_types[n] == 'c':
                fldc = fld
            elif output_types[n] == 'u':
                fldc = 0.5 * (fld[:, :, 1:] + fld[:, :, :-1])
            elif output_types[n] == 'v':
                fldc = 0.5 * (fld[:, 1:] + fld[:, :-1])
            elif output_types[n] == 'w':
                fldc = 0.5 * (fld[1:] + fld[:-1])
            a, ny_sub, nx_sub = fldc.shape
            flds_out.append((int_zs_z * fldc.flatten()).reshape(nz, ny_sub, nx_sub))
        fld_tps_out = ['c' for tp in fld_tps]

    else:
        flds_out = output_fields
        fld_tps_out = output_types

    if rank == 0:

        t_units = param_dict['t_units_ref']
        file_base = param_dict['simulation_name']
        outfile = Dataset('./OUTPUT/' + file_base + '_' + str(out_step).zfill(5) + '.nc', 'w', type = 'NETCDF4')
        xdim = outfile.createDimension('x', nx)
        ydim = outfile.createDimension('y', ny)
        zdim = outfile.createDimension('z', nz)
        if not output_z:
            x2dim = outfile.createDimension('x2', nx + 1)
            y2dim = outfile.createDimension('y2', ny + 1)
            z2dim = outfile.createDimension('z2', nz + 1)
        tdim = outfile.createDimension('time', 1)

        xvar = outfile.createVariable('x', np.float, 'x')
        xvar[:] = xcoord
        yvar = outfile.createVariable('y', np.float, 'y')
        yvar[:] = ycoord
        zvar = outfile.createVariable('z', np.float, 'z')
        zvar[:] = zcoord
        if not output_z:
            x2var = outfile.createVariable('x2', np.float, 'x2')
            x2var[:] = x2coord
            y2var = outfile.createVariable('y2', np.float, 'y2')
            y2var[:] = y2coord
            z2var = outfile.createVariable('z2', np.float, 'z2')
            z2var[:] = z2coord
        tvar = outfile.createVariable('time', int, 'time')
        tvar[:] = int(time)
        tvar.units = t_units

    for n, name in enumerate(output_names):
        nlev = nz

        if fld_tps_out[n] == 'w':
            nlev = nz + 1
        if rank == 0:
            nc_vars = []   
            if output_z:
                nc_vars.append(outfile.createVariable(name, np.float, ('time', 'z', 'y', 'x')))
                fld_full = np.empty([nlev, ny, nx])
            else:
                if fld_tps_out[n] == 'c':
                    nc_vars.append(outfile.createVariable(name, np.float, ('time', 'z', 'y', 'x')))
                    fld_full = np.empty([nlev, ny, nx])
                elif fld_tps_out[n] == 'u':
                    nc_vars.append(outfile.createVariable(name, np.float, ('time', 'z', 'y', 'x2')))
                    fld_full = np.zeros([nlev, ny, nx + 1])
                elif fld_tps_out[n] == 'v':
                    nc_vars.append(outfile.createVariable(name, np.float, ('time', 'z', 'y2', 'x')))
                    fld_full = np.empty([nlev, ny + 1, nx])
                elif fld_tps_out[n] == 'w':
                    nc_vars.append(outfile.createVariable(name, np.float, ('time', 'z2', 'y', 'x')))
                    fld_full = np.empty([nlev, ny, nx])
 
        else:
            fld_full = [None for k in range(nlev)]

        buff = flds_out[n][ng:-ng, ng:-ng, ng:-ng]
        for k in range(nlev): 
            ddcp.gather_data(mpicomm, fld_full, buff[k],  nri, ncj, k, pids, type=fld_tps_out[n])
        if rank == 0:
            nc_vars[-1][0] = fld_full
            nc_vars[-1].long_name = output_longnames[n]
            nc_vars[-1].units = output_units[n]
            nc_vars[-1].crs = "local metric coordinates"

    if rank == 0:
        outfile.close()    
