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
import sps_operators as ops


def init_grid(mpicomm, param_dict):
    """
    Root process opens the grid file and reads in
    the grid information and fields.

    mpicomm... MPI communicator
    param_dict... parameter dictionary
    """

    rank = mpicomm.Get_rank()

    ng = int(param_dict['n_ghost'])

    ncfile = Dataset('./INPUT/' + param_dict['simulation_name'] + '.nc', 'r')

    if rank == 0:
        uhl = ncfile.variables['uhl'][:]
        ufl = ncfile.variables['ufl'][:]
        x = ncfile.variables['x'][:]
        y = ncfile.variables['y'][:]
        x2 = ncfile.variables['x2'][:]        
        y2 = ncfile.variables['y2'][:]
        dx = x2[1:] - x2[:-1] 
        dy = y2[1:] - y2[:-1]
        dz = uhl[1:] - uhl[:-1]        
        ffx = ncfile.variables['fx'][:]
        ffy = ncfile.variables['fy'][:]
        ffz = ncfile.variables['fz'][:]       
        fvol = ncfile.variables['fvol'][:]

        hsurf = ncfile.variables['hsurf'][:]
        
        dhsurfdx = np.zeros_like(ffx)
        dhsurfdy = np.zeros_like(ffy)
         
        dhsurfdx[:, :, 1:-1] = (
                                   (hsurf[:, :, 1:] - hsurf[:, :, :-1]) / 
                                   (dx[1:] + dx[:-1]).reshape(1, 1, dx.size - 1)
                               )
        dhsurfdy[:, 1:-1] = (
                                (hsurf[:, 1:] - hsurf[:, :-1]) / 
                                (dy[1:] + dy[:-1]).reshape(1, dy.size - 1, 1)
                            )      

        deff_v = ncfile.variables['deffv'][:]
        deff_hx = ncfile.variables['deffhx'][:]
        deff_hy = ncfile.variables['deffhy'][:]

        area_v = ncfile.variables['areav'][:]
        area_hx = ncfile.variables['areahx'][:]
        area_hy = ncfile.variables['areahy'][:]

        z0 = ncfile.variables['z0'][:]

        ncfile.close()

        if not param_dict['elev_fluxes']:
            area_v[1 + ng:] = 0.0
             
        nz, ny, nx = fvol.shape

        correct_ffactors(ffx, ffy, ffz, fvol, lim_fv=param_dict['lim_fv'])                         

        area_v[fvol < param_dict['lim_fv']] = 0.0
        area_hx[fvol < param_dict['lim_fv']] = 0.0
        area_hy[fvol < param_dict['lim_fv']] = 0.0

        areas_x = ffx.copy()
        areas_y = ffy.copy()
        areas_z = ffz.copy()
        vols = fvol.copy()

        areas_x.fill(1.0)
        areas_y.fill(1.0)
        areas_z.fill(1.0)
        vols.fill(1.0)

        areas_x = areas_x * dy.reshape(1, dy.size, 1) * dz.reshape(dz.size, 1, 1)
        areas_y = areas_y * dx.reshape(1, 1, dx.size) * dz.reshape(dz.size, 1, 1)
        areas_z = areas_z * dx.reshape(1, 1, dx.size) * dy.reshape(1, dy.size, 1)
        vols = vols * dx.reshape(1, 1, dx.size) * dy.reshape(1, dy.size, 1)  * dz.reshape(dz.size, 1, 1)

        ff1d = ops.put_1d([ffx, ffy, ffz])
        areas = ops.put_1d([areas_x, areas_y, areas_z])

        vols1d = vols.flatten()
        fvol1d = fvol.flatten()
     
        vols_eff = vols1d * fvol1d

        areas_eff = areas * ff1d        
 
        vols_u = ops.vol_stag(vols_eff, 'u', nz, ny, nx)
        vols_v = ops.vol_stag(vols_eff, 'v', nz, ny, nx)
        
        vols_w = ops.vol_stag(vols_eff, 'w', nz, ny, nx)        
        vols_stag = vols_u + vols_v + vols_w

        
        dginv = areas_eff / vols_stag
        dginv_x, dginv_y, dginv_z = ops.put_3d(dginv, [nz, ny, nx])      


        grid_flds = [
                        ffx, ffy, ffz, fvol, areas_x, areas_y, areas_z, vols, 
                        dginv_x, dginv_y, dginv_z,  
                        deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy, z0,
                        hsurf, dhsurfdx, dhsurfdy
                    ]

        fld_tps = [
                      'u', 'v', 'w', 'c', 'u', 'v', 'w', 'c', 
                      'u', 'v', 'w',
                      'c', 'c', 'c', 'c', 'c', 'c', 'c',
                      'c', 'u', 'v'
                  ]        

        hsrf_min = np.min(hsurf[0])
        hsrf_max = np.max(hsurf[0])
        h_max = hsrf_max + uhl[-1]
        dh_min = uhl[1] - uhl[0]
        dh_max = uhl[-1] - uhl[-2] 
        h_zconst = np.linspace(hsrf_min, hsrf_max, int((hsrf_max - hsrf_min) / dh_min) + 1).tolist()
        h_zconst.extend((hsrf_max + uhl[1:]).tolist())
        h_min = h_zconst[-1] 
        h_zconst.extend(np.linspace(h_min, h_max, int((h_max - h_min) / dh_max)).tolist())

    else:

        x = ncfile.variables['x'][:]
        x2 = ncfile.variables['x2'][:]
        nx = x.size
        y = ncfile.variables['y'][:]
        y2 = ncfile.variables['y2'][:]
        ny = y.size
        dx = x2[1:] - x2[:-1]
        dy = y2[1:] - y2[:-1]
        ufl = ncfile.variables['ufl'][:]
        uhl = ncfile.variables['uhl'][:]
        dz = uhl[1:] - uhl[:-1]
        nz = ufl.size        

        hsurf = ncfile.variables['hsurf'][:]
        ncfile.close()
        grid_flds = [None for i in range(21)]
        fld_tps = [None for i in range(21)]

        hsrf_min = np.min(hsurf[0])
        hsrf_max = np.max(hsurf[0])
        h_max = hsrf_max + uhl[-1]
        dh_min = uhl[1] - uhl[0]
        dh_max = uhl[-1] - uhl[-2]
        h_zconst = np.linspace(hsrf_min, hsrf_max, int((hsrf_max - hsrf_min) / dh_min) + 1).tolist()
        h_zconst.extend((hsrf_max + uhl[1:]).tolist())
        h_min = h_zconst[-1]
        h_zconst.extend(np.linspace(h_min, h_max, int((h_max - h_min) / dh_max)).tolist())
    
    # additional parameters needed for friction computation
    p_0 = (x[0], y[0])
    p_x1 = (x[1], y[0])
    p_y1 = (x[0], y[1])  
 
    param_dict.update({'dx':dx[ng:-ng]})
    param_dict.update({'dy':dy[ng:-ng]})
    param_dict.update({'dz':dz[ng:-ng]})
    param_dict.update({'dx_ghst':dx})
    param_dict.update({'dy_ghst':dy})
    param_dict.update({'dz_ghst':dz})
    param_dict.update({'xcoord':x[ng:-ng]})
    param_dict.update({'x2coord':x2[ng:-ng]})
    param_dict.update({'ycoord':y[ng:-ng]})
    param_dict.update({'y2coord':y2[ng:-ng]})
    param_dict.update({'zcoord':ufl[ng:-ng]})
    param_dict.update({'z2coord':uhl[ng:-ng]})
    param_dict.update({'xcoord_ghst':x})
    param_dict.update({'x2coord_ghst':x2})
    param_dict.update({'ycoord_ghst':y})
    param_dict.update({'y2coord_ghst':y2})
    param_dict.update({'zcoord_ghst':ufl})
    param_dict.update({'z2coord_ghst':uhl})

    param_dict.update({'dh':min(np.min(dx), np.min(dy))})
    param_dict.update({'h_zconst':h_zconst})

    param_dict.update({'l_units':'m'})
    param_dict.update({'t_units':'s'})
    
    nz = nz - 2 * ng
    ny = ny - 2 * ng
    nx = nx - 2 * ng

    return nz, ny, nx, grid_flds, fld_tps


def correct_ffactors(ffx, ffy, ffz, fvol, lim_fv=1e-2):
    """
    This function is for post-processing of the derived area and
    volume-scaling factors.
    It removes inconsistency between the area and volume scaling factors:

    (1) It closes off all grid cells, that do not contain a significant physical volume.
    (2) It sets the physical volume of completely filled cells to a small numerical value         
    (3) It removes the physical volume of cells with only one open face area (holes).

    ffx, ffy, ffz... area-scaling fators
    fvol... volume-scaling factors
    lim_fv... threshold volume fraction below which cells are closed off
    """


    nz, ny, nx = fvol.shape[:]

    eps1 = 0.0
    eps2 = 1e-80

    zero = fvol <= lim_fv
    ffx[:, :, 1:][zero] = eps1
    ffx[:, :, :-1][zero] = eps1
    ffy[:, 1:][zero] = eps1
    ffy[:, :-1][zero] = eps1
    ffz[1:][zero] = eps1
    ffz[:-1][zero] = eps1
    fvol[zero] = eps2

    for i in range(20):
        closed_fx_l = ffx[:, :, :-1] <= lim_fv 
        closed_fx_r = ffx[:, :, 1:] <= lim_fv 
        closed_fy_l = ffy[:, :-1] <= lim_fv 
        closed_fy_r = ffy[:, 1:] <= lim_fv 
        closed_fz_l = ffz[:-1] <= lim_fv 
        closed_fz_r = ffz[1:] <= lim_fv 
        ffx[:, :, :-1][closed_fx_l] = eps1 ** 2
        ffx[:, :, 1:][closed_fx_r] = eps1 ** 2
        ffy[:, :-1][closed_fy_l] = eps1 ** 2
        ffy[:, 1:][closed_fy_r] = eps1 ** 2
        ffz[:-1][closed_fz_l] = eps1 ** 2
        ffz[1:][closed_fz_r] = eps1 ** 2
        closed_count = closed_fx_l + closed_fx_r + closed_fy_l + closed_fy_r + closed_fz_l + closed_fz_r
        shut_cells = closed_count >= 5
        ffx[:, :, :-1][shut_cells] = eps1 ** 2
        ffx[:, :, 1:][shut_cells] = eps1 ** 2
        ffy[:, :-1][shut_cells] = eps1 ** 2
        ffy[:, 1:][shut_cells] = eps1 ** 2
        ffz[:-1][shut_cells] = eps1 ** 2
        ffz[1:][shut_cells] = eps1 ** 2
        fvol[shut_cells] = eps2 ** 1.5
