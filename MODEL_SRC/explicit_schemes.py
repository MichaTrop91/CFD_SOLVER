# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# load external python packages
import numpy as np
import scipy
from netCDF4 import Dataset
from time import time

# load model specific *py files
import domain_decomp as ddcp
import sps_operators as ops
import coupling as cpl


def init_schemes(comm, grid_flds, param_dict):
    """
    This function initializes the data and objects
    used by the explicit schemes in this script,
    like the advection,  diffusion and surface routines.
    It also distributes the grid fields from root process
    to the subdomain processes, and initializes boundary
    objects for communication.

    comm... communicator
    grid_flds... grid fields located on root process
    param_dict... parameter dictionary
    
    grid_flds_sub... returned subdomain grid fields used by other routines not contained in this script
    trans_z_ops... returned interpolation operators used by the output routine to map from terrain-following coordinates to
                   a non-curvlinear Cartesian grid    
    """

    global ng

    global bnds_expl_u_r, bnds_expl_u_s
    global bnds_expl_v_r, bnds_expl_v_s
    global bnds_expl_w_r, bnds_expl_w_s
    global bnds_expl_c_r, bnds_expl_c_s
    global bnds_expl_p_r, bnds_expl_p_s
    global bnds_expl_chem_r, bnds_expl_chem_s

    global bnds_expl_cycl_u_r, bnds_expl_cycl_u_s
    global bnds_expl_cycl_v_r, bnds_expl_cycl_v_s
    global bnds_expl_cycl_w_r, bnds_expl_cycl_w_s
    global bnds_expl_cycl_c_r, bnds_expl_cycl_c_s
    global bnds_expl_cycl_p_r, bnds_expl_cycl_p_s
    global bnds_expl_cycl_chem_r, bnds_expl_cycl_chem_s

    global bnds_expl_c_x_s, bnds_expl_c_y_s, bnds_expl_c_z_s
    global bnds_expl_cycl_c_x_s, bnds_expl_cycl_c_y_s, bnds_expl_cycl_c_z_s
    global bnds_expl_c_x_r, bnds_expl_c_y_r, bnds_expl_c_z_r
    global bnds_expl_cycl_c_x_r, bnds_expl_cycl_c_y_r, bnds_expl_cycl_c_z_r

    global area_eff_x, area_eff_y, area_eff_z, vols_eff
    global area_x, area_y, area_z, vols

    global frict_data

    global terrain

    global rec_type

    global bnds_global_lst

    global grid_flds_stag

    global lmix_h, lmix_v, lmix_hv

    global thetav_bnds_s, thetav_bnds_r

    global trans_zs_z, trans_z_zs, hhl_z

    global vols_map_z

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    pids = comm.pids    
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz    
    ind_pr = comm.ind_pr
    ind_pc = comm.ind_pc
    npc = comm.npc
    npr = comm.npr

    nx = ncj[ind_pc + 1] - ncj[ind_pc]
    ny = nri[ind_pr + 1] - nri[ind_pr]

    n_ghost = int(param_dict['n_ghost'])
    ng = n_ghost

    # initialization of communication objects between subdomains

    bnd_x = param_dict['bnd_xl']
    bnd_y = param_dict['bnd_yl']
    bnd_z = param_dict['bnd_zl']
    bnd_pres_x = param_dict['bnd_pres_x']
    bnd_pres_y = param_dict['bnd_pres_y']
    bnd_pres_z = param_dict['bnd_pres_z']
    bnd_chem_x = param_dict['bnd_chem_xl']
    bnd_chem_y = param_dict['bnd_chem_yl']
    bnd_chem_z = param_dict['bnd_chem_zr']

    bnds_expl_u_s, bnds_expl_u_r, bnds_expl_cycl_u_s, bnds_expl_cycl_u_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='u', ghst_inds = ng)
    bnds_expl_v_s, bnds_expl_v_r, bnds_expl_cycl_v_s, bnds_expl_cycl_v_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='v', ghst_inds = ng)
    bnds_expl_w_s, bnds_expl_w_r, bnds_expl_cycl_w_s, bnds_expl_cycl_w_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='w', ghst_inds = ng)
    bnds_expl_c_s, bnds_expl_c_r, bnds_expl_cycl_c_s, bnds_expl_cycl_c_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='c', ghst_inds = ng)
    bnds_expl_p_s, bnds_expl_p_r, bnds_expl_cycl_p_s, bnds_expl_cycl_p_r =  ddcp.make_halos(comm, bnd_pres_x, bnd_pres_y, bnd_pres_z, type='p', ghst_inds = ng)
    bnds_expl_chem_s, bnds_expl_chem_r, bnds_expl_cycl_chem_s, bnds_expl_cycl_chem_r =  ddcp.make_halos(comm, bnd_chem_x, bnd_chem_y, bnd_chem_z, type='c', ghst_inds = ng)
    bnds_expl_c3_s, bnds_expl_c3_r, bnds_expl_cycl_c3_s, bnds_expl_cycl_c3_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='c', ghst_inds = ng + 1)
    bnds_expl_u3_s, bnds_expl_u3_r, bnds_expl_cycl_u3_s, bnds_expl_cycl_u3_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='u', ghst_inds = ng + 1)
    bnds_expl_v3_s, bnds_expl_v3_r, bnds_expl_cycl_v3_s, bnds_expl_cycl_v3_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='v', ghst_inds = ng + 1)
    bnds_expl_w3_s, bnds_expl_w3_r, bnds_expl_cycl_w3_s, bnds_expl_cycl_w3_r =  ddcp.make_halos(comm, bnd_x, bnd_y, bnd_z, type='w', ghst_inds = ng + 1)

    thetav_bnds_s, thetav_bnds_r, empty_list, empty_list2 = ddcp.make_halos(comm, 'dirichlet', 'dirichlet', 'dirichlet', type='c', ghst_inds=ng)

    #boundary objects used in momentum advection
    bnd_lst_send, bnd_lst_recv = ddcp.make_bnds_adv_mom(comm, bnd_x, bnd_y, bnd_z, ng)
    bnds_expl_c_x_s, bnds_expl_c_y_s, bnds_expl_c_z_s, bnds_expl_cycl_c_x_s, bnds_expl_cycl_c_y_s, bnds_expl_cycl_c_z_s = bnd_lst_send[:]
    bnds_expl_c_x_r, bnds_expl_c_y_r, bnds_expl_c_z_r, bnds_expl_cycl_c_x_r, bnds_expl_cycl_c_y_r, bnds_expl_cycl_c_z_r = bnd_lst_recv[:]


    # distribution of grid fields to subdomains

    ffx_full, ffy_full, ffz_full, fvol_full, area_x_full, area_y_full, area_z_full, vols_full = grid_flds[:8] # volume and area scaling fields, geometric areas and volumes
    dginv_xfull, dginv_yfull, dginv_zfull  = grid_flds[8:11]  # effective grid spacings for the gradient
    deff_v_full, deff_hx_full, deff_hy_full, area_v_full, area_hx_full, area_hy_full, z0_full = grid_flds[11:18] # fields for the surface scheme
    hsurf_full, dsurfdx_full, dsurfdy_full= grid_flds[18:21]  # fields for terrain-following coordinates

    if rank == 0:
        vols_eff_full = fvol_full * vols_full
        area_x_eff_full = area_x_full * ffx_full
        area_y_eff_full = area_y_full * ffy_full
        area_z_eff_full = area_z_full * ffz_full                
    else:
        vols_eff_full = None
        area_x_eff_full = None
        area_y_eff_full = None
        area_z_eff_full = None    

    area_eff_x = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    area_eff_y = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    area_eff_z = np.zeros([nz + 2 * ng + 1, ny + 2 * ng, nx + 2 * ng])
    vols_eff = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    dginv_x = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    dginv_y = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    dginv_z = np.zeros([nz + 2 * ng + 1, ny + 2 * ng, nx + 2 * ng])

    area_x = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    area_y = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    area_z = np.zeros([nz + 2 * ng + 1, ny + 2 * ng, nx + 2 * ng])
    vols = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])

    vols_a = np.full([nz + 2 * (ng + 1), ny + 2 * (ng + 1), nx + 2 * (ng + 1)], 1e-40)
    area_x_eff_a = np.zeros([nz + 2 * (ng + 1), ny + 2 * (ng + 1), nx + 2 * (ng + 1) + 1])
    area_y_eff_a = np.zeros([nz + 2 * (ng + 1), ny + 2 * (ng + 1) + 1, nx + 2 * (ng + 1)])
    area_z_eff_a = np.zeros([nz + 2 * (ng + 1) + 1, ny + 2 * (ng + 1), nx + 2 * (ng + 1)])


    deff_v = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    deff_hx = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    deff_hy = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    area_v = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    area_hx = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    area_hy = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    z0 = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])

    hsurf = np.zeros([nz + 2 * (ng + 1), ny + 2 * (ng + 1), nx + 2 * (ng + 1)])
    dsurfdx = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    dsurfdy = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    
    ddcp.distribute_data(mpicomm, area_x_eff_full, area_eff_x, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_y_eff_full, area_eff_y, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_z_eff_full, area_eff_z, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, vols_eff_full, vols_eff, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, dginv_xfull, dginv_x, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, dginv_yfull, dginv_y, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, dginv_zfull, dginv_z, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_x_full, area_x, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_y_full, area_y, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)    
    ddcp.distribute_data(mpicomm, area_z_full, area_z, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, vols_full, vols, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, vols_eff_full, vols_a[1:-1, 1:-1, 1:-1], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_x_eff_full, area_x_eff_a[1:-1, 1:-1, 1:-1], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_y_eff_full, area_y_eff_a[1:-1, 1:-1, 1:-1], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_z_eff_full, area_z_eff_a[1:-1, 1:-1, 1:-1], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)

    ddcp.distribute_data(mpicomm, deff_v_full, deff_v, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, deff_hx_full, deff_hx, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, deff_hy_full, deff_hy, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_v_full, area_v, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_hx_full, area_hx, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, area_hy_full, area_hy, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, z0_full, z0, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)

   
    fill_boundaries(vols_a, axis=0, number=1)
    fill_boundaries(vols_a, axis=1, number=1)
    fill_boundaries(vols_a, axis=2, number=1)
    fill_boundaries(area_x_eff_a, axis=0, number=1)
    fill_boundaries(area_x_eff_a, axis=1, number=1)
    fill_boundaries(area_x_eff_a, axis=2, number=1)
    fill_boundaries(area_y_eff_a, axis=0, number=1)
    fill_boundaries(area_y_eff_a, axis=1, number=1)
    fill_boundaries(area_y_eff_a, axis=2, number=1)
    fill_boundaries(area_z_eff_a, axis=0, number=1)
    fill_boundaries(area_z_eff_a, axis=1, number=1)
    fill_boundaries(area_z_eff_a, axis=2, number=1)

    ddcp.distribute_data(mpicomm, hsurf_full, hsurf[1:-1, 1:-1, 1:-1], nri[:-1],  nri[1:], ncj[:-1], ncj[1:], pids)

    fill_boundaries(hsurf, axis=0, number=1)
    fill_boundaries(hsurf, axis=1, number=1)
    fill_boundaries(hsurf, axis=2, number=1)

    ddcp.distribute_data(mpicomm, dsurfdx_full, dsurfdx, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, dsurfdy_full, dsurfdy, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)  
 
    
    ddcp.cptobounds(vols_a, bnds_expl_c3_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_c3_s, bnds_expl_c3_r)
    ddcp.cpfrombounds(vols_a, bnds_expl_c3_r, mode = 'repl')   

    ddcp.cptobounds(area_x_eff_a, bnds_expl_u3_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_u3_s, bnds_expl_u3_r)
    ddcp.cpfrombounds(area_x_eff_a, bnds_expl_u3_r, mode = 'repl')

    ddcp.cptobounds(area_y_eff_a, bnds_expl_v3_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_v3_s, bnds_expl_v3_r)
    ddcp.cpfrombounds(area_y_eff_a, bnds_expl_v3_r, mode = 'repl')

    ddcp.cptobounds(area_z_eff_a, bnds_expl_w3_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_w3_s, bnds_expl_w3_r)
    ddcp.cpfrombounds(area_z_eff_a, bnds_expl_w3_r, mode = 'repl')

    ddcp.cptobounds(hsurf, bnds_expl_c3_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_c3_s, bnds_expl_c3_r)
    ddcp.cpfrombounds(hsurf, bnds_expl_c3_r, mode = 'repl')    
        

    # grid information of shifted grids
    grid_flds_stag = avg_grid_flds_stag()

    # weights for flux limiter in reconstruction scheme for advection

    ffx = area_eff_x / area_x
    ffy = area_eff_y / area_y
    ffz = area_eff_z / area_z

    grid_flds_a = [area_x_eff_a, area_y_eff_a, area_z_eff_a, vols_a]

    if param_dict['adv_scheme'] == 'upwind':
        init_upwind_recon(grid_flds_a, [ffx, ffy, ffz], param_dict) 
        rec_type = 'upwind'
    elif param_dict['adv_scheme'] == 'ENO':
        init_WENO_recon(grid_flds_a, param_dict)
        rec_type = 'ENO'
    elif param_dict['adv_scheme'] == 'WENO':
        init_WENO_recon(grid_flds_a, param_dict)
        rec_type = 'WENO'
    else:         
        print "Wrong name in parameter list for advection scheme"
        raise ValueError


    # data for surface scheme
    karm = 0.4
    frict_data = [deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy, z0, karm]


    # terrain information    
    terrain = [hsurf, dsurfdx, dsurfdy]
  
    # z-zs mapping matrices for buoyancy calculation
    trans_zs_z, trans_z_zs, hhl_z = ops.make_trans_zs_z(comm, hsurf[ng + 1, ng + 1:-(ng + 1), ng + 1:-(ng + 1)], nx, ny, param_dict)
    trans_z_ops = [trans_zs_z, trans_z_zs]

    vols_map_z = (trans_zs_z * vols_eff[ng:-ng, ng:-ng, ng:-ng].flatten()).reshape(hhl_z.size - 1, ny, nx)

    # mixing length for diffusion scheme
    lmix_h, lmix_v, lmix_hv = derive_lmix(param_dict)


    # global (lateral) boundary communication objects
    ntracer = int(param_dict['ntracer'])
    nfields = 7 + ntracer

    bnds_global_lst = cpl.init_bnds_global(comm, param_dict, nfields)

    # initialize mass flux correction scheme for global mass conservation
    cpl.init_bnd_flux_corr(comm, area_eff_x, area_eff_y, area_eff_z, dsurfdx, dsurfdy, param_dict)

    cpl.init_neumann_bc(comm, param_dict)

    grid_flds_sub = [area_eff_x, area_eff_y, area_eff_z, area_x, area_y, area_z, vols_eff, vols, dginv_x, dginv_y, dginv_z, dsurfdx, dsurfdy]

    return grid_flds_sub, trans_z_ops



def init_fields(comm, bnd_file, param_dict):
    """
    Function to initialize all prognostic fields
    and diagnostic fields for forcing and
    turbulence generation.

    comm... communicator
    bnd_file... name of netcdf input file containing init data
    param_dict... parameter dictionary

    flds_sub... returned initialized and distributed fields 
    """

    global vel_flds
    global thetav, rho
    global tr_flds
    global th_surf
    global qv_surf
    global qv

    global ng

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()   
    pids = comm.pids
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz
    ind_pr = comm.ind_pr
    ind_pc = comm.ind_pc 

    nx = ncj[ind_pc + 1] - ncj[ind_pc]
    ny = nri[ind_pr + 1] - nri[ind_pr]

    u = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    v = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    w = np.zeros([nz + 2 * ng + 1, ny + 2 * ng, nx + 2 * ng])
    vel_flds = [u, v, w]

    thetav = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    rho = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    qv = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    th_surf = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])
    qv_surf = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng])

    urms = np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng + 1])
    vrms = np.zeros([nz + 2 * ng, ny + 2 * ng + 1, nx + 2 * ng])
    wrms = np.zeros([nz + 2 * ng + 1, ny + 2 * ng, nx + 2 * ng])

    ntracer = int(param_dict['ntracer'])
    tr_names = param_dict['tracers']
    tr_flds = []
    for n in range(ntracer):
       tr_flds.append(np.zeros([nz + 2 * ng, ny + 2 * ng, nx + 2 * ng]))

    if rank == 0:

        tr_full_lst = []
        f_open = Dataset('./INPUT/' + bnd_file, 'r')        
        u_full = f_open.variables['U'][0][ng:-ng, ng:-ng, ng:-ng]                        
        v_full = f_open.variables['V'][0][ng:-ng, ng:-ng, ng:-ng] 
        w_full = f_open.variables['W'][0][ng:-ng, ng:-ng, ng:-ng] 
        urms_full = f_open.variables['U_rms'][0][ng:-ng, ng:-ng, ng:-ng] 
        vrms_full = f_open.variables['V_rms'][0][ng:-ng, ng:-ng, ng:-ng] 
        wrms_full = f_open.variables['W_rms'][0][ng:-ng, ng:-ng, ng:-ng] 
        theta_full = f_open.variables['Theta'][0][ng:-ng, ng:-ng, ng:-ng] 
        QV_full = f_open.variables['QV'][0][ng:-ng, ng:-ng, ng:-ng] 
        QV_surf = f_open.variables['QV_S'][0, :]
        thetav_full = theta_full * (1.0 + 0.61 * QV_full)
        Th_surf = f_open.variables['Th_S'][0, :]
        rho_full = f_open.variables['Rho'][0][ng:-ng, ng:-ng, ng:-ng] 

        if any([param_dict['bnd_chem_xl'] == 'dirichlet', param_dict['bnd_chem_yl'] == 'dirichlet']):
            for name in tr_names:
                tr_full_lst.append(f_open.variables[name][0][ng:-ng, ng:-ng, ng:-ng])
        f_open.close()     

    else:

        u_full = None
        v_full = None
        w_full = None
        urms_full = None
        vrms_full = None
        wrms_full = None
        thetav_full = None
        rho_full = None
        QV_full = None
        QV_surf = None
        Th_surf = None

        tr_full_lst = []
        for name in tr_names:
            tr_full_lst.append(None)        

    ddcp.distribute_data(mpicomm, u_full, u[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, v_full, v[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, w_full, w[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, urms_full, urms[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, vrms_full, vrms[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, wrms_full, wrms[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, thetav_full, thetav[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, rho_full, rho[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, QV_full, qv[ng:-ng, ng:-ng, ng:-ng], nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, Th_surf, th_surf, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)
    ddcp.distribute_data(mpicomm, QV_surf, qv_surf, nri[:-1], nri[1:], ncj[:-1], ncj[1:], pids)

    if any([param_dict['bnd_chem_xl'] == 'dirichlet', param_dict['bnd_chem_yl'] == 'dirichlet']):
        for n, fld in enumerate(tr_full_lst):
            ddcp.distribute_data(mpicomm, fld, tr_flds[n][ng:-ng, ng:-ng, ng:-ng], nri[:-1],  nri[1:], ncj[:-1], ncj[1:], pids)

    fill_boundaries(urms, axis=1, number=ng)
    fill_boundaries(urms, axis=2, number=ng)
    fill_boundaries(vrms, axis=1, number=ng)
    fill_boundaries(vrms, axis=2, number=ng)
    fill_boundaries(wrms, axis=1, number=ng)
    fill_boundaries(wrms, axis=2, number=ng)
    fill_boundaries(thetav, axis=1, number=ng)
    fill_boundaries(thetav, axis=2, number=ng)
    fill_boundaries(rho, axis=0, number=ng)
    fill_boundaries(rho, axis=1, number=ng)
    fill_boundaries(rho, axis=2, number=ng)
    fill_boundaries(qv, axis=1, number=ng)
    fill_boundaries(qv, axis=2, number=ng)

    update_bnds(mpicomm, u, type='u')
    update_bnds(mpicomm, v, type='v')
    update_bnds(mpicomm, w, type='w')
    update_bnds(mpicomm, urms, type='u')
    update_bnds(mpicomm, vrms, type='v') 
    update_bnds(mpicomm, wrms, type='w')
    update_bnds(mpicomm, thetav, type='c')    
    update_bnds(mpicomm, rho, type='c')
    update_bnds(mpicomm, qv, type='c')
    
    for fld in tr_flds:
        update_bnds(mpicomm, fld, type='c')

    for fld in tr_flds:
        update_bnds(mpicomm, fld, type='c')

    p_per = np.zeros_like(rho)
    flds_sub = [u, v, w, p_per, rho, thetav, qv] + tr_flds + [th_surf, qv_surf] + [urms, vrms, wrms]
    
    return flds_sub


def avg_grid_flds_stag():
    """
    Interpolates the grid fields 
    on the  shifted grids used by
    the diffusion routine.
 
    grid_flds_stag... returned list of interpolated and original grid fields 
    """    

    eps = 1e-100

    global area_eff_x, area_eff_y, area_eff_z, vols_eff
    global area_x, area_y, area_z, vols
    global ffx, ffy, ffz
    global ffx_j, ffy_j, ffz_j
    global ffx_i, ffy_i, ffz_i
    global ffx_k, ffy_k, ffz_k

#    area_eff_xj = 0.5 * (area_eff_x[:, :, 1:] + area_eff_x[:, :, :-1])
    area_eff_xj = np.minimum(area_eff_x[:, :, 1:], area_eff_x[:, :, :-1])
    area_eff_xi = 0.5 * (area_eff_x[:, 1:] +  area_eff_x[:, :-1])
    area_eff_xk = 0.5 * (area_eff_x[1:] +  area_eff_x[:-1])
    area_eff_yj = 0.5 * (area_eff_y[:, :, 1:] + area_eff_y[:, :, :-1])
#    area_eff_yi = 0.5 * (area_eff_y[:, 1:] + area_eff_y[:, :-1])
    area_eff_yi = np.minimum(area_eff_y[:, 1:], area_eff_y[:, :-1])
    area_eff_yk = 0.5 * (area_eff_y[1:] + area_eff_y[:-1])
    area_eff_zi = 0.5 * (area_eff_z[:, 1:] + area_eff_z[:, :-1])
    area_eff_zj = 0.5 * (area_eff_z[:, :, 1:] + area_eff_z[:, :, :-1])
#    area_eff_zk = 0.5 * (area_eff_z[1:] + area_eff_z[:-1])
    area_eff_zk = np.minimum(area_eff_z[1:], area_eff_z[:-1])

    vols_eff_j = 0.5 * (vols_eff[:, :, 1:] + vols_eff[:, :, :-1])
    vols_eff_i = 0.5 * (vols_eff[:, 1:] + vols_eff[:, :-1])
    vols_eff_k = 0.5 * (vols_eff[1:] + vols_eff[:-1])

    vols_j = 0.5 * (vols[:, :, 1:] + vols[:, :, :-1])
    vols_i = 0.5 * (vols[:, 1:] + vols[:, :-1])
    vols_k = 0.5 * (vols[1:] + vols[:-1])

    ffx = area_eff_x / area_x
    ffy = area_eff_y / area_y
    ffz = area_eff_z / area_z
    
    ffx_j = 0.5 * (ffx[:, :, 1:] + ffx[:, :, :-1])
    ffy_j = 0.5 * (ffy[:, :, 1:] + ffy[:, :, :-1])
    ffz_j = 0.5 * (ffz[:, :, 1:] + ffz[:, :, :-1])

    ffx_i = 0.5 * (ffx[:, 1:] + ffx[:, :-1])
    ffy_i = 0.5 * (ffy[:, 1:] + ffy[:, :-1])
    ffz_i = 0.5 * (ffz[:, 1:] + ffz[:, :-1])

    ffx_k = 0.5 * (ffx[1:] + ffx[:-1])
    ffy_k = 0.5 * (ffy[1:] + ffy[:-1])
    ffz_k = 0.5 * (ffz[1:] + ffz[:-1])

    dx = vols / area_x[:, :, 1:]
    dy = vols / area_y[:, 1:]

    grid_flds_stag = [    
                         area_eff_x, area_eff_xj, area_eff_xi, area_eff_xk, 
                         area_eff_y, area_eff_yj, area_eff_yi, area_eff_yk, 
                         area_eff_z, area_eff_zj, area_eff_zi, area_eff_zk,
                         vols_eff, vols_eff_j, vols_eff_i, vols_eff_k,
                         vols, vols_j, vols_i, vols_k, 
                         area_x, area_y, area_z,
                         dx, dy
                     ]

    return grid_flds_stag


def avg_vel_flds_stag(mpicomm, u, v, w, param_dict):
    """
    Interpolates the velocity fields 
    on the  shifted grids used by
    some routines within this script.


    mpicomm... mpi-communicator
    u, v, w... velocity fields defined on the standard grid faces
    param_dict... parameter dictionary
    """

    global vel_flds_stag
    global terrain    
    global grid_flds_stag
    global fluxes
    global ng

    ng1 = ng - 1

    hsurf, dsurfdx, dsurfdy = terrain[:]

    area_eff_x, area_eff_xj, area_eff_xi, area_eff_xk = grid_flds_stag[:4]
    area_eff_y, area_eff_yj, area_eff_yi, area_eff_yk = grid_flds_stag[4:8]
    area_eff_z, area_eff_zj, area_eff_zi, area_eff_zk = grid_flds_stag[8:12]
    vols_eff, vols_eff_j, vols_eff_i, vols_eff_k  = grid_flds_stag[12:16]

    shp = vols_eff[ng:-ng, ng:-ng, ng:-ng].shape

    u_ = u
    u_k = 0.5 * (u[1:] + u[:-1])
    u_i = 0.5 * (u[:, 1:] + u[:, :-1])
    u_j = 0.5 * (u[:, :, 1:] + u[:, :, :-1])
    
    v_ = v
    v_j = 0.5 * (v[:, :, 1:] + v[:, :, :-1])
    v_i = 0.5 * (v[:, 1:] + v[:, :-1])
    v_k = 0.5 * (v[1:] + v[:-1])

    w_contra = w.copy()    

    w_x_c = 0.5 * (u[:, :, 1:] * dsurfdx[:, :, 1:] + u[:, :, :-1] * dsurfdx[:, :, :-1])
    w_y_c = 0.5 * (v[:, 1:] * dsurfdy[:, 1:] + v[:, :-1] * dsurfdy[:, :-1])
    w_x = 0.5 * (w_x_c[1:, 1:-1, 1:-1] + w_x_c[:-1, 1:-1, 1:-1])
    w_y = 0.5 * (w_y_c[1:, 1:-1, 1:-1] + w_y_c[:-1, 1:-1, 1:-1])
    w_x[ng] = 0.0
    w_y[ng] = 0.0
 
    w_contra[1:-1, 1:-1, 1:-1] -= w_x + w_y
    
    ddcp.cptobounds(w_contra, bnds_expl_w_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_w_s, bnds_expl_w_r)
    ddcp.cpfrombounds(w_contra, bnds_expl_w_r, mode = 'repl')   
    
    w_ = w       
    w_j = 0.5 * (w[:, :, 1:] + w[:, :, :-1])
    w_i = 0.5 * (w[:, 1:] + w[:, :-1])
    w_k = 0.5 * (w[1:] + w[:-1])

    u_j_sq = u_j ** 2
    v_i_sq = v_i ** 2
    w_k_sq = w_k ** 2

    speed = np.sqrt(u_j_sq + v_i_sq + w_k_sq)
    speed_uv = np.sqrt(u_j_sq + v_i_sq)
    speed_uw = np.sqrt(u_j_sq + w_k_sq)
    speed_vw = np.sqrt(v_i_sq + w_k_sq)

    vel_flds_stag = [u_, u_j, u_i, u_k, v_, v_j, v_i, v_k, w_, w_j, w_i, w_k, speed, speed_uv, speed_uw, speed_vw, w_contra]

    fluxes = [u * area_eff_x, v * area_eff_y, w_contra * area_eff_z]



def strain(param_dict):
    """
    Derives the velocity derivatives and strain rates.
    Stores the computed fields in a global list accessible 
    by all routines within this script.

    param_dict... parameter dictionary
    """

    global vel_flds_stag
    global grid_flds_stag
    global ng
    global fluxes

    dh = param_dict['dh']
    dz = param_dict['dz']
    nz = param_dict['zcoord'].size
    
    dz_h = np.empty([nz + 2 * ng, 1, 1])
    dz_h[:ng] = dz[0]
    dz_h[ng:-ng] = dz.reshape(dz.size, 1, 1)          
    dz_h[-ng:] = dz[-1]
    dz_c = 0.5 * (dz_h[1:] + dz_h[:-1])
    

    area_eff_x, area_eff_xj, area_eff_xi, area_eff_xk = grid_flds_stag[:4]
    area_eff_y, area_eff_yj, area_eff_yi, area_eff_yk = grid_flds_stag[4:8]
    area_eff_z, area_eff_zj, area_eff_zi, area_eff_zk = grid_flds_stag[8:12]
    vols_eff, vols_eff_j, vols_eff_i, vols_eff_k  = grid_flds_stag[12:16]

    u, u_j, u_i, u_k, v, v_j, v_i, v_k, w, w_j, w_i, w_k = vel_flds_stag[:12]

    uf, vf, wf = fluxes[:]
  
    nr = area_eff_x.shape[1] - 2 * ng
    nc = area_eff_y.shape[2] - 2 * ng


    n = ng - 1
    nl = ng - 2
    nch = ng + nc + 2
    nrh = ng + nr + 2
    nzh = ng + nz + 2    


    eps = 1e-20

    dudx = (uf[n:-n, n:-n, n:nch + 1] - uf[n:-n, n:-n, nl:nch]) / (vols_eff[n:-n, n:-n, nl:nch] + eps)
    dudy = (
               2.0 * (u[n:-n, n:nrh, n:-n] - u[n:-n, nl:-n, n:-n] ) * area_eff_yj[n:-n, n:-n, nl:nch - 1] / 
               (vols_eff_j[n:-n, n:nrh, nl:nch - 1] + vols_eff_j[n:-n, nl:-n, nl:nch - 1] + eps) 
           )
    dudz = (
               2.0 * (u[n:nzh, n:-n, n:-n] - u[nl:-n, n:-n, n:-n]) * area_eff_zj[n:-n, n:-n, nl:nch - 1] / 
               (vols_eff_j[n:nzh, n:-n, nl:nch - 1] + vols_eff_j[nl:-n, n:-n, nl:nch - 1] + eps)
           )
    dvdx = (
               2.0 * (v[n:-n, n:-n, n:nch] - v[n:-n, n:-n, nl:-n]) * area_eff_xi[n:-n, nl:nrh - 1, n:-n] / 
               (vols_eff_i[n:-n, nl:nrh - 1, n:nch] + vols_eff_i[n:-n, nl:nrh - 1, nl:-n] + eps)
           )
    dvdy = (vf[n:-n, n:nrh + 1, n:-n] - vf[n:-n, nl:-n, n:-n]) / (vols_eff[n:-n, nl:nrh, n:-n] + eps)
    dvdz = (
               2.0 * (v[n:nzh, n:-n, n:-n] - v[nl:-n, n:-n, n:-n]) * area_eff_zi[n:-n, nl:nrh - 1, n:-n] / 
               (vols_eff_i[n:nzh, nl:nrh - 1, n:-n] + vols_eff_i[nl:-n, nl:nrh - 1, n:-n] + eps)
           )

    dwdx = (
               2.0 * (w[n:-n, n:-n, n:nch] - w[n:-n, n:-n, nl:-n]) * area_eff_xk[nl:nzh - 1, n:-n, n:-n] / 
               (vols_eff_k[nl:nzh - 1, n:-n, n:nch] + vols_eff_k[nl:nzh - 1, n:-n, nl:-n] + eps)
           )
    dwdy = (
               2.0 * (w[n:-n, n:nrh, n:-n] - w[n:-n, nl:-n, n:-n]) * area_eff_yk[nl:nzh - 1, n:-n, n:-n] / 
               (vols_eff_k[nl:nzh - 1, n:nrh, n:-n] + vols_eff_k[nl:nzh - 1, nl:-n, n:-n] + eps)
           )
    dwdz = (
               (w[n:nzh + 1, n:-n, n:-n] * area_eff_z[n:nzh + 1, n:-n, n:-n] - w[nl:-n, n:-n, n:-n] * area_eff_z[nl:-n, n:-n, n:-n]) / 
               (vols_eff[nl:nzh, n:-n, n:-n] + eps)
           )

    sxy = 0.5 * (dudy + dvdx)
    syz = 0.5 * (dvdz + dwdy)
    szx = 0.5 * (dwdx + dudz)

    s_abs_z = np.sqrt(
                          (0.25 * (dudx[:, 1:, 2:-1] + dudx[:, 1:, 1:-2] + dudx[:, :-1, 2:-1] + dudx[:, :-1, 1:-2])) ** 2 + 
                          (0.25 * (dvdy[:, 2:-1, 1:] + dvdy[:, 2:-1, :-1] + dvdy[:, 1:-2, 1:] + dvdy[:, 1:-2, :-1])) ** 2 +
                          (0.25 * (dwdz[1:-1, 1:, 1:] + dwdz[1:-1, 1:, :-1] + dwdz[1:-1, :-1, 1:] + dwdz[1:-1, :-1, :-1])) ** 2 +
                          2.0 * sxy[:, 1:-1, 1:-1] ** 2 + 
                          2.0 * (0.25 * (syz[1:, 1:-1, 1:] + syz[1:, 1:-1, :-1] + syz[:-1, 1:-1, 1:] + syz[:-1, 1:-1, :-1])) ** 2 + 
                          2.0 * (0.25 * (szx[1:, 1:, 1:-1] + szx[:-1, 1:, 1:-1] + szx[1:, :-1, 1:-1] + szx[:-1, :-1, 1:-1])) ** 2
                     ) * np.sqrt(2.0) 


    s_abs_x = np.sqrt(
                         (0.25 * (dudx[1:, 1:, 1:-1] + dudx[1:, :-1, 1:-1] + dudx[:-1, 1:, 1:-1] + dudx[:-1, :-1, 1:-1])) ** 2 +
                         (0.25 * (dvdy[1:, 2:-1, :] + dvdy[1:, 1:-2, :] + dvdy[:-1, 2:-1, :] + dvdy[:-1, 1:-2, :])) ** 2 +
                         (0.25 * (dwdz[2:-1, 1:, :] + dwdz[2:-1, :-1, :] + dwdz[1:-2, 1:, :] + dwdz[1:-2, :-1, :])) ** 2 +
                         2.0 * syz[1:-1, 1:-1] ** 2 + 
                         2.0 * (0.25 * (szx[1:-1, 1:, 1:] + szx[1:-1, :-1, 1:] + szx[1:-1, 1:, :-1] + szx[1:-1, :-1, :-1])) ** 2 + 
                         2.0 * (0.25 * (sxy[1:, 1:-1, 1:] + sxy[1:, 1:-1, :-1] + sxy[:-1, 1:-1, 1:] + sxy[:-1, 1:-1, :-1])) ** 2
                     ) * np.sqrt(2.0)

    s_abs_y = np.sqrt(
                         (0.25 * (dudx[1:, :, 2:-1] + dudx[1:, :, 1:-2] + dudx[:-1, :, 2:-1] + dudx[:-1, :, 1:-2])) ** 2 +
                         (0.25 * (dvdy[1:, 1:-1, 1:] + dvdy[1:, 1:-1, :-1] + dvdy[:-1, 1:-1, 1:] + dvdy[:-1, 1:-1, :-1])) ** 2 +
                         (0.25 * (dwdz[2:-1, :, 1:] + dwdz[2:-1, :, :-1] + dwdz[1:-2, :, 1:] + dwdz[1:-2, :, :-1])) ** 2 +
                         2.0 * szx[1:-1, :, 1:-1] ** 2 + 
                         2.0 * (0.25 * (sxy[1:, 1:, 1:-1] + sxy[:-1, 1:, 1:-1] + sxy[1:, :-1, 1:-1] + sxy[:-1, :-1, 1:-1])) ** 2 + 
                         2.0 * (0.25 * (syz[1:-1, 1:, 1:] + syz[1:-1, 1:, :-1] + syz[1:-1, :-1, 1:] + syz[1:-1, :-1, :-1])) ** 2
                     ) * np.sqrt(2.0)
    
    s_abs_c = np.sqrt(
                         dudx[:, :, 1:-1] ** 2 + dvdy[:, 1:-1] ** 2 + dwdz[1:-1] ** 2 +
                         2.0 * (0.25 * (sxy[:, 1:, 1:] + sxy[:, 1:, :-1] + sxy[:, :-1, 1:] + sxy[:, :-1, :-1])) ** 2 + 
                         2.0 * (0.25 * (syz[1:, 1:] + syz[1:, :-1] + syz[:-1, 1:] + syz[:-1, :-1])) ** 2 +
                         2.0 * (0.25 * (szx[1:, :, 1:] + szx[1:, :, :-1] + szx[:-1, :, 1:] + szx[:-1, :, :-1])) ** 2
                     ) * np.sqrt(2.0)    

    global strain_flds

    strain_flds = [dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, sxy, syz, szx, s_abs_z, s_abs_x, s_abs_y, s_abs_c, dh, dz_h, dz_c]



def derive_lmix(param_dict):
    """
    Computes the static mixing lengths 
    at model initialization and stores the fields.

    param_dict... parameter dictionary

    lmix_h, lmix_v, lmix_hv... returned mixing lengths for horizontal and vertical mixing.
    """


    global grid_fld_stag
    global frict_data 
    global ng

    deff_v, deff_hx, deff_hy = frict_data[:3]

    area_eff_x, area_eff_xj, area_eff_xi, area_eff_xk = grid_flds_stag[:4]
    area_eff_y, area_eff_yj, area_eff_yi, area_eff_yk = grid_flds_stag[4:8]
    area_eff_z, area_eff_zj, area_eff_zi, area_eff_zk = grid_flds_stag[8:12]
    vols_eff, vols_eff_j, vols_eff_i, vols_eff_k = grid_flds_stag[12:16]
    dx, dy = grid_flds_stag[23:25]

    nz = vols_eff.shape[0] - 2 * ng
    nzh = nz + ng + 2
  

    dz = np.zeros_like(area_eff_zk) 
    dz[ng:-ng] = param_dict['dz'].reshape(param_dict['dz'].size, 1, 1)
    dz[:ng] = param_dict['dz'][0]
    dz[-ng:] = param_dict['dz'][-1]

    lmix_hx = np.maximum(np.minimum(vols_eff / (area_eff_xj + 1e-50), dx), 1e-3)
    lmix_hy = np.maximum(np.minimum(vols_eff / (area_eff_yi + 1e-50), dy), 1e-3)
    lmix_v = np.maximum(np.minimum(vols_eff / (area_eff_zk + 1e-50), dz), 1e-3)

    c_smag = param_dict['c_smag']    

    lmix_hx[ng:nzh] = np.minimum(1.8 * np.minimum(deff_hx[ng:nzh], deff_v[ng:nzh]), lmix_hx[ng:nzh])
    lmix_hy[ng:nzh] = np.minimum(1.8 * np.minimum(deff_hy[ng:nzh], deff_v[ng:nzh]), lmix_hy[ng:nzh])
    lmix_v[ng:nzh] = np.minimum(1.8 * np.minimum(0.5 * (deff_hx[ng:nzh] + deff_hy[ng:nzh]), deff_v[ng:nzh]), lmix_v[ng:nzh])

    lmix_h = np.sqrt(lmix_hx * lmix_hy)
    lmix_hv = np.sqrt(lmix_h * lmix_v) 
    lmix_h = lmix_h * c_smag
    lmix_v = lmix_v * c_smag 
    lmix_hv = lmix_hv * c_smag 

    lmix_v[:ng + 4] = lmix_v[:ng + 4] * param_dict['mag_vdiff']
    lmix_hv[:ng + 4] = lmix_hv[:ng + 4] * np.sqrt(param_dict['mag_vdiff'])

    return lmix_h, lmix_v, lmix_hv


def diffusion(field, type='c', ktype='mom'):
    """
    A standard Smagorinsky subscale turbulence model.

    field... field to apply diffusion
    type... field type ('c': volume centred; 'u', 'v', 'w': area centred)
    ktype... type of diffusion coefficient (mom: momentum diffusion, heat: scalar diffusion)

    df... returned diffusive tendencies
    turb... returned subscale turbulent intensities
    """

    global strain_flds
    dudx, dudy, dudz, dvdx, dvdy, dvdz, dwdx, dwdy, dwdz, sxy, syz, szx, s_abs_z, s_abs_x, s_abs_y, s_abs_c, dh, dz_h, dz_c = strain_flds[:]

    global grid_fld_stag
    area_eff_x, area_eff_xj, area_eff_xi, area_eff_xk = grid_flds_stag[:4]
    area_eff_y, area_eff_yj, area_eff_yi, area_eff_yk = grid_flds_stag[4:8]
    area_eff_z, area_eff_zj, area_eff_zi, area_eff_zk = grid_flds_stag[8:12]
    vols_eff, vols_eff_j, vols_eff_i, vols_eff_k  = grid_flds_stag[12:16]

    global vel_flds
    u, v, w = vel_flds[:]

    global lmix_h, lmix_v, lmix_hv

    global thetav

    global ng    
    
    df = np.zeros_like(field)

    if ktype == 'mom':
        kfac = 1.0
    elif  ktype == 'heat':
        kfac = 3.0 / 2.0 
    else:
        print "Diffusion types can only be 'mom' for momentum diffusion or 'heat' for heat diffusion"
        raise ValueError

    ng1 = ng - 1 
    ng2 = ng + 1

    
    if type == 'c':

        thetav_k = np.zeros_like(w)
        thetav_k[1:-1] = 0.5 * (thetav[1:] + thetav[:-1])
        fill_boundaries(thetav_k, 0, 1)
        dthetavdz = (thetav_k[1:] - thetav_k[:-1]) / dz_h        

        dcdz = (field[ng:-ng1, ng:-ng, ng:-ng] - field[ng1:-ng, ng:-ng, ng:-ng]) * area_eff_z[ng:-ng, ng:-ng, ng:-ng] / (vols_eff_k[ng1:-ng1, ng:-ng, ng:-ng] + 1e-20)
        dcdy = (field[ng:-ng, ng:-ng1, ng:-ng] - field[ng:-ng, ng1:-ng, ng:-ng]) * area_eff_y[ng:-ng, ng:-ng, ng:-ng] / (vols_eff_i[ng:-ng, ng1:-ng1, ng:-ng] + 1e-20)
        dcdx = (field[ng:-ng, ng:-ng, ng:-ng1] - field[ng:-ng, ng:-ng, ng1:-ng]) * area_eff_x[ng:-ng, ng:-ng, ng:-ng] / (vols_eff_j[ng:-ng, ng:-ng, ng1:-ng1] + 1e-20)

        N_c = 9.81 * dthetavdz / (thetav + 1e-20)
        ri_c = N_c[ng1:-ng1, ng1:-ng1, ng1:-ng1] / (s_abs_c ** 2 + 1e-20)

        if ktype == 'mom':
            f = (1.0 - ri_c / 0.25) ** 4.0
            f[ri_c<0.0] = np.sqrt(1.0 - 16.0 * ri_c[ri_c<0.0])
            f[ri_c>=0.25] = 0.0
        elif ktype == 'heat':
            f = (1.0 - ri_c / 0.25) ** 4.0 * ( 1.0 - 1.2 * ri_c) * 3.0 / 2.0
            f[ri_c<0.0] = np.sqrt(1.0 - 40.0 * ri_c[ri_c<0.0]) * 3.0 / 2.0
            f[ri_c>=0.25] = 0.0

        s_abs_c_f = s_abs_c * f * kfac

        a = dcdx * 0.5 * (s_abs_c_f[1:-1, 1:-1, 1:] + s_abs_c_f[1:-1, 1:-1, :-1]) * (0.5 * (lmix_h[ng:-ng, ng:-ng, ng1:-ng] + lmix_h[ng:-ng, ng:-ng, ng:-ng1])) ** 2 
        b = dcdy * 0.5 * (s_abs_c_f[1:-1, 1:, 1:-1] + s_abs_c_f[1:-1, :-1, 1:-1]) * (0.5 * (lmix_h[ng:-ng, ng1:-ng, ng:-ng] + lmix_h[ng:-ng, ng:-ng1, ng:-ng])) ** 2 
        c = dcdz * 0.5 * (s_abs_c_f[1:, 1:-1, 1:-1] + s_abs_c_f[:-1, 1:-1, 1:-1]) * (0.5 * (lmix_v[ng1:-ng, ng:-ng, ng:-ng] + lmix_v[ng:-ng1, ng:-ng, ng:-ng])) ** 2

        da = a[:, :, 1:] * area_eff_x[ng:-ng, ng:-ng, ng2:-ng] - a[:, :, :-1] * area_eff_x[ng:-ng, ng:-ng, ng:-ng2]
        db = b[:, 1:] * area_eff_y[ng:-ng, ng2:-ng, ng:-ng] - b[:, :-1] * area_eff_y[ng:-ng, ng:-ng2, ng:-ng]
        dc = c[1:] * area_eff_z[ng2:-ng, ng:-ng, ng:-ng] - c[:-1] * area_eff_z[ng:-ng2, ng:-ng, ng:-ng]

        turb = None
        df[ng:-ng, ng:-ng, ng:-ng] =  (da + db + dc) / (vols_eff[ng:-ng, ng:-ng, ng:-ng] + 1e-20)

    if type == 'u':

        
        thetav_k = np.zeros_like(w)        
        thetav_k[1:-1] = 0.5 * (thetav[1:] + thetav[:-1])
        min_th =  np.min(thetav[3:-3])
        fill_boundaries(thetav_k, 0, 1)
        dthetavdz = (thetav_k[1:] - thetav_k[:-1]) / dz_h
        N_c = 9.81 * dthetavdz / (thetav + 1e-20)
        ri_c = N_c[ng1:-ng1, ng1:-ng1, ng1:-ng1] / (s_abs_c ** 2 + 1e-20)
        f = (1.0 - ri_c / 0.25) ** 4.0
        f[ri_c<0.0] = np.sqrt(1.0 - 16.0 * ri_c[ri_c<0.0])
        f[ri_c>=0.25] = 0.0

        s_abs_c_f = s_abs_c * f *  kfac
        s_abs_z_f = s_abs_z * 0.25 * (f[:, 1:, 1:] + f[:, :-1, 1:] + f[:, 1:, :-1] + f[:, :-1, :-1]) * kfac
        s_abs_y_f = s_abs_y * 0.25 * (f[1:, :, 1:] + f[1:, :, :-1] + f[:-1, :, 1:] + f[:-1, :, :-1]) * kfac

        a = 2.0 * dudx[1:-1, 1:-1, 1:-1] * s_abs_c_f[1:-1, 1:-1, :] * lmix_h[ng:-ng, ng:-ng, ng1:-ng1] ** 2
        b = 2.0 * sxy[1:-1, 1:-1, 1:-1] * s_abs_z_f[1:-1] * (0.25 * (
                                                                        lmix_h[ng:-ng, ng1:-ng, ng1:-ng] + lmix_h[ng:-ng, ng1:-ng, ng:-ng1] + 
                                                                        lmix_h[ng:-ng, ng:-ng1, ng1:-ng] + lmix_h[ng:-ng, ng:-ng1, ng:-ng1]
                                                                    )) ** 2
        c = 2.0 * szx[1:-1, 1:-1, 1:-1] * s_abs_y_f[:, 1:-1] * (0.25 * (
                                                                           lmix_hv[ng1:-ng, ng:-ng, ng1:-ng] + lmix_hv[ng:-ng1, ng:-ng, ng1:-ng] + 
                                                                           lmix_hv[ng1:-ng, ng:-ng, ng:-ng1] + lmix_hv[ng:-ng1, ng:-ng, ng:-ng1]
                                                                       )) ** 2


        da = a[:, :, 1:] * area_eff_xj[ng:-ng, ng:-ng, ng:-ng1] - a[:, :, :-1] * area_eff_xj[ng:-ng, ng:-ng, ng1:-ng]
        db = b[:, 1:] * area_eff_yj[ng:-ng, ng2:-ng, ng1:-ng1]  - b[:, :-1] * area_eff_yj[ng:-ng, ng:-ng2, ng1:-ng1]
        dc = c[1:] * area_eff_zj[ng2:-ng, ng:-ng, ng1:-ng1] - c[:-1] * area_eff_zj[ng:-ng2, ng:-ng, ng1:-ng1]

        turb = np.zeros_like(u)
        turb[ng:-ng, ng:-ng, ng:-ng] = 0.5 * (a[:, :, 1:] + a[:, :, :-1]) 

        df[ng:-ng, ng:-ng, ng:-ng] = (da + db + dc) / (vols_eff_j[ng:-ng, ng:-ng, ng1:-ng1] + 1e-20)

    if type == 'v':

        thetav_k = np.zeros_like(w)
        thetav_k[1:-1] = 0.5 * (thetav[1:] + thetav[:-1])
        fill_boundaries(thetav_k, 0, 1)
        dthetavdz = (thetav_k[1:] - thetav_k[:-1]) / dz_h

        N_c = 9.81 * dthetavdz / (thetav + 1e-20)
        ri_c = N_c[ng1:-ng1, ng1:-ng1, ng1:-ng1] / (s_abs_c ** 2 + 1e-20)
        f = (1.0 - ri_c / 0.25) ** 4.0
        f[ri_c<0.0] = np.sqrt(1.0 - 16.0 * ri_c[ri_c<0.0])
        f[ri_c>=0.25] = 0.0

        s_abs_c_f = s_abs_c * f * kfac
        s_abs_z_f = s_abs_z * 0.25 * (f[:, 1:, 1:] + f[:, :-1, 1:] + f[:, 1:, :-1] + f[:, :-1, :-1]) * kfac
        s_abs_x_f = s_abs_x * 0.25 * (f[1:, 1:] + f[1:, :-1] + f[:-1, 1:] + f[:-1, :-1]) * kfac

        a = 2.0 * dvdy[1:-1, 1:-1, 1:-1] * s_abs_c_f[1:-1, :, 1:-1] * lmix_h[ng:-ng, ng1:-ng1, ng:-ng] ** 2
        b = 2.0 * sxy[1:-1, 1:-1, 1:-1] * s_abs_z_f[1:-1] * (0.25 * (
                                                                        lmix_h[ng:-ng, ng1:-ng, ng1:-ng] + lmix_h[ng:-ng, ng1:-ng, ng:-ng1] + 
                                                                        lmix_h[ng:-ng, ng:-ng1, ng1:-ng] + lmix_h[ng:-ng, ng:-ng1, ng:-ng1]
                                                                    )) ** 2 
        c = 2.0 * syz[1:-1, 1:-1, 1:-1] * s_abs_x_f[:, :, 1:-1] * (0.25 * (
                                                                              lmix_hv[ng1:-ng, ng1:-ng, ng:-ng] + lmix_hv[ng:-ng1, ng1:-ng, ng:-ng] + 
                                                                              lmix_hv[ng1:-ng, ng:-ng1, ng:-ng] + lmix_hv[ng:-ng1, ng:-ng1, ng:-ng]
                                                                          )) ** 2

        da = a[:, 1:] * area_eff_yi[ng:-ng, ng:-ng1, ng:-ng] - a[:, :-1] * area_eff_yi[ng:-ng, ng1:-ng, ng:-ng]
        db = b[:, :, 1:] * area_eff_xi[ng:-ng, ng1:-ng1, ng2:-ng] - b[:, :, :-1] * area_eff_xi[ng:-ng, ng1:-ng1, ng:-ng2]
        dc = c[1:] * area_eff_zi[ng2:-ng, ng1:-ng1, ng:-ng] - c[:-1] * area_eff_zi[ng:-ng2, ng1:-ng1, ng:-ng]


        turb = np.zeros_like(v)
        turb[ng:-ng, ng:-ng, ng:-ng] = 0.5 * (a[:, 1:] + a[:, :-1])
        df[ng:-ng, ng:-ng, ng:-ng] = (da + db + dc) / (vols_eff_i[ng:-ng, ng1:-ng1, ng:-ng] + 1e-20)

    if type == 'w':

        thetav_k = np.zeros_like(w)
        thetav_k[1:-1] = 0.5 * (thetav[1:] + thetav[:-1])
        fill_boundaries(thetav_k, 0, 1)
        dthetavdz = (thetav_k[1:] - thetav_k[:-1]) / dz_h
        
        N_c = 9.81 * dthetavdz / (thetav + 1e-20)
        ri_c = N_c[ng1:-ng1, ng1:-ng1, ng1:-ng1] / (s_abs_c ** 2 + 1e-20)
        f = (1.0 - ri_c / 0.25) ** 4.0
        f[ri_c<0.0] = np.sqrt(1.0 - 16.0 * ri_c[ri_c<0.0])
        f[ri_c>=0.25] = 0.0

        s_abs_c_f = s_abs_c * f * kfac
        s_abs_y_f = s_abs_y * 0.25 * (f[1:, :, 1:] + f[1:, :, :-1] + f[:-1, :, 1:] + f[:-1, :, :-1]) * kfac
        s_abs_x_f = s_abs_x * 0.25 * (f[1:, 1:] + f[1:, :-1] + f[:-1, 1:] + f[:-1, :-1]) * kfac

        a = 2.0 * dwdz[1:-1, 1:-1, 1:-1] * s_abs_c_f[:, 1:-1, 1:-1] * lmix_v[ng1:-ng1, ng:-ng, ng:-ng] ** 2
        b = 2.0 * szx[1:-1, 1:-1, 1:-1] * s_abs_y_f[:, 1:-1] * (0.25 * (
                                                                           lmix_hv[ng1:-ng, ng:-ng, ng1:-ng] + lmix_hv[ng1:-ng, ng:-ng, ng:-ng1] + 
                                                                           lmix_hv[ng:-ng1, ng:-ng, ng1:-ng] + lmix_hv[ng:-ng1, ng:-ng, ng:-ng1]
                                                                       )) ** 2 
        c = 2.0 * syz[1:-1, 1:-1, 1:-1] * s_abs_x_f[:, :, 1:-1] * (0.25 * (
                                                                               lmix_hv[ng1:-ng, ng1:-ng, ng:-ng] + lmix_hv[ng1:-ng, ng:-ng1, ng:-ng] + 
                                                                               lmix_hv[ng:-ng1, ng1:-ng, ng:-ng] + lmix_hv[ng:-ng1, ng:-ng1, ng:-ng]
                                                                          )) ** 2 

        da = a[1:] * area_eff_zk[ng:-ng1, ng:-ng, ng:-ng] - a[:-1] * area_eff_zk[ng1:-ng, ng:-ng, ng:-ng]
        db = b[:, :, 1:] * area_eff_xk[ng1:-ng1, ng:-ng, ng2:-ng] - b[:, :, :-1] * area_eff_xk[ng1:-ng1, ng:-ng, ng:-ng2]
        dc = c[:, 1:] * area_eff_yk[ng1:-ng1, ng2:-ng, ng:-ng] - c[:, :-1] * area_eff_yk[ng1:-ng1, ng:-ng2, ng:-ng]

        df[ng:-ng, ng:-ng, ng:-ng] = (da + db + dc) / (vols_eff_k[ng1:-ng1, ng:-ng, ng:-ng] + 1e-20)

        turb = np.zeros_like(w)
        turb[ng:-ng, ng:-ng, ng:-ng] = 0.5 * (a[1:] + a[:-1])        

    return df, turb


def advect_momentum(mpicomm, u, v, w, fluxcon=True):
    """
    A finite volume scheme for momentum advection.
    Reconstructions can be of arbitrary odd order and either be upwind or WENO.    
    For each velocity component, two cell-centred scalars are advected and interpolated to 
    re-obtain the tendency on the cell face.

    mpicomm... MPI communicator
    u, v, w... velocity components to advect
    fluxcon... use flux-conservative form DIV(u * c) (the subtraction of the velocity-divergence term  c * DIV(u) is omitted)
    
    du, dv, dw... returned advective tendencies
    """

    global grid_flds_stag
    global rec_type
    global wghtxl_lim, wghtyl_lim, wghtzl_lim
    global wghtxr_lim, wghtyr_lim, wghtzr_lim
    global inds_lim_xl, inds_lim_yl, inds_lim_zl
    global inds_lim_xr, inds_lim_yr, inds_lim_zr

    global ng
    ng1 = ng - 1

    eps = 1e-40
    
    du_l = advect_scalar(
                             u[:, :, :-1], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim, 
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,  
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )
    du_r = advect_scalar(
                             u[:, :, 1:], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim, 
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim, 
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )
    dv_l = advect_scalar(
                             v[:, :-1], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim,
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )
    dv_r = advect_scalar(
                             v[:, 1:], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim,
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )
    dw_l = advect_scalar(
                             w[:-1], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim,
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )
    dw_r = advect_scalar(
                             w[1:], fluxcon=fluxcon, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim,
                             wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,
                             inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                             inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                        )


    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dw = np.zeros_like(w)

    du[ng:-ng, ng:-ng, ng:-ng] = (
                                     (du_l[ng:-ng, ng:-ng, ng:-ng1] * vols_eff[ng:-ng, ng:-ng, ng:-ng1] + 
                                      du_r[ng:-ng, ng:-ng, ng1:-ng] * vols_eff[ng:-ng, ng:-ng, ng1:-ng]) / 
                                     (vols_eff[ng:-ng, ng:-ng, ng1:-ng] + vols_eff[ng:-ng, ng:-ng, ng:-ng1] + eps)
                                 )
    dv[ng:-ng, ng:-ng, ng:-ng] = (
                                     (dv_l[ng:-ng, ng:-ng1, ng:-ng] * vols_eff[ng:-ng, ng:-ng1, ng:-ng] + 
                                      dv_r[ng:-ng, ng1:-ng, ng:-ng] * vols_eff[ng:-ng, ng1:-ng, ng:-ng]) / 
                                     (vols_eff[ng:-ng, ng1:-ng, ng:-ng] + vols_eff[ng:-ng, ng:-ng1, ng:-ng] + eps)
                                 ) 
    dw[ng:-ng, ng:-ng, ng:-ng] = (
                                     (dw_l[ng:-ng1, ng:-ng, ng:-ng] * vols_eff[ng:-ng1, ng:-ng, ng:-ng] + 
                                      dw_r[ng1:-ng, ng:-ng, ng:-ng] * vols_eff[ng1:-ng, ng:-ng, ng:-ng]) / 
                                     (vols_eff[ng1:-ng, ng:-ng, ng:-ng] + vols_eff[ng:-ng1, ng:-ng, ng:-ng] + eps)

                                 )

    update_bnds(mpicomm, du, type='c_x', mode='add')    
    update_bnds(mpicomm, dv, type='c_y', mode='add')
    update_bnds(mpicomm, dw, type='c_z', mode='add')    

    return du, dv, dw



def advect_scalar(
                     c, fluxcon=True, wghtxl_lim=0.0, wghtyl_lim=0.0, wghtzl_lim=0.0, 
                     wghtxr_lim=0.0, wghtyr_lim=0.0, wghtzr_lim=0.0, 
                     inds_lim_xl=None, inds_lim_yl=None, inds_lim_zl=None,
                     inds_lim_xr=None, inds_lim_yr=None, inds_lim_zr=None
                 ):
    """
    An odd order upwind-biased advection scheme
    considering the effective cell geometry. 

    c... field to advect
    type... field type to advect
    fluxcon... use flux-conservative form DIV(u * c) (the subtraction of the velocity-divergence term  c * DIV(u) is omitted)
    wghtxl_lim, wghtyl_lim, wghtzl_lim... obstacle specific flux-limiter weights for positive upwind reconstruction
    wghtxr_lim, wghtyr_lim, wghtzr_lim... obstacle specific flux-limiter weights for negative upwind reconstruction
    inds_lim_xl, inds_lim_yl, inds_lim_zl... lists of field indices where to apply flux limiter
    inds_lim_xr, inds_lim_yr, inds_lim_zr... lists of field indices where to apply flux limiter

    dc... returned advective tendency of scalar c
    """

    global grid_flds_stag
    global rec_type
    global fluxes

    global ng

    vols_eff = grid_flds_stag[12]

    c_lx, c_ly, c_lz, c_rx, c_ry, c_rz = globals()[rec_type + '_recon_scalar'](c, wghtxl_lim=wghtxl_lim, wghtyl_lim=wghtyl_lim, wghtzl_lim=wghtzl_lim, 
                                                                                  wghtxr_lim=wghtxr_lim, wghtyr_lim=wghtyr_lim, wghtzr_lim=wghtzr_lim,
                                                                                  inds_lim_xl=inds_lim_xl, inds_lim_yl=inds_lim_yl, inds_lim_zl=inds_lim_zl,
                                                                                  inds_lim_xr=inds_lim_xr, inds_lim_yr=inds_lim_yr, inds_lim_zr=inds_lim_zr,
                                                                              )[:]
    uf, vf, wf = fluxes[:]   
    
    uf_c = uf[ng:-ng, ng:-ng, ng:-ng]
    vf_c = vf[ng:-ng, ng:-ng, ng:-ng] 
    wf_c = wf[ng:-ng, ng:-ng, ng:-ng]
    vols_eff_crop = vols_eff[ng:-ng, ng:-ng, ng:-ng]
    
    cfu = 0.5 * (uf_c * c_lx + uf_c * c_rx - np.absolute(uf_c) * (c_rx - c_lx))
    cfv = 0.5 * (vf_c * c_ly + vf_c * c_ry - np.absolute(vf_c) * (c_ry - c_ly))
    cfw = 0.5 * (wf_c * c_lz + wf_c * c_rz - np.absolute(wf_c) * (c_rz - c_lz))

    #central  (only experimental) 
#    cfu = 0.5 * (uf_c * c_lx + uf_c * c_rx)
#    cfv = 0.5 * (vf_c * c_ly + vf_c * c_ry)
#    cfw = 0.5 * (wf_c * c_lz + wf_c * c_rz)

    dc = np.zeros_like(c)

    dc[ng:-ng, ng:-ng, ng:-ng] =   (
                                      cfu[:, :, :-1]  - cfu[:, :, 1:] + 
                                      cfv[:, :-1] - cfv[:, 1:] + 
                                      cfw[:-1] - cfw[1:]               
                             ) / vols_eff_crop
    
    if not fluxcon:         
        div = (
                  uf_c[:, :, 1:] - uf_c[:, :, :-1] + 
                  vf_c[:, 1:]  - vf_c[:, :-1] + 
                  wf_c[1:] - wf_c[:-1]
              ) / vols_eff_crop                        
        
        dc[ng:-ng, ng:-ng, ng:-ng] += c[ng:-ng, ng:-ng, ng:-ng] * div                        

    return dc


def buoyancy(comm, w, thetav, param_dict):
    """
    Calculates the buoyant tendency based on
    the Boussinesq approximation.
    The horizontal averaging of thetav is carried out
    on z=const planes, which requires remapping of the computation 
    grid using terrain-following coordinates.
    
    comm... ddcp.communicator
    w... vertical velocity field
    thetav... virtual potential temperature field
    param_dict... parameter dictionary
    
    dw... returned buoyant tendency
    """

    global thetav_bnds_s, thetav_bnds_r
    global trans_zs_z, trans_z_zs, hhl_z
    global vols_map_z
    global ng

    ng2 = ng + 1

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    pids = comm.pids

    nz, ny, nx = thetav[ng:-ng, ng:-ng, ng:-ng].shape
    
    thetav_z = (trans_zs_z * thetav[ng:-ng, ng:-ng, ng:-ng].flatten()).reshape(hhl_z.size - 1, ny, nx)        
    thmean = np.mean(vols_map_z * thetav_z, axis = (1, 2)) / (np.mean(vols_map_z, axis = (1, 2)) + 1e-100)    
    thetav_tmp = thetav_z.copy()
    thetav_tmp[thmean == 0.0] = 1.0 
    thetav_tmp[thetav_tmp <= 0.0] = np.nan
    thetav_avg_loc = np.nanmean(thetav_tmp, axis = (1, 2))
    thetav_avg_loc[thetav_avg_loc == 1.0] = 0.0
    
    if rank == 0:
        buffs = [thetav_avg_loc.copy() for id in pids[1:]]
        reqs = []
        for k, id in enumerate(pids[1:]):
            reqs.append(mpicomm.Irecv(buffs[k], source=id))            
        received = [0 for id in pids[1:]]
        nzeros = np.zeros([thetav_avg_loc.size])
        nzeros += thetav_avg_loc > 1.0

        while not all(received):
            for k, req in enumerate(reqs):
                if not received[k]:
                    received[k] = req.test()[0]
                else:
                    continue
                if received[k]:
                    thetav_avg_loc += buffs[k]
                    nzeros += buffs[k] > 1.0
                    
        thetav_avg_glob = thetav_avg_loc / (1e-20 + nzeros)        
        reqs = []
        for id in pids[1:]:
            reqs.append(mpicomm.Isend(thetav_avg_glob, dest = id))
        for req in reqs:
            req.wait()
    else:
         reqs = mpicomm.Isend(thetav_avg_loc, dest=0)
         thetav_avg_glob = thetav_avg_loc.copy()
         reqr = mpicomm.Irecv(thetav_avg_glob, source=0)
         reqr.wait()

    thetav_avg_glob = thetav_avg_glob.reshape(thetav_avg_glob.size, 1, 1)       
    dw_z = -9.81 * (thetav_avg_glob - thetav_z) / thetav_avg_glob    
    dw_c = (trans_z_zs * dw_z.flatten()).reshape(nz, ny, nx)    

    shape = w.shape
    nz_sub, ny_sub, nx_sub = shape
    dw = np.zeros([nz_sub, ny_sub, nx_sub])
    dw[ng2:-ng2, ng:-ng, ng:-ng] = 0.5 * (dw_c[:-1] + dw_c[1:])    

    if param_dict['bnd_zl'] == 'cyclic':        
        dw[ng, ng:-ng, ng:-ng] =  0.5 * (dw_c[0] + dw_c[-1])
        dw[-ng2, ng:-ng, ng:-ng] =  0.5 * (dw_c[0] + dw_c[-1])

    return dw


def coriolis(param_dict):
    """
    Computes the Coriolis tendencies for a mean geographic latitude.
    Neglects metric correction terms.
    
    param_dict... parameter dictionary

    du, dv, dw... returned Coriolis tendencies
    """

    global vel_flds_stag
    global ng

    ng1 = ng - 1
    ng2 = ng + 1

    u, u_j, u_i, u_k, v, v_j, v_i, v_k, w, w_j, w_i, w_k = vel_flds_stag[:12]

    latitude = param_dict['latitude']
    omega = 2.0 * param_dict["omega_coriolis"]
    conv_fac = np.pi / 180.0
    latitude = param_dict['latitude'] * conv_fac

    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dw = np.zeros_like(w)   

    if not bool(param_dict['coriolis']):
        return du, dv, dw    
    
    du[ng:-ng, ng:-ng, ng:-ng] = -2.0 * omega * (
                                                    np.cos(latitude) * 0.5 * (w_j[ng2:-ng, ng:-ng, ng1:-ng1] + w_j[ng:-ng2, ng:-ng, ng1:-ng1]) - 
                                                    np.sin(latitude) * 0.5 * (v_j[ng:-ng, ng2:-ng, ng1:-ng1] + v_j[ng:-ng, ng:-ng2, ng1:-ng1]) 
                                                )

    dv[ng:-ng, ng:-ng, ng:-ng] = -2.0 * omega * np.sin(latitude) * 0.5 * (u_i[ng:-ng, ng1:-ng1, ng2:-ng] + u_i[ng:-ng, ng1:-ng1, ng:-ng2]) 

    dw[ng:-ng, ng:-ng, ng:-ng] = 2.0 * omega * np.cos(latitude) * 0.5 * (u_k[ng1:-ng1, ng:-ng, ng2:-ng] + u_k[ng1:-ng1, ng:-ng, ng:-ng2])

    return du, dv, dw


def lsc_pressure_grad(comm, u, v, param_dict):
    """
    Computes the large-scale pressure gradient tendency
    based on the geostrophic approximation using the 
    horizontally averaged horizontal velocity at the domain-top.
   
    comm... ddcp.communicator
    u, v... horizontal velocity components
    param_dict... parameter dictionary

    du, dv... returned large-scale pressure gradient tendencies
    """

    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    
    global ng
    ng2 = ng + 1

    if param_dict['lsc_pres']:
   
        u_top = ddcp.fld_mean_para(comm, u[-2:-1, ng:-ng, ng:-n2], axis=(1,2))
        v_top = ddcp.fld_mean_para(comm, v[-2:-1, ng:-n2, ng:-ng], axis=(1,2))
        f_cor = 4.0 * np.pi / 86164.1 * np.sin(param_dict['latitude'] / 180.0 * np.pi)
        
        du[ng:-ng, ng:-ng, ng:-ng] = -f_cor * v_top
        dv[ng:-ng, ng:-ng, ng:-ng] =  f_cor * u_top

    return du, dv

def free_slip(u, v, w, ffx, ffy, ffz):
    """
    Set the velocity components to zero on impermeable cell faces.

    u, v, w... velocity component fields
    ffx, ffy, ffz... area-scaling fields
    """
   
    global ng
    ng1 = ng - 1 

    face_x = ffx[ng:-ng, ng:-ng, ng:-ng] == 0.0
    face_y = ffy[ng:-ng, ng:-ng, ng:-ng] == 0.0
    face_z = ffz[ng:-ng, ng:-ng, ng:-ng] == 0.0

    u[ng:-ng, ng:-ng, ng:-ng][face_x] = 0.0
    v[ng:-ng, ng:-ng, ng:-ng][face_y] = 0.0
    w[ng:-ng, ng:-ng, ng:-ng][face_z] = 0.0

    
def update_bnds(mpicomm, field, type='c', mode='repl'):
    """
    This routine updates the subdomain boundaries.
    
    mpicomm... MPI communicator
    field... subdomain field
    type... field type ('c': volume centred; 'u', 'v', 'w': area centred)
    """  
   
    bnds_expl_s = globals()['bnds_expl_' + type + '_s']
    bnds_expl_r = globals()['bnds_expl_' + type + '_r']
    bnds_expl_cycl_s = globals()['bnds_expl_cycl_' + type + '_s']
    bnds_expl_cycl_r = globals()['bnds_expl_cycl_' + type + '_r']

    ddcp.cptobounds(field, bnds_expl_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_s, bnds_expl_r)
    ddcp.cpfrombounds(field, bnds_expl_r, mode=mode)

    ddcp.cptobounds(field, bnds_expl_cycl_s)
    ddcp.exchange_fields(mpicomm, bnds_expl_cycl_s, bnds_expl_cycl_r)
    ddcp.cpfrombounds(field, bnds_expl_cycl_r, mode=mode)
              


def surface_tendencies(comm, param_dict, elev=1):
    """
    Computes the parameterized surface fluxes of
    momentum and heat, and derives the surface tendencies.

    comm... ddcp.communicator
    param_dict... parameter dictionary  
    elev.. number of additional z levels evaluate the surface stress
           between higher levels and the ground.

    du, dv, dw, dthetav, dqv... returned surface tendencies for velocity, thetav, and specific humidity
    """

    global vel_flds_stag
    global frict_data
    global thetav, th_surf, rho, qv, qv_surf
    global ng
    global grid_flds_stag

    n2 = ng
    n1 = ng - 1
    n0 = n1 - 1
    n3 = ng + 1

    u, u_j, u_i, u_k, v, v_j, v_i, v_k, w, w_j, w_i, w_k, speed, speed_uv, speed_uw, speed_vw = vel_flds_stag[:16]
    deff_v, deff_hx, deff_hy, area_v, area_hx, area_hy, z0, karm = frict_data[:]  


    vols_eff, vols_eff_j, vols_eff_i, vols_eff_k = grid_flds_stag[12:16]

    z = param_dict['zcoord']    
    du = np.zeros_like(u)
    dv = np.zeros_like(v)
    dw = np.zeros_like(w)
    dthetav = np.zeros_like(thetav)
    dq_surf = qv - qv_surf
    dqv = np.zeros_like(thetav)

    # vertical fluxes from horizontal surfaces

    theta = thetav / (1.0 + 0.61 * qv)

    theta_elev = theta.copy()
    u_elev = u.copy()
    v_elev = v.copy()
    speed_uv_elev = speed_uv.copy()
    qv_elev = qv.copy()

    theta_elev[:-elev] = theta[elev:]
    u_elev[:-elev] = u[elev:]
    v_elev[:-elev] = v[elev:]
    speed_uv_elev[:-elev] = speed_uv[elev:]
    qv_elev[:-elev] = qv[elev:]
 
    th_diff = theta_elev  - th_surf

    dq_surf = qv_elev - qv_surf

    dz_elev = deff_v.copy()
    dz_elev[n2:-elev - n2 - 1] = (z[elev + 1:] - z[1:-elev]).reshape(z[elev + 1:].size, 1, 1)
    deff_v_stag =  deff_v + dz_elev


    riB =  9.81 * (
                      (theta_elev[n1:-n1, n1:-n1, n1:-n1] - th_surf[n1:-n1, n1:-n1, n1:-n1]) * (deff_v_stag[n1:-n1, n1:-n1, n1:-n1]) / 
                      (th_surf[n1:-n1, n1:-n1, n1:-n1] * speed_uv_elev[n1:-n1, n1:-n1, n1:-n1] ** 2 + 1e-20)
                  )

    C_m_n = (karm / np.log(deff_v_stag[n1:-n1, n1:-n1, n1:-n1] / (z0[n1:-n1, n1:-n1, n1:-n1] + 1e-20) + 1e-20)) ** 2 

    C_h_n = C_m_n 
    fh = riB.copy()
    fm = riB.copy()
    stable = riB >= 0.0
    fm[stable] = 1.0 / (1.0 + 2.0 * 5.0 * riB[stable] * (1.0 + 5.0 * riB[stable]) ** (-0.5))
    fh[stable] = 1.0 / (1.0 + 3.0 * 5.0 * riB[stable] * (1.0 + 5.0 * riB[stable]) ** (-0.5))
    unstable = riB < 0.0
    fm[unstable] = (
                       1.0 + 2.0 * 5.0 * np.absolute(riB[unstable]) / 
                       (1.0 + 3.0 * 5.0 * 5.0 * C_m_n[unstable] * 
                       (
                           (np.maximum(deff_v_stag[n1:-n1, n1:-n1, n1:-n1] , 2.0 * z0[n1:-n1, n1:-n1, n1:-n1]) / 
                           (z0[n1:-n1, n1:-n1, n1:-n1]  + 1e-20))[unstable] ** (1.0 / 3.0) - 1.0
                       ) ** (3.0 / 2.0) * (np.absolute(riB[unstable])) ** 0.5)
                   )

    fh[unstable] = (
                       1.0 + 3.0 * 5.0 * np.absolute(riB[unstable]) / 
                       (1.0 + 3.0 * 5.0 * 5.0 * C_h_n[unstable] * 
                       (
                           (np.maximum(deff_v_stag[n1:-n1, n1:-n1, n1:-n1], 2.0 * z0[n1:-n1, n1:-n1, n1:-n1]) / 
                           (z0[n1:-n1, n1:-n1, n1:-n1] + 1e-20))[unstable] ** (1.0 / 3.0) - 1.0
                       ) ** (3.0 / 2.0) * (np.absolute(riB[unstable])) ** 0.5)
                   ) 

    C_m = C_m_n * fm * area_v[n1:-n1, n1:-n1, n1:-n1] * speed_uv_elev[n1:-n1, n1:-n1, n1:-n1]
    C_h = C_h_n * fh * area_v[n1:-n1, n1:-n1, n1:-n1] * speed_uv_elev[n1:-n1, n1:-n1, n1:-n1]
     
    # momentum sink term for du    
    S_uz = -0.5 * (C_m[1:-1, 1:-1, :-1] + C_m[1:-1, 1:-1, 1:]) * u_elev[n2:-n2, n2:-n2, n2:-n2] 
    du[n2:-n2, n2:-n2, n2:-n2] = S_uz / (vols_eff_j[n2:-n2, n2:-n2, n1:-n1] + 1e-20)
    
    # momentum sink term for dv
    S_vz = -0.5 * (C_m[1:-1, :-1, 1:-1] + C_m[1:-1, 1:, 1:-1]) * v_elev[n2:-n2, n2:-n2, n2:-n2] 

    dv[n2:-n2, n2:-n2, n2:-n2] = S_vz / (vols_eff_i[n2:-n2, n1:-n1, n2:-n2] + 1e-20)

    # temperature  and moisture sink/source terms

    St_z =  -C_h[1:-1, 1:-1, 1:-1] * th_diff[n2:-n2, n2:-n2, n2:-n2]
    Sq_z = -C_h[1:2, 1:-1, 1:-1] * dq_surf[n2:n3, n2:-n2, n2:-n2]

    dth = St_z / (vols_eff[n2:-n2, n2:-n2, n2:-n2] +  1e-20)

    if param_dict['src_qv']:
        dqv[n2:n3, n2:-n2, n2:-n2] = Sq_z / (vols_eff[n2:n3, n2:-n2, n2:-n2] +  1e-20)

    # horizontal fluxes from vertical walls
    if param_dict['hor_fluxes']:

        # fluxes from y-orientated surfaces
        C_m_n = (karm / np.log(deff_hy[n1:-n1, n1:-n1, n1:-n1] / (z0[n1:-n1, n1:-n1, n1:-n1] + 1e-20) + 1e-20)) ** 2
        C_m = C_m_n * area_hy[n1:-n1, n1:-n1, n1:-n1] * speed_uw[n1:-n1, n1:-n1, n1:-n1]
        C_h = C_m
        S_uy = -0.5 * (C_m[1:-1, 1:-1, :-1] + C_m[1:-1, 1:-1, 1:]) * u[n2:-n2, n2:-n2, n2:-n2]        
        du[n2:-n2, n2:-n2, n2:-n2] += S_uy / (vols_eff_j[n2:-n2, n2:-n2, n1:-n1] + 1e-20)
   
        S_wy = -0.5 * (C_m[:-1, 1:-1, 1:-1] + C_m[1:, 1:-1, 1:-1]) * w[n2:-n2, n2:-n2, n2:-n2] 
        dw[n2:-n2, n2:-n2, n2:-n2] += S_wy / (vols_eff_k[n1:-n1, n2:-n2, n2:-n2] + 1e-20) 

        St_z = -C_h[1:-1, 1:-1, 1:-1] * th_diff[n2:-n2, n2:-n2, n2:-n2]
        dth += St_z / (vols_eff[n2:-n2, n2:-n2, n2:-n2] +  1e-20)

        # fluxes from x-orientated surfaces
        C_m_n = (karm / np.log(deff_hx[n1:-n1, n1:-n1, n1:-n1] / (z0[n1:-n1, n1:-n1, n1:-n1] + 1e-20) + 1e-20)) ** 2
        C_m = C_m_n * area_hx[n1:-n1, n1:-n1, n1:-n1] * speed_vw[n1:-n1, n1:-n1, n1:-n1]
        C_h = C_m
        S_vx = -0.5 * (C_m[1:-1, :-1, 1:-1] + C_m[1:-1, 1:, 1:-1]) * v[n2:-n2, n2:-n2, n2:-n2]
        dv[n2:-n2, n2:-n2, n2:-n2] += S_vx / (vols_eff_i[n2:-n2, n1:-n1, n2:-n2] + 1e-20)

        S_wx = -0.5 * (C_m[:-1, 1:-1, 1:-1] + C_m[1:, 1:-1, 1:-1]) * w[n2:-n2, n2:-n2, n2:-n2]
        dw[n2:-n2, n2:-n2, n2:-n2] += S_wx / (vols_eff_k[n1:-n1, n2:-n2, n2:-n2] + 1e-20)

        St_z = -C_h[1:-1, 1:-1, 1:-1] * th_diff[n2:-n2, n2:-n2, n2:-n2]
        dth += St_z / (vols_eff[n2:-n2, n2:-n2, n2:-n2] +  1e-20)
    

    if param_dict['src_theta']:     
        dthetav[n2:-n2, n2:-n2, n2:-n2] = (
                                              dth * (1.0 + 0.61 * qv[n2:-n2, n2:-n2, n2:-n2]) + 
                                              theta[n2:-n2, n2:-n2, n2:-n2] * 0.61 * dqv[n2:-n2, n2:-n2, n2:-n2]
                                          ) 
    
    return du, dv, dw, dthetav, dqv
 

def emit(emiss, time):
    """
    Adds an emission to a tracer field.

    emiss... emission field
    emiss_facs... scaling factors for emiss
    time... model time

    dem... returned emission tendency
    """

    global vols_eff
    global ng

    dem = np.zeros_like(vols_eff)
    
#    vals = emiss_facs
    
#    keys = np.arange(0, 3600 * len(vals), 3600)

#    ind_t = np.argwhere(time >= keys)[0][0]

#    emiss_fac = vals[ind_t] + (time - keys[ind_t]) / (keys[ind_t + 1] - keys[ind_t]) * vals[ind_t + 1]
    
   

    dem[ng:-ng, ng:-ng, ng:-ng] = emiss[ng:-ng, ng:-ng, ng:-ng] / vols_eff[ng:-ng, ng:-ng, ng:-ng]    

    return dem



def dt_CFL(comm, param_dict):
    """
    Applies the CFL criterion to derive the largest possible  model time step.
    
    comm... communicator
    param_dict... parameter dictionary

    dt... returned largest model time step
    """

    global vel_flds_stag
    global grid_flds_stag
    global vols_eff
    global ng

    u_j = vel_flds_stag[1]
    v_i = vel_flds_stag[6]
    w_k = vel_flds_stag[11]

    cmax = param_dict['cmax']
    dz = param_dict['dz']
 

    area_eff_xj = grid_flds_stag[1]
    area_eff_yi = grid_flds_stag[6]
    area_eff_zk = grid_flds_stag[11]

    mpicomm = comm.mpicomm
    pids = comm.pids
    rank = mpicomm.Get_rank()    
    u, v, w = vel_flds    
    
    dt_con =  cmax * vols_eff[ng:-ng, ng:-ng, ng:-ng] /   (
                                                              np.absolute(u_j[ng:-ng, ng:-ng, ng:-ng]) * area_eff_xj[ng:-ng, ng:-ng, ng:-ng] +
                                                              np.absolute(v_i[ng:-ng, ng:-ng, ng:-ng]) * area_eff_yi[ng:-ng, ng:-ng, ng:-ng] +
                                                              np.absolute(w_k[ng:-ng, ng:-ng, ng:-ng]) * area_eff_zk[ng:-ng, ng:-ng, ng:-ng] +
                                                      1e-20)


    dt = np.min(dt_con[vols_eff[ng:-ng, ng:-ng, ng:-ng] > 1e-10])
    
    if rank == 0:
        dt_arr = [np.empty([1]) for p in pids]
        dt_arr[0] = dt
        ddcp.gather_point(mpicomm, dt_arr[1:], pids[1:])
        dt = [np.min(np.array(dt_arr))]
        buffs = [dt[0] for p in pids[1:]]
        ddcp.scatter_point(mpicomm, buffs, pids[1:])
    else:
        req = mpicomm.Isend(dt, dest=0)
        dt = np.empty([1])
        req = mpicomm.Irecv(dt, source=0)
        req.wait()
    
    return dt[0]


def vorticity(u, v, w):
    """ 
    Computes the 3d-vorticity vector.
 
    u, v, w... prognostic velocity component fields

    vort_x, vort_y, vort_z... returned vorticity components
    """

    global grid_flds_stag
    global vols_eff
    global ng

    n2 = ng
    n1 = ng - 1
    n0 = ng - 2
    
    nrh = w.shape[1] - n0
    nzh = v.shape[0] - n0
    nch = w.shape[2] - n0

    area_eff_xj = grid_flds_stag[1]
    area_eff_yi = grid_flds_stag[6]
    area_eff_zk = grid_flds_stag[11]

    vort_x = np.zeros_like(vols_eff)
    vort_y = np.zeros_like(vols_eff)
    vort_z = np.zeros_like(vols_eff)

    vort_x[n1:-n1, n1:-n1, n1:-n1] = 0.25 * (
                                                (   
                                                    (w[n2:-n1, n2:nrh, n1:-n1] + w[n1:-n2, n2:nrh, n1:-n1]) - 
                                                    (w[n2:-n1, n0:-n2, n1:-n1] + w[n1:-n2, n0:-n2, n1:-n1])
                                                ) * area_eff_yi[n1:-n1, n1:-n1, n1:-n1] -
                                                (
                                                    (v[n2:nzh, n2:-n1, n1:-n1] + v[n2:nzh, n1:-n2, n1:-n1]) - 
                                                    (v[n0:-n2, n2:-n1, n1:-n1] + v[n0:-n2, n1:-n2, n1:-n1])
                                                ) * area_eff_zk[n1:-n1, n1:-n1, n1:-n1]
                                            ) / np.maximum(vols_eff[n1:-n1, n1:-n1, n1:-n1], 1e-20)

    vort_y[n1:-n1, n1:-n1, n1:-n1] = 0.25 * (
                                                (
                                                    (u[n2:nzh, n1:-n1, n2:-n1] + u[n2:nzh, n1:-n1, n1:-n2]) - 
                                                    (u[n0:-n2, n1:-n1, n2:-n1] + u[n0:-n2, n1:-n1, n1:-n2])
                                                ) * area_eff_zk[n1:-n1, n1:-n1, n1:-n1] - 
                                                (
                                                    (w[n2:-n1, n1:-n1, n2:nch] + w[n1:-n2, n1:-n1, n2:nch]) -
                                                    (w[n2:-n1, n1:-n1, n0:-n2] + w[n1:-n2, n1:-n1, n0:-n2])
                                                ) * area_eff_xj[n1:-n1, n1:-n1, n1:-n1] 
                                            ) / np.maximum(vols_eff[n1:-n1, n1:-n1, n1:-n1], 1e-20)

    vort_z[n1:-n1, n1:-n1, n1:-n1] = 0.25 * (
                                                (
                                                    (v[n1:-n1, n2:-n1, n2:nch] + v[n1:-n1, n1:-n2, n2:nch]) - 
                                                    (v[n1:-n1, n2:-n1, n0:-n2] + v[n1:-n1, n1:-n2, n0:-n2])
                                                ) * area_eff_xj[n1:-n1, n1:-n1, n1:-n1] - 
                                                (
                                                    (u[n1:-n1, n2:nrh, n2:-n1] + u[n1:-n1, n2:nrh, n1:-n2]) - 
                                                    (u[n1:-n1, n0:-n2, n2:-n1] + u[n1:-n1, n0:-n2, n1:-n2])
                                                ) * area_eff_yi[n1:-n1, n1:-n1, n1:-n1]
                                            ) / np.maximum(vols_eff[n1:-n1, n1:-n1, n1:-n1], 1e-20)

    return vort_x, vort_y, vort_z


def rayleigh_damping(field, param_dict, type='c'):
    """
    A top-domain Rayleigh damping formulation as an
    additional explicit tendency

    field... prognostic field to apply damping on
    param_dict... parameter dictionary
    type... field type ('c': volume centred; 'u', 'v', 'w': area centred)

    dfield... returned damping tendency
    """

    global ng

    dfield = np.zeros_like(field)
    l_damp = max(param_dict['ldamp'], 1e-3)
    tau_damp = param_dict['taudamp']

    if type in ['c', 'u', 'v']:
        zcoord = param_dict['zcoord']
        l = zcoord[-1] - zcoord
        r = 1.0 - np.cos(np.pi * (zcoord - (zcoord[-1] -  l_damp)) / l_damp)        
        r[zcoord < zcoord[-1] - l_damp] = 0.0
        
    elif type == 'w':
        z2coord = param_dict['z2coord']
        l = z2coord[-1] - z2coord
        r = 1.0 - np.cos(np.pi * (z2coord - (z2coord[-1] -  l_damp)) / l_damp)
        r[z2coord < z2coord[-1] - l_damp] = 0.0


    dfield[ng:-ng, ng:-ng, ng:-ng] = 1.0 / (2.0 * tau_damp) * (field[-2:-1, ng:-ng, ng:-ng] -  field[ng:-ng, ng:-ng, ng:-ng]) * r.reshape(r.size, 1, 1)

    return dfield


def fill_boundaries(field, axis, number):
    """
    Fills the boundaries of a field with 
    interior values.

    field... field
    axis... determines the axis of the boundaries to fill
    number... number of ghost layers
    """

    if axis == 0:
        pass
    elif axis == 1:
        field = np.swapaxes(field, 0, axis)
    else:
        field = np.swapaxes(field, 0, axis)
    for i in range(number):
        field[i] = field[number]
        field[-1 - i] = field[-number - 1]

    if axis == 0:
        pass
    elif axis == 1:
        field = np.swapaxes(field, 0, axis)
    else:
        field = np.swapaxes(field, 0, axis)
  

# reconstruction routines for advection



def init_upwind_recon(grid_flds, area_factors, param_dict, lim_d=1e-10):
    """
    Initializes the upwind-scheme reconstruction
    by calculating the linear  reconstruction weights
    and the weights for flux limiting.

    grid_flds... effective areas and volumes
    area_factors... area-scaling factors
    param_dict... parameter dictionary
    lim_d... limit minimum and maximum effective grid spacing of pseudo grid
    """

    eps = 1e-80

    global K_xl_c, K_yl_c, K_zl_c
    global K_xr_c, K_yr_c, K_zr_c
    global K_xc_c, K_yc_c, K_zc_c

    global wghtxl_lim, wghtyl_lim, wghtzl_lim
    global wghtxr_lim, wghtyr_lim, wghtzr_lim
    global inds_lim_xl, inds_lim_yl, inds_lim_zl
    global inds_lim_xr, inds_lim_yr, inds_lim_zr

    global ng

    order = int(param_dict['adv_order'])

    if ng * 2 - 1 != order:
        print "Error: Number of ghost cells does not match order of advection scheme!"
        raise ValueError
    if order%2 == 0:
        print "Error: Even orders of advection scheme not supported (only upwind-biased schemes)"
        raise ValueError

    area_eff_x, area_eff_y, area_eff_z, vols_eff = grid_flds[:]
    ffx, ffy, ffz = area_factors[:]

    ng1 = ng + 1
    nc = vols_eff.shape[2] - 2 * ng1 + 1
    nr = vols_eff.shape[1] - 2 * ng1 + 1

    dxr_eff = vols_eff[ng1:-ng1, ng1:-ng1, 1:-1] / np.maximum(area_eff_x[ng1:-ng1, ng1:-ng1, 1:-2], eps)
    dyr_eff = vols_eff[ng1:-ng1, 1:-1, ng1:-ng1] / np.maximum(area_eff_y[ng1:-ng1, 1:-2, ng1:-ng1], eps)
    dzr_eff = vols_eff[1:-1, ng1:-ng1, ng1:-ng1] / np.maximum(area_eff_z[1:-2, ng1:-ng1, ng1:-ng1], eps)

    dxl_eff = vols_eff[ng1:-ng1, ng1:-ng1, 1:-1] / np.maximum(area_eff_x[ng1:-ng1, ng1:-ng1, 2:-1], eps)
    dyl_eff = vols_eff[ng1:-ng1, 1:-1, ng1:-ng1] / np.maximum(area_eff_y[ng1:-ng1, 2:-1, ng1:-ng1], eps)
    dzl_eff = vols_eff[1:-1, ng1:-ng1, ng1:-ng1] / np.maximum(area_eff_z[2:-1, ng1:-ng1, ng1:-ng1], eps)

    dxl_eff = np.maximum(np.minimum(dxl_eff, 1.0 / lim_d), lim_d)
    dyl_eff = np.maximum(np.minimum(dyl_eff, 1.0 / lim_d), lim_d)
    dzl_eff = np.maximum(np.minimum(dzl_eff, 1.0 / lim_d), lim_d)

    dxr_eff = np.maximum(np.minimum(dxr_eff, 1.0 / lim_d), lim_d)
    dyr_eff = np.maximum(np.minimum(dyr_eff, 1.0 / lim_d), lim_d)
    dzr_eff = np.maximum(np.minimum(dzr_eff, 1.0 / lim_d), lim_d)

    wghtxl_lim = np.ones_like(ffx)
    wghtyl_lim = np.ones_like(ffy)
    wghtzl_lim = np.ones_like(ffz)
    wghtxr_lim = np.ones_like(ffx)
    wghtyr_lim = np.ones_like(ffy)
    wghtzr_lim = np.ones_like(ffz)

    dffxl = abs(ffx[:, :, 1:-1] - ffx[:, :, :-2])
    dffxr = abs(ffx[:, :, 1:-1] - ffx[:, :, 2:])
    dffyl = abs(ffy[:, 1:-1] - ffy[:, :-2])
    dffyr = abs(ffy[:, 1:-1] - ffy[:, 2:])
    dffzl = abs(ffz[1:-1] - ffz[:-2])
    dffzr = abs(ffz[1:-1] - ffz[2:])

    wghtxl_lim[:, :, 1:-1] = (1.0 - dffxl)
    wghtxr_lim[:, :, 1:-1] = (1.0 - dffxr)
    wghtyl_lim[:, 1:-1] = (1.0 - dffyl)
    wghtyr_lim[:, 1:-1] = (1.0 - dffyr)
    wghtzl_lim[1:-1] = (1.0 - dffzl)
    wghtzr_lim[1:-1] = (1.0 - dffzr)

    wghtxl_lim[ffx == 0] = 1.0
    wghtyl_lim[ffy == 0] = 1.0
    wghtzl_lim[ffz == 0] = 1.0
    wghtxr_lim[ffx == 0] = 1.0
    wghtyr_lim[ffy == 0] = 1.0
    wghtzr_lim[ffz == 0] = 1.0

    wghtxl_lim = wghtxl_lim[ng:-ng, ng:-ng, ng:-ng]
    wghtyl_lim = wghtyl_lim[ng:-ng, ng:-ng, ng:-ng]
    wghtzl_lim = wghtzl_lim[ng:-ng, ng:-ng, ng:-ng]
    wghtxr_lim = wghtxr_lim[ng:-ng, ng:-ng, ng:-ng]
    wghtyr_lim = wghtyr_lim[ng:-ng, ng:-ng, ng:-ng]
    wghtzr_lim = wghtzr_lim[ng:-ng, ng:-ng, ng:-ng]

    inds_lim_xl = np.where(wghtxl_lim < 1)
    inds_lim_yl = np.where(wghtyl_lim < 1)
    inds_lim_zl = np.where(wghtzl_lim < 1)
    inds_lim_xr = np.where(wghtxr_lim < 1)
    inds_lim_yr = np.where(wghtyr_lim < 1)
    inds_lim_zr = np.where(wghtzr_lim < 1)

    wghtxl_lim = wghtxl_lim[inds_lim_xl]
    wghtyl_lim = wghtyl_lim[inds_lim_yl]
    wghtzl_lim = wghtzl_lim[inds_lim_zl]
    wghtxr_lim = wghtxr_lim[inds_lim_xr]
    wghtyr_lim = wghtyr_lim[inds_lim_yr]
    wghtzr_lim = wghtzr_lim[inds_lim_zr]

    K_xl_c, K_yl_c, K_zl_c, K_xr_c, K_yr_c, K_zr_c =  Upwind_coeff(dxl_eff, dyl_eff, dzl_eff, dxr_eff, dyr_eff, dzr_eff, order)



def init_WENO_recon(grid_flds, param_dict, lim_d=1e-3):
    """
    Initializes the WENO-reconstruction
    by calculating the  reconstruction weights of the 
    WENO stencils and the WENO weights.

    grid_flds... effective areas and volumes
    param_dict... parameter dictionary
    lim_d... limit minimum and maximum effective grid spacing of pseudo grid
    """

    eps = 1e-80

    global Ks_xl, Ks_yl, Ks_zl
    global Ks_xr, Ks_yr, Ks_zr
    global Ks_xc, Ks_yc, Ks_zc

    global dx_eff, dy_eff, dz_eff
    global Xs_xl, Xs_yl, Xs_zl
    global Xs_xr, Xs_yr, Xs_zr

    global L_xl, L_yl, L_zl
    global L_xr, L_yr, L_zr

    global wghtxl_lim, wghtyl_lim, wghtzl_lim
    global wghtxr_lim, wghtyr_lim, wghtzr_lim
    global inds_lim_xl, inds_lim_yl, inds_lim_zl
    global inds_lim_xr, inds_lim_yr, inds_lim_zr

    global ng

    order = int(param_dict['adv_order'])

    schemes = {3:(2,2), 5:(3,3), 7:(4,4), 9:(5,5)}
    order_sub, n_stencil = schemes[order][:]

    ng1 = ng + 1

    area_eff_x, area_eff_y, area_eff_z, vols_eff = grid_flds[:]

    dx_eff = 2.0 * vols_eff[ng1:-ng1, ng1:-ng1, 1:-1] / np.maximum(area_eff_x[ng1:-ng1, ng1:-ng1, 2:-1] + area_eff_x[ng1:-ng1, ng1:-ng1, 1:-2], eps)
    dy_eff = 2.0 * vols_eff[ng1:-ng1, 1:-1, ng1:-ng1] / np.maximum(area_eff_y[ng1:-ng1, 2:-1, ng1:-ng1] + area_eff_y[ng1:-ng1, 1:-2, ng1:-ng1], eps)
    dz_eff = 2.0 * vols_eff[1:-1, ng1:-ng1, ng1:-ng1] / np.maximum(area_eff_z[2:-1, ng1:-ng1, ng1:-ng1] + area_eff_z[1:-2, ng1:-ng1, ng1:-ng1], eps)
    dz_eff[ng] = vols_eff[ng + 1, ng1:-ng1, ng1:-ng1] / np.maximum(area_eff_z[ng + 2, ng1:-ng1, ng1:-ng1], eps)

    dx_eff = np.maximum(np.minimum(dx_eff, 1.0 / lim_d), lim_d)
    dy_eff = np.maximum(np.minimum(dy_eff, 1.0 / lim_d), lim_d)
    dz_eff = np.maximum(np.minimum(dz_eff, 1.0 / lim_d), lim_d)

    Ks_xl, Ks_yl, Ks_zl, Ks_xr, Ks_yr, Ks_zr = ENO_coeff(dx_eff, dy_eff, dz_eff, order_sub, n_stencil=n_stencil)
    Xs_xl, Xs_yl, Xs_zl, Xs_xr, Xs_yr, Xs_zr = ENO_stencils(dx_eff, dy_eff, dz_eff, order_sub, n_stencil=n_stencil)
    Khs_xl, Khs_yl, Khs_zl, Khs_xr, Khs_yr, Khs_zr = ENO_coeff(dx_eff, dy_eff, dz_eff, order, n_stencil=1)

    L_xl  = linear_weights(Khs_xl[0], Ks_xl)
    L_yl  = linear_weights(Khs_yl[0], Ks_yl)
    L_zl  = linear_weights(Khs_zl[0], Ks_zl)
    L_xr  = linear_weights(Khs_xr[0], Ks_xr[::-1])[::-1]
    L_yr  = linear_weights(Khs_yr[0], Ks_yr[::-1])[::-1]
    L_zr  = linear_weights(Khs_zr[0], Ks_zr[::-1])[::-1]


    wghtxl_lim = None
    wghtyl_lim = None
    wghtzl_lim = None
    wghtxr_lim = None
    wghtyr_lim = None
    wghtzr_lim = None
    inds_lim_xl = None
    inds_lim_yl = None
    inds_lim_zl = None
    inds_lim_xr = None
    inds_lim_yr = None
    inds_lim_zr = None


def upwind_recon_scalar(
                            c, wghtxl_lim=1.0, wghtyl_lim=1.0, wghtzl_lim=1.0, 
                            wghtxr_lim=1.0, wghtyr_lim=1.0, wghtzr_lim=1.0, 
                            inds_lim_xl=None, inds_lim_yl=None, inds_lim_zl=None,
                            inds_lim_xr=None, inds_lim_yr=None, inds_lim_zr=None
                       ):
    """
    Uses upwind reconstruction of arbitrary order
    for scalar advection on all 3 bounding faces.
    Additional flux limiting can applied.

    c... the scalar field to be reconstructed
    wghtx_lim, wghty_lim, wghtz_lim... the linear weights to merge the unlimited and limited reconstruction.
                                       For a value of 1, full limiting is applied
    inds_lim_x, inds_lim_y, inds_lim_z... the index arrays for fancy indexing to apply limiting only on an array subset
                                       For a value of None, the full field is limited.


    recons... returned reconstruction components, two on each cell face
    """

    global K_xl_c, K_yl_c, K_zl_c
    global K_xr_c, K_yr_c, K_zr_c
    global ng

    shp_c = c.shape

    nzh = shp_c[0] - ng + 2
    nrh = shp_c[1] - ng + 2
    nch = shp_c[2] - ng + 2

    int_xl, int_yl, int_zl, int_xr, int_yr, int_zr = upwind_recon_linear(K_xl_c, K_yl_c, K_zl_c, K_xr_c, K_yr_c, K_zr_c, c)

    c_rec_lx = c[ng:-ng, ng:-ng, (ng - 1):-ng] + int_xl
    c_rec_rx = c[ng:-ng, ng:-ng, ng:-(ng - 1)] + int_xr
    c_rec_ly = c[ng:-ng, (ng - 1):-ng, ng:-ng] + int_yl
    c_rec_ry = c[ng:-ng, ng:-(ng - 1), ng:-ng] + int_yr
    c_rec_lz = c[(ng - 1):-ng, ng:-ng, ng:-ng] + int_zl
    c_rec_rz = c[ng:-(ng - 1), ng:-ng, ng:-ng] + int_zr

    c_ll = c[ng:-ng, ng:-ng, (ng - 2):-(ng + 1)][inds_lim_xl]
    c_l = c[ng:-ng, ng:-ng, (ng - 1):-ng][inds_lim_xl]
    c_r = c[ng:-ng, ng:-ng, ng:-(ng - 1)][inds_lim_xr]
    c_rr = c[ng:-ng, ng:-ng, (ng + 1):nch][inds_lim_xr]

    delta_cxl = c_l - c_ll + 1e-50
    delta_cxr = c_r - c_rr + 1e-50

    r_lx = (c[ng:-ng, ng:-ng, ng:-(ng - 1)][inds_lim_xl] - c_l) / delta_cxl
    r_rx = (c[ng:-ng, ng:-ng, (ng - 1):-ng][inds_lim_xr] - c_r) / delta_cxr

    phi_lx = int_xl[inds_lim_xl] / delta_cxl
    phi_rx = int_xr[inds_lim_xr] / delta_cxr

    limit(phi_lx, r_lx, wghtxl_lim)
    limit(phi_rx, r_rx, wghtxr_lim)

    c_rec_lx[inds_lim_xl] = c_l + phi_lx * delta_cxl
    c_rec_rx[inds_lim_xr] = c_r + phi_rx * delta_cxr

    c_ll = c[ng:-ng, (ng - 2):-(ng + 1), ng:-ng][inds_lim_yl]
    c_l = c[ng:-ng, (ng - 1):-ng, ng:-ng][inds_lim_yl]
    c_r = c[ng:-ng, ng:-(ng - 1), ng:-ng][inds_lim_yr]
    c_rr = c[ng:-ng, (ng + 1):nrh, ng:-ng][inds_lim_yr]

    delta_cyl = c_l - c_ll + 1e-50
    delta_cyr = c_r - c_rr + 1e-50

    r_ly = (c[ng:-ng, ng:-(ng - 1), ng:-ng][inds_lim_yl] - c_l) / delta_cyl
    r_ry = (c[ng:-ng, (ng - 1):-ng, ng:-ng][inds_lim_yr] - c_r) / delta_cyr

    phi_ly = int_yl[inds_lim_yl] / delta_cyl
    phi_ry = int_yr[inds_lim_yr] / delta_cyr

    limit(phi_ly, r_ly, wghtyl_lim)
    limit(phi_ry, r_ry, wghtyr_lim)

    c_rec_ly[inds_lim_yl] = c_l + phi_ly * delta_cyl
    c_rec_ry[inds_lim_yr] = c_r + phi_ry * delta_cyr

    c_ll = c[(ng - 2):-(ng + 1), ng:-ng, ng:-ng][inds_lim_zl]
    c_l = c[(ng - 1):-ng, ng:-ng, ng:-ng][inds_lim_zl]
    c_r = c[ng:-(ng - 1), ng:-ng, ng:-ng][inds_lim_zr]
    c_rr = c[(ng + 1):nzh, ng:-ng, ng:-ng][inds_lim_zr]

    delta_czl = c_l - c_ll + 1e-50
    delta_czr = c_r - c_rr + 1e-50

    r_lz = (c[ng:-(ng - 1), ng:-ng, ng:-ng][inds_lim_zl] - c_l) / delta_czl
    r_rz = (c[(ng - 1):-ng, ng:-ng, ng:-ng][inds_lim_zr] - c_r) / delta_czr

    phi_lz = int_zl[inds_lim_zl] / delta_czl
    phi_rz = int_zr[inds_lim_zr] / delta_czr

    limit(phi_lz, r_lz, wghtzl_lim)
    limit(phi_rz, r_rz, wghtzr_lim)

    c_rec_lz[inds_lim_zl] = c_l + phi_lz * delta_czl
    c_rec_rz[inds_lim_zr] = c_r + phi_rz * delta_czr

    recons = [c_rec_lx, c_rec_ly, c_rec_lz, c_rec_rx, c_rec_ry, c_rec_rz]

    return recons


def ENO_recon_scalar(
                        c, flimit=False,
                        wghtxl_lim=None, wghtyl_lim=None, wghtzl_lim=None,
                        wghtxr_lim=None, wghtyr_lim=None, wghtzr_lim=None,
                        inds_lim_xl=None, inds_lim_yl=None, inds_lim_zl=None,
                        inds_lim_xr=None, inds_lim_yr=None, inds_lim_zr=None
                    ):
    """
    Uses ENO reconstruction of 2nd or 3rd order
    for scalar advection on all 3 bounding faces.

    c... the scalar field to be reconstructed
    flimit... just place-holder for generic routine call
    wghtx_lim, wghty_lim, wghtz_lim... just place-holders for generic routine call 


    recons... returned reconstruction  components on each cell face
    """

    global Ks_xl, Ks_yl, Ks_zl
    global Ks_xr, Ks_yr, Ks_zr

    global L_xl, L_yl, L_zl
    global L_xr, L_yr, L_zr

    global Xs_xl, Xs_yl, Xs_zl
    global Xs_xr, Xs_yr, Xs_zr

    n_stencil = len(Ks_xl)
    order_sub = len(Ks_xl[0])
    order = order_sub + n_stencil - 1
    nc = c.shape[2] - 2 * ng
    nr = c.shape[1] - 2 * ng
    nz = c.shape[0] - 2 * ng

    # compute the Newton divided differences

    # positive upwind stencils
    div_diff_xl = []
    div_diff_yl = []
    div_diff_zl = []

    # negative upwind stencils
    div_diff_xr = []
    div_diff_yr = []
    div_diff_zr = []

    for s in range(n_stencil):

        Y = [c[ng:-ng, ng:-ng, i + s:i + s + nc + 1]  for i in range(order_sub)]
        div_diff_xl.append(np.absolute(divided_differences(Xs_xl[s], Y)))
        Y = [c[ng:-ng, i + s:i + s + nr + 1, ng:-ng]  for i in range(order_sub)]
        div_diff_yl.append(np.absolute(divided_differences(Xs_yl[s], Y)))
        Y = [c[i + s:i + s + nz + 1, ng:-ng, ng:-ng]  for i in range(order_sub)]
        div_diff_zl.append(np.absolute(divided_differences(Xs_zl[s], Y)))


    lb = 2 * ng - order_sub
    Y = [c[ng:-ng, ng:-ng, i + lb:i + lb + nc + 1]  for i in range(order_sub)]
    div_diff_xr.append(np.absolute(divided_differences(Xs_xr[0], Y)))
    Y = [c[ng:-ng, i + lb:i + lb + nr + 1, ng:-ng]  for i in range(order_sub)]
    div_diff_yr.append(np.absolute(divided_differences(Xs_yr[0], Y)))
    Y = [c[i + lb:i + lb + nz + 1, ng:-ng, ng:-ng]  for i in range(order_sub)]
    div_diff_zr.append(np.absolute(divided_differences(Xs_zr[0], Y)))

    for s in range(1, n_stencil):
        div_diff_xr.append(div_diff_xl[-s])
        div_diff_yr.append(div_diff_yl[-s])
        div_diff_zr.append(div_diff_zl[-s])

    Cs_xl = ENO_decide_stencil(div_diff_xl)
    Cs_yl = ENO_decide_stencil(div_diff_yl)
    Cs_zl = ENO_decide_stencil(div_diff_zl)
    Cs_xr = ENO_decide_stencil(div_diff_xr)
    Cs_yr = ENO_decide_stencil(div_diff_yr)
    Cs_zr = ENO_decide_stencil(div_diff_zr)

    ints_xr = []
    ints_xl = []
    ints_yr = []
    ints_yl = []
    ints_zr = []
    ints_zl = []

    int_xl = np.zeros_like(Ks_xl[0][0])
    int_yl = np.zeros_like(Ks_yl[0][0])
    int_zl = np.zeros_like(Ks_zl[0][0])
    int_xr = np.zeros_like(Ks_xl[0][0])
    int_yr = np.zeros_like(Ks_yl[0][0])
    int_zr = np.zeros_like(Ks_zl[0][0])

    for s in range(n_stencil):

        ints_xl.append(reduce(np.add, [c[ng:-ng, ng:-ng, i + s:i + s + nc + 1] * K for i, K in  enumerate(Ks_xl[s])]))
        ints_yl.append(reduce(np.add, [c[ng:-ng, i + s:i + s + nr + 1, ng:-ng] * K for i, K in  enumerate(Ks_yl[s])]))
        ints_zl.append(reduce(np.add, [c[i + s:i + s + nz + 1, ng:-ng, ng:-ng] * K for i, K in  enumerate(Ks_zl[s])]))

    lb = 2 * ng - order_sub
    ints_xr.append(reduce(np.add, [c[ng:-ng, ng:-ng, i + lb:i + lb + nc + 1] * K for i, K in  enumerate(Ks_xr[0])]))
    ints_yr.append(reduce(np.add, [c[ng:-ng, i + lb:i + lb + nr + 1, ng:-ng] * K for i, K in  enumerate(Ks_yr[0])]))
    ints_zr.append(reduce(np.add, [c[i + lb:i + lb + nz + 1, ng:-ng, ng:-ng] * K for i, K in  enumerate(Ks_zr[0])]))

    for s in range(1, n_stencil):
        ints_xr.append(ints_xl[-s])
        ints_yr.append(ints_yl[-s])
        ints_zr.append(ints_zl[-s])

    int_xr = reduce(np.add, [Cs_xr[i] * ints_xr[i] for i in range(n_stencil)])
    int_xl = reduce(np.add, [Cs_xl[i] * ints_xl[i] for i in range(n_stencil)])
    int_yr = reduce(np.add, [Cs_yr[i] * ints_yr[i] for i in range(n_stencil)])
    int_yl = reduce(np.add, [Cs_yl[i] * ints_yl[i] for i in range(n_stencil)])
    int_zr = reduce(np.add, [Cs_zr[i] * ints_zr[i] for i in range(n_stencil)])
    int_zl = reduce(np.add, [Cs_zl[i] * ints_zl[i] for i in range(n_stencil)])

    recons = [int_xl, int_yl, int_zl, int_xr, int_yr, int_zr]

    return recons


def WENO_recon_scalar(
                            c, flimit=False, 
                            wghtxl_lim=None, wghtyl_lim=None, wghtzl_lim=None,
                            wghtxr_lim=None, wghtyr_lim=None, wghtzr_lim=None,
                            inds_lim_xl=None, inds_lim_yl=None, inds_lim_zl=None,
                            inds_lim_xr=None, inds_lim_yr=None, inds_lim_zr=None
                     ):

    """
    Uses WENO reconstruction of optimally 3rd or 5th order
    for scalar advection on all 3 bounding faces.

    c... the scalar field to be reconstructed
    flimit... just place-holder for generic routine call
    wghtx_lim, wghty_lim, wghtz_lim... just place-holders for generic routine call 


    recons... returned reconstruction  components on each cell face
    """


    global Ks_xl, Ks_yl, Ks_zl
    global Ks_xr, Ks_yr, Ks_zr

    global L_xl, L_yl, L_zl
    global L_xr, L_yr, L_zr

    global Xs_xl, Xs_yl, Xs_zl
    global Xs_xr, Xs_yr, Xs_zr

    n_stencil = len(Ks_xl)
    order_sub = len(Ks_xl[0])
    order = order_sub + n_stencil - 1
    nc = c.shape[2] - 2 * ng
    nr = c.shape[1] - 2 * ng
    nz = c.shape[0] - 2 * ng 

    # smoothnes indicators
    # positive upwind stencils
    smo_xl = []
    smo_yl = []
    smo_zl = []

    # negative upwind stencils
    smo_xr = []
    smo_yr = []
    smo_zr = []
      

    # compute smoothnes indicators
    for s in range(n_stencil):

        Y = [c[ng:-ng, ng:-ng, i + s:i + s + nc + 1]  for i in range(order_sub)]
        smo_xl.append(smoothness_indicators(Y))
        Y = [c[ng:-ng, i + s:i + s + nr + 1, ng:-ng]  for i in range(order_sub)]
        smo_yl.append(smoothness_indicators(Y))
        Y = [c[i + s:i + s + nz + 1, ng:-ng, ng:-ng]  for i in range(order_sub)]
        smo_zl.append(smoothness_indicators(Y))

    lb = 2 * ng - order_sub
    Y = [c[ng:-ng, ng:-ng, i + lb:i + lb + nc + 1]  for i in range(order_sub)]
    smo_xr.append(smoothness_indicators(Y))
    Y = [c[ng:-ng, i + lb:i + lb + nr + 1, ng:-ng]  for i in range(order_sub)]
    smo_yr.append(smoothness_indicators(Y))
    Y = [c[i + lb:i + lb + nz + 1, ng:-ng, ng:-ng]  for i in range(order_sub)]
    smo_zr.append(smoothness_indicators(Y))

    for s in range(1, n_stencil):
        smo_xr.append(smo_xl[-s])
        smo_yr.append(smo_yl[-s])
        smo_zr.append(smo_zl[-s])

    # undivided differences

    Y = [c[ng:-ng, ng:-ng, i:i + nc + 1]  for i in range(order)]
    udiv_diff_xl = np.power(undivided_differences(Y), 2.0)
    Y = [c[ng:-ng, i:i + nr + 1, ng:-ng]  for i in range(order)]
    udiv_diff_yl = np.power(undivided_differences(Y), 2.0)
    Y = [c[i:i + nz + 1, ng:-ng, ng:-ng]  for i in range(order)]
    udiv_diff_zl = np.power(undivided_differences(Y), 2.0)

    Y = [c[ng:-ng, ng:-ng, i + 1:i + 1 + nc + 1]  for i in range(order)]
    udiv_diff_xr = np.power(undivided_differences(Y), 2.0)
    Y = [c[ng:-ng, i + 1:i + 1 + nr + 1, ng:-ng]  for i in range(order)]
    udiv_diff_yr = np.power(undivided_differences(Y), 2.0)
    Y = [c[i + 1:i + 1 + nz + 1, ng:-ng, ng:-ng]  for i in range(order)]
    udiv_diff_zr = np.power(undivided_differences(Y), 2.0)

    Cs_xl = WENO_weights(L_xl, smo_xl, udiv_diff_xl, order_sub)
    Cs_yl = WENO_weights(L_yl, smo_yl, udiv_diff_yl, order_sub)
    Cs_zl = WENO_weights(L_zl, smo_zl, udiv_diff_zl, order_sub)
    Cs_xr = WENO_weights(L_xr, smo_xr, udiv_diff_xr, order_sub)
    Cs_yr = WENO_weights(L_yr, smo_yr, udiv_diff_yr, order_sub)
    Cs_zr = WENO_weights(L_zr, smo_zr, udiv_diff_zr, order_sub)

    ints_xr = []
    ints_xl = []
    ints_yr = []
    ints_yl = []
    ints_zr = []
    ints_zl = []

    int_xl = np.zeros_like(Ks_xl[0][0])
    int_yl = np.zeros_like(Ks_yl[0][0])
    int_zl = np.zeros_like(Ks_zl[0][0])
    int_xr = np.zeros_like(Ks_xl[0][0])
    int_yr = np.zeros_like(Ks_yl[0][0])
    int_zr = np.zeros_like(Ks_zl[0][0])

    for s in range(n_stencil):

        ints_xl.append(reduce(np.add, [c[ng:-ng, ng:-ng, i + s:i + s + nc + 1] * K for i, K in  enumerate(Ks_xl[s])]))
        ints_yl.append(reduce(np.add, [c[ng:-ng, i + s:i + s + nr + 1, ng:-ng] * K for i, K in  enumerate(Ks_yl[s])]))
        ints_zl.append(reduce(np.add, [c[i + s:i + s + nz + 1, ng:-ng, ng:-ng] * K for i, K in  enumerate(Ks_zl[s])]))

    lb = 2 * ng - order_sub
    ints_xr.append(reduce(np.add, [c[ng:-ng, ng:-ng, i + lb:i + lb + nc + 1] * K for i, K in  enumerate(Ks_xr[0])]))
    ints_yr.append(reduce(np.add, [c[ng:-ng, i + lb:i + lb + nr + 1, ng:-ng] * K for i, K in  enumerate(Ks_yr[0])]))
    ints_zr.append(reduce(np.add, [c[i + lb:i + lb + nz + 1, ng:-ng, ng:-ng] * K for i, K in  enumerate(Ks_zr[0])]))

    for s in range(1, n_stencil):
        ints_xr.append(ints_xl[-s])
        ints_yr.append(ints_yl[-s])
        ints_zr.append(ints_zl[-s])

    int_xr = reduce(np.add, [Cs_xr[i] * ints_xr[i] for i in range(n_stencil)])
    int_xl = reduce(np.add, [Cs_xl[i] * ints_xl[i] for i in range(n_stencil)])
    int_yr = reduce(np.add, [Cs_yr[i] * ints_yr[i] for i in range(n_stencil)])
    int_yl = reduce(np.add, [Cs_yl[i] * ints_yl[i] for i in range(n_stencil)])
    int_zr = reduce(np.add, [Cs_zr[i] * ints_zr[i] for i in range(n_stencil)])
    int_zl = reduce(np.add, [Cs_zl[i] * ints_zl[i] for i in range(n_stencil)])

    recons = [int_xl, int_yl, int_zl, int_xr, int_yr, int_zr]

    return recons



def upwind_recon_linear(K_xl, K_yl, K_zl, K_xr, K_yr, K_zr, field):
    """
    Performs linear upwind reconstruction
    without  limiting.
  
    K_xl, K_yl, K_zl, K_xr, K_yr, K_zr... lists of interpolation coefficients
    field... field to reconstruct on cell faces

    recon_lin... returned linear reconstruction components, two on each cell face
    """

    global ng
    
    int_xr = reduce(np.add, (field[ng:-ng, ng:-ng, (i + 1):(-2 * (ng - 1) + i)] * K for i, K in  enumerate(K_xr[:-1])))
    int_xr += K_xr[-1] * field[ng:-ng, ng:-ng, len(K_xr):]
   
    int_yr = reduce(np.add, (field[ng:-ng, (i + 1):(-2 * (ng - 1) + i), ng:-ng] * K for i, K in  enumerate(K_yr[:-1])))
    int_yr += K_yr[-1] * field[ng:-ng, len(K_yr):, ng:-ng]

    int_zr = reduce(np.add, (field[(i + 1):(-2 * (ng - 1) + i), ng:-ng, ng:-ng] * K for i, K in  enumerate(K_zr[:-1])))
    int_zr += K_zr[-1] * field[len(K_zr):, ng:-ng, ng:-ng]

    int_xl = reduce(np.add, (field[ng:-ng, ng:-ng, i:(-2 * (ng - 1) + i - 1)] * K for i, K in  enumerate(K_xl)))
    int_yl = reduce(np.add, (field[ng:-ng, i:(-2 * (ng - 1) + i - 1), ng:-ng] * K for i, K in  enumerate(K_yl)))
    int_zl = reduce(np.add, (field[i:(-2 * (ng - 1) + i - 1), ng:-ng, ng:-ng] * K for i, K in  enumerate(K_zl)))

    recon_lin = [int_xl, int_yl, int_zl, int_xr, int_yr, int_zr]

    return recon_lin


def limit(b, r, weight, delta=2.2):
    """
    A flux limiter function for high-resolution fluxes
    used in the upwind reconstruction (Sweby, 1984).


    b... high-order reconstruction to limit
    r... slope ratio
    weight... weights for merging of limited and unlimited reconstructions

    delta... free parameter in the flux limiter scheme.
    """

    b[:] = (
               weight * b +
               (1.0 - weight) * np.maximum(0.0,  np.minimum(r, np.minimum(delta, b)))
           )


def Upwind_coeff(dxl_eff, dyl_eff, dzl_eff, dxr_eff, dyr_eff, dzr_eff, order):
    """
    Computes the linear upwind reconstruction
    coefficients.

    dx_eff, dy_eff, dz_eff... effective  grid spacings of the pseudo grid
    order... order of reconstruction

    recon_coeff... returned lists of reconstruction coefficients
    """

    global ng

    dX = [dxl_eff[:, :, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)]
    dX.extend([dxr_eff[:, :, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1, order)])
    x = reduce(np.add, (dxl_eff[:, :, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)))
    K_xl = L_coeff(dX[:], x=x)
    K_xl[int(order / 2)] -= 1.0

    dY = [dyl_eff[:, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)]
    dY.extend([dyr_eff[:, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1, order)])
    y = reduce(np.add, (dyl_eff[:, i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)))
    K_yl = L_coeff(dY[:], x=y)
    K_yl[int(order / 2)] -= 1.0

    dZ = [dzl_eff[i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)]
    dZ.extend([dzr_eff[i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1, order)])
    x = reduce(np.add, (dzl_eff[i:(-2 * (ng - 1) + i - 1)]  for i in range((order - 1) / 2 + 1)))
    K_zl = L_coeff(dZ, x=x)
    K_zl[int(order / 2)] -= 1.0

    dX = [dxl_eff[:, :, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)]
    dX.extend([dxr_eff[:, :, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2, order - 1)])
    dX.append(dxr_eff[:, :, order:])
    x = reduce(np.add, (dxl_eff[:, :, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)))
    K_xr = L_coeff(dX[:], x=x)
    K_xr[int(order / 2)] -= 1.0
    
    dY = [dyl_eff[:, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)]
    dY.extend([dyr_eff[:, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2, order - 1)])
    dY.append(dyr_eff[:, order:])
    y = reduce(np.add, (dyl_eff[:, i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)))
    K_yr = L_coeff(dY[:], x=y)
    K_yr[int(order / 2)] -= 1.0

    dZ = [dzl_eff[i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)]
    dZ.extend([dzr_eff[i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2, order - 1)])
    dZ.append(dzr_eff[order:])
    x = reduce(np.add, (dzl_eff[i + 1:(-2 * (ng - 1) + i)]  for i in range((order - 1) / 2)))
    K_zr = L_coeff(dZ, x=x)
    K_zr[int(order / 2)] -= 1.0

    recon_coeff = [K_xl, K_yl, K_zl, K_xr, K_yr, K_zr]

    return recon_coeff


def ENO_stencils(dx_eff, dy_eff, dz_eff, order, n_stencil=3):
    """
    Computes the local pseudo grid coordinates of the stencils
    used by the ENO/WENO reconstruction.

    dx_eff, dy_eff, dz_eff... effective grid spacings of the pseudo grid
    order... order of reconstruction (of the individual stencils, not the combined)
    n_stencil... number of stencils in each dimension

    Xs... returned lists of local coordinates of all stencils
    """
 
    global ng

    nc = dx_eff.shape[2] - 2 * ng
    nr = dy_eff.shape[1] - 2 * ng
    nz = dz_eff.shape[0] - 2 * ng
    

    Xs_xl = []
    Xs_yl = []
    Xs_zl = []
    Xs_xr = []
    Xs_yr = []
    Xs_zr = []   

    for s in range(n_stencil):
        # positive upwind stencils
        Xs_xl.append([reduce(np.add, [dx_eff[:, :, s + k:s + k + nc + 1] for k in range(i)] + [dx_eff[:, :, s + i:s + i + nc + 1] / 2.0]) for i in range(order)])
        Xs_yl.append([reduce(np.add, [dy_eff[:, s + k:s + k + nr + 1] for k in range(i)] + [dy_eff[:, s + i:s + i + nr + 1] / 2.0]) for i in range(order)])
        Xs_zl.append([reduce(np.add, [dz_eff[s + k:s + k + nz + 1] for k in range(i)] + [dz_eff[s + i:s + i + nz + 1] / 2.0]) for i in range(order)]) 
        # negative upwind stencils

    lb = 2 * ng - order
    Xs_xr.append([reduce(np.add, [dx_eff[:, :, k + lb:k + lb + nc + 1] for k in range(i)] + [dx_eff[:, :, i + lb:i + lb + nc + 1] / 2.0]) for i in range(order)])
    Xs_yr.append([reduce(np.add, [dy_eff[:, k + lb:k + lb + nr + 1] for k in range(i)] + [dy_eff[:, i + lb:i + lb + nr + 1] / 2.0]) for i in range(order)])
    Xs_zr.append([reduce(np.add, [dz_eff[k + lb:k + lb + nz + 1] for k in range(i)] + [dz_eff[i + lb:i + lb + nz + 1] / 2.0]) for i in range(order)])

    for s in range(n_stencil - 1):
        Xs_xr.append(Xs_xl[-s - 1])
        Xs_yr.append(Xs_yl[-s - 1])
        Xs_zr.append(Xs_zl[-s - 1])

    Xs = [Xs_xl, Xs_yl, Xs_zl, Xs_xr, Xs_yr, Xs_zr]
    return Xs


def ENO_coeff(dx_eff, dy_eff, dz_eff, order, n_stencil=3):
    """
    Computes the linear reconstruction coefficients of the stencils
    used by the ENO/WENO reconstruction.

    dx_eff, dy_eff, dz_eff... effective grid spacings of the pseudo grid
    order... order of reconstruction (of the individual stencils, not the combined)
    n_stencil... number of stencils in each dimension

    Ks... returned lists of reconstruction coefficients of all stencils
    """

    global ng

    nc = dx_eff.shape[2] - 2 * ng
    nr = dy_eff.shape[1] - 2 * ng
    nz = dz_eff.shape[0] - 2 * ng

    Ks_xl = []
    Ks_xr = [] 
    Ks_yl = []
    Ks_yr = []
    Ks_zl = []
    Ks_zr = []
    
    for s in range(n_stencil): 
        # positive upwind stencils
        dX = [dx_eff[:, :, i + s:i + s + nc + 1]  for i in range(order)] 
        x = reduce(np.add, [dx_eff[:, :, i + s:i + s + nc + 1]  for i in range(ng - s)] + [np.zeros_like(dx_eff[:, :, i + s:i + s + nc + 1])])       
        K_xl = L_coeff(dX, x=x)            
        Ks_xl.append(K_xl)
        dX = [dy_eff[:, i + s:i + s + nr + 1]  for i in range(order)]
        x = reduce(np.add, [dy_eff[:, i + s:i + s + nr + 1]  for i in range(ng - s)] + [np.zeros_like(dy_eff[:, i + s:i + s + nr + 1])])
        K_yl = L_coeff(dX, x=x)
        Ks_yl.append(K_yl)
        dX = [dz_eff[i + s:i + s + nz + 1]  for i in range(order)]
        x = reduce(np.add, [dz_eff[i + s:i + s + nz + 1]  for i in range(ng - s)] + [np.zeros_like(dz_eff[i + s:i + s + nz + 1])])
        K_zl = L_coeff(dX, x=x)
        Ks_zl.append(K_zl)



        # negative upwind stencils 
    lb = 2 * ng - order# - s
        
    dX = [dx_eff[:, :, i + lb:i + lb + nc + 1]  for i in range(order)]
    x = reduce(np.add, [dx_eff[:, :, i + lb:i + lb + nc + 1]  for i in range(ng - lb)] + [np.zeros_like(dx_eff[:, :, i + s:i + s + nc + 1])])        
    K_xr = L_coeff(dX, x=x)
    Ks_xr.append(K_xr)     
    dX = [dy_eff[:, i + lb:i + lb + nr + 1]  for i in range(order)] 
    x = reduce(np.add, [dy_eff[:, i + lb:i + lb + nr + 1]  for i in range(ng - lb)] + [np.zeros_like(dy_eff[:, i + s:i + s + nr + 1])])
    K_yr = L_coeff(dX, x=x)
    Ks_yr.append(K_yr)
    dX = [dz_eff[i + lb:i + lb + nz + 1]  for i in range(order)]
    x = reduce(np.add, [dz_eff[i + lb:i + lb + nz + 1]  for i in range(ng - lb)] +  [np.zeros_like(dz_eff[i + s:i + s + nz + 1])])
    K_zr = L_coeff(dX, x=x)
    Ks_zr.append(K_zr)       

    for s in range(1, n_stencil):
        Ks_xr.append(Ks_xl[-s])
        Ks_yr.append(Ks_yl[-s])
        Ks_zr.append(Ks_zl[-s])

    Ks = [Ks_xl, Ks_yl, Ks_zl, Ks_xr,  Ks_yr, Ks_zr]
    return Ks



def L_coeff(dX, x=0):
    """
    Evaluates the coefficients for
    the reconstruction polynomial of 
    order len(dX) = k at x.

    dX... List of numpy arrays corresponding to the grid sizes of the interpolation stencil
    x... Evaluation point
    C... returned list of same structure as dX, containing the coefficients to reconstruct the value at position x:
         V(x) = sum_k(v[i] * C[i])
    """

    k = len(dX)

    X = [0.0]
    for l in range(k):
       X.append(X[-1] + dX[l])

    C = [np.zeros_like(dX[0]) for j in range(k)]
    for j in range(k):
        for m in range(j + 1, k + 1):
            sumk = np.zeros_like(dX[0])
            pk2 = np.ones_like(dX[0])
            for l in range(k + 1):
                pk = np.ones_like(dX[0])
                if not l ==m:
                    for q in range(k + 1):
                        if (not q == m) and (not q == l): 
                            pk = pk * (x - X[q])
                    sumk += pk
               
                    pk2 = pk2 * (X[m] - X[l])
            C[j] += sumk / pk2 
        C[j] = C[j] * (dX[j]) 
    
    return C    
    

    return [C1, C2, C3]

def divided_differences(X, Y):
    """
    Computes the Newton divided
    differences as the smoothing
    indicator for the ENO-stencil 
    selection algorithm.

    X... local coordinates of the pseudo-grid stencil
    Y... values of the field to reconstruct at same position of X 

    div_diff... returned divided differences
    """

    if len(Y) == 1:
        div_diff = Y[0]
    else:
        div_diff = (divided_differences(X[1:], Y[1:]) - divided_differences(X[:-1], Y[:-1])) / (X[-1] - X[0])

    return div_diff

def undivided_differences(Y):
    """
    Computes the Newton undivided
    differences as the smoothing
    indicator for the ENO-stencil 
    selection algorithm.

    Y... values of the field to reconstruct at the reconstruction stencil locations
    
    udiv_diff... returned undivided differences
    """    

    if len(Y) == 1:
        udiv_diff = Y[0]
    else:
        udiv_diff = undivided_differences(Y[1:]) - undivided_differences(Y[:-1])
    return udiv_diff


def smoothness_indicators(Y):
    """
    Computes the smoothness indicators for 
    the fast FWENO scheme (Baeza et al., 2018).

    Y... values of the field to reconstruct at the reconstruction stencil locations

    smooth_ind... returned smoothness indicators
    """

    smooth_ind = reduce(np.add, [(Y[i + 1] - Y[i]) ** 2 for i in range(len(Y) - 1)])
    return smooth_ind



def WENO_weights(linear_wghts, smo, undiv, order_sub):
    """
    Applies the fast non-linear FWENO weights based
    on Baeza et al. (2018)

    linear_wghts... linear reconstruction coefficients
    smo... smoothness indicators
    undiv... undivided differences
    order_sub... order of the reconstruction stencils (not the combined order)

    alphas... returned WENO-reconstruction weights of higher order
    """

    eps = 1e-6
    n_stencil = len(smo)
    s2 = 10
    alphas = [linear_wghts[i] * (1.0 + (undiv / (smo[i] + eps))) ** s2 for i in range(n_stencil)]
    sum_alpa = reduce(np.add, alphas)
    alphas = [alpha / sum_alpa for alpha in list(alphas)]

    return alphas


def ENO_decide_stencil(div_diff, side='l'):
    """
    Selection of optimal stencil based on the
    smallest absolute divided differences.

    div_diff... divided diffferences
    side... direction of upwinding ('l', 'r'

    Cs... returned lists of ENO coefficients (zeros or ones) 
    """

    Cs = []
    n_stencil = len(div_diff)
#    b_fac = [1.0, 0.5]#, 0.01, 1.0, 0.0] # bias factors to favour the standard upwind-biased stencil
   
    for n in range(n_stencil):
        a = [np.zeros_like(div_diff[0], dtype=float)]
        a.extend(Cs)
        a = reduce(np.add, a)
        for k in range(n + 1, n_stencil):
#            a += np.maximum(div_diff[n] * b_fac[n] - div_diff[k] * b_fac[k], 0.0)
            a += np.maximum(div_diff[n] - div_diff[k], 0.0)
        C = np.logical_not(a)
        Cs.append(np.array(C, dtype=float))
    return Cs



def linear_weights(K_h, Ks_l):
    """
    Derives the linear coefficients for the
    convex combination of n smooth stencils
    of order k to a stencil of order n + k - 1.

    K_h... Coefficients of the high-order stencil.
    Ks_l... Coefficients lists of the low-order stencils. 

    D... returned list of linear weights
    """
 

    eps = 1e-40
    order_l = len(Ks_l[0])            
    D = []
    for n in range(len(Ks_l)):
        D.append((K_h[n] - reduce(np.add, [D[-1 - i] * Ks_l[n - 1 - i][i + 1] for i in range(min(len(D), order_l - 1))] + [np.zeros_like(K_h[n])])) / (Ks_l[n][0] + eps))
    return D
