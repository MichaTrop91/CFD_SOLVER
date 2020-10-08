# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# load external python packages
import numpy as np
from netCDF4 import Dataset
import scipy
from scipy.sparse import eye, diags
import sps_operators as ops
from time import time
from copy import deepcopy

# load model specific *py files
import domain_decomp as ddcp
from iterschemes import csr_gauss_seidel, csr_gauss_seidel_reverse



def init_multigrid_pressure_solver(comm, grid_flds, grid_flds_sub, param_dict):
    """
    This function is used to initialize the multigrid pressure solver.
    The tasks include
    - definition of the coarse-grids
    - definition and organization of subdomains and initialization of communication objects
    - for DCA, the fine-grid fields like volume and areas are coarsened
      by agglomeration and distributed to the subdomains.
    - for GCA, the fine-grid operator is collected from existing operator parts
      when the subdomains don't overlap.
    - construction of fine and coarse-grid operators by direct discretization (DCA)
      or Galerkin interpolation (GCA)
    - construction of inter-grid transfer operators (prolongation, restriction)
    - construction of smoother
    - initialization of boundary objects for communication in smoothing
   
    comm... communicator
    grid_flds... grid fields of fine grid on root node
    grid_flds_sub... distributed fine-grid fields
    param_dict... parameter dictionary
    """


    # definition of globals used repeatedly in multiple functions of this script

    global PROL_lev, REST_lev
    global MAT_lev
    global GRAD_lev
    global GRAD_dirich
    global DIV_lev
    global DIV_contra_lev
    global TRANSx_contra_lev, TRANSy_contra_lev
    global TRANS_gradx_lev, TRANS_grady_lev
    global bounds_ext
    global bounds_s_lev, bounds_r_lev, bounds_r_tmp_lev
    global bounds_bl_s_lev, bounds_bl_r_lev, bounds_bl_r_tmp_lev
    global bounds_rd_s_lev, bounds_rd_r_lev, bounds_rd_r_tmp_lev
    global vols1d_eff_lev, areas_eff_lev, vols_eff_lev, vols1d_lev

    global subdomain_from_lev, subdomain_same, subdomain_to_lev
    global X_tmp_shp, X_tmp
    global dom_bnds_lev

    global MPICOMM_lev
    global pids_lev
    global nr_lev, nc_lev, nz_lev
    global npr_lev, npc_lev
    global nri_lev, ncj_lev

    global set_zeros

    global ngrids

    global X_comp_lev
    global Y_comp_lev
    global z_comp_lev

    global dsurfdx, dsurfdy

    global SMO_lev
    global Dinv_lev
    global RDBL_MAT_lev
    global RD_lev, BL_lev
    global BL_inds_lev, RD_inds_lev

    global smooth_type

    global ng, ng1

    global fld_inds_c, fld_inds_u, fld_inds_v, fld_inds_w
    global shp_c, shp_u, shp_v, shp_w
    global fld_inds_n1_c, fld_inds_n1_u, fld_inds_n1_v, fld_inds_n1_w
    global shp_n1_c, shp_n1_u, shp_n1_v, shp_n1_w
    global sl_c

    mpicomm = comm.mpicomm
    npr = comm.npr
    npc = comm.npc
    nnodes = npr * npc

    nz = comm.nz
    nri = deepcopy(comm.nri)
    ncj = deepcopy(comm.ncj)

    pids = comm.pids
    rank = mpicomm.Get_rank()
    pid = pids.index(rank)
    pid_r = int(pid / npc)
    pid_c = pid - pid_r * npc

    ng = int(param_dict['n_ghost'])
    ng1 = ng - 1    

    area_effx_sub, area_effy_sub, area_effz_sub = deepcopy(grid_flds_sub[:3])
    areax_sub, areay_sub, areaz_sub = deepcopy(grid_flds_sub[3:6])
    vols_eff_sub = grid_flds_sub[6].copy()
    vols_sub = grid_flds_sub[7].copy()

    dginvx_sub, dginvy_sub, dginvz_sub = deepcopy(grid_flds_sub[8:11])
    dsurfdx_sub, dsurfdy_sub = grid_flds_sub[11:13]

    # Boundary conditions for the pressure solver:
    # It makes only sense to use Neumann boundary conditions for non-cyclic simulations
    # and cyclic boundary conditions for cyclic simulations.
    # In case of Neumann boundary conditions, the solver will only converge, if
    # the integrability condition of the velocity boundary condition is satisfied, 
    # i.e. the mass influx is balanced by the outflux 
    # integrated over the whole domain surface within the given error tolerance.

    bnd_pres_x = param_dict['bnd_pres_x']
    bnd_pres_y = param_dict['bnd_pres_y']
    bnd_pres_z = param_dict['bnd_pres_z']

    nr_sub = nri[pid_r + 1] - nri[pid_r]
    nc_sub = ncj[pid_c + 1] - ncj[pid_c]

    jc_st_glob = ng
    jc_end_glob = -ng
    ir_st_glob = ng
    ir_end_glob = -ng
    k_st_glob = ng
    k_end_glob = -ng

    nr_sub = nri[pid_r + 1] - nri[pid_r]
    nc_sub = ncj[pid_c + 1] - ncj[pid_c]
    
    k_st = ng
    ir_st = ng
    jc_st = ng

    if bnd_pres_x != 'cyclic':
        ncj = [ nc + 1 for nc in ncj ]
        ncj[0] = 0
        ncj[-1] += 1
        if pid_c == 0:
            jc_st = ng1
            nc_sub += 1
        if pid_c == npc - 1:
            nc_sub += 1
        jc_st_glob -= 1
        jc_end_glob += 1

    if bnd_pres_y != 'cyclic':
        nri = [ nr + 1 for nr in nri ]
        nri[0] = 0
        nri[-1] += 1
        if pid_r == 0:
            ir_st = ng1
            nr_sub += 1
        if pid_r == npr - 1:
            nr_sub += 1
        ir_st_glob -= 1
        ir_end_glob += 1

    if bnd_pres_z != 'cyclic':
        k_st = ng1
        nz += 2
        k_st_glob -= 1
        k_end_glob +=  1 

    jc_end = jc_st + nc_sub
    ir_end = ir_st + nr_sub
    k_end = k_st + nz
    
    sl_c = [(k_st, k_end), (ir_st, ir_end), (jc_st, jc_end)]

    fld_inds_c = list(
                         np.array([ (k, i, j) for k in range(k_st, k_end) 
                         for i in range(ir_st, ir_end) for j in range(jc_st, jc_end) ]).T
                     )
    fld_inds_u = list(
                         np.array([ (k, i, j) for k in range(k_st, k_end) 
                         for i in range(ir_st, ir_end) for j in range(jc_st, jc_end + 1) ]).T
                     )
    fld_inds_v = list(
                         np.array([ (k, i, j) for k in range(k_st, k_end) 
                         for i in range(ir_st, ir_end + 1) for j in range(jc_st, jc_end) ]).T
                     )
    fld_inds_w = list(
                         np.array([ (k, i, j) for k in range(k_st, k_end + 1) 
                         for i in range(ir_st, ir_end) for j in range(jc_st, jc_end) ]).T
                     )

    shp_c = [len(set(fld_inds_c[0])), len(set(fld_inds_c[1])), len(set(fld_inds_c[2]))]
    shp_u = [len(set(fld_inds_u[0])), len(set(fld_inds_u[1])), len(set(fld_inds_u[2]))]
    shp_v = [len(set(fld_inds_v[0])), len(set(fld_inds_v[1])), len(set(fld_inds_v[2]))]
    shp_w = [len(set(fld_inds_w[0])), len(set(fld_inds_w[1])), len(set(fld_inds_w[2]))]

    fld_inds_n1_c = list(
                            np.array([ (k, i, j) for k in range(k_st - 1, k_end + 1) 
                            for i in range(ir_st - 1, ir_end + 1) for j in range(jc_st - 1, jc_end + 1) ]).T
                        )
    fld_inds_n1_u = list(
                            np.array([ (k, i, j) for k in range(k_st - 1, k_end + 1) 
                            for i in range(ir_st - 1, ir_end + 1) for j in range(jc_st - 1, jc_end + 2) ]).T
                        )
    fld_inds_n1_v = list(
                            np.array([ (k, i, j) for k in range(k_st - 1, k_end + 1) 
                            for i in range(ir_st - 1, ir_end + 2) for j in range(jc_st - 1, jc_end + 1) ]).T
                        )
    fld_inds_n1_w = list(
                            np.array([ (k, i, j) for k in range(k_st - 1, k_end + 2) 
                            for i in range(ir_st - 1, ir_end + 1) for j in range(jc_st - 1, jc_end + 1) ]).T
                        )

    shp_n1_c = [len(set(fld_inds_n1_c[0])), len(set(fld_inds_n1_c[1])), len(set(fld_inds_n1_c[2]))]
    shp_n1_u = [len(set(fld_inds_n1_u[0])), len(set(fld_inds_n1_u[1])), len(set(fld_inds_n1_u[2]))]
    shp_n1_v = [len(set(fld_inds_n1_v[0])), len(set(fld_inds_n1_v[1])), len(set(fld_inds_n1_v[2]))]
    shp_n1_w = [len(set(fld_inds_n1_w[0])), len(set(fld_inds_n1_w[1])), len(set(fld_inds_n1_w[2]))]

    nr = nri[-1]
    nc = ncj[-1]
    
    ncrop = 1
    
    if jc_end_glob + ncrop == 0:
        jc_end_glob_crop = None
    else:
        jc_end_glob_crop = jc_end_glob + ncrop
    if ir_end_glob + ncrop == 0:
        ir_end_glob_crop = None
    else:
        ir_end_glob_crop = ir_end_glob + ncrop
    if k_end_glob + ncrop == 0:
        k_end_glob_crop = None
    else:
        k_end_glob_crop = k_end_glob + ncrop  
 

    x2_fine = param_dict['x2coord_ghst'][jc_st_glob - ncrop:jc_end_glob_crop]
    y2_fine = param_dict['y2coord_ghst'][ir_st_glob - ncrop:ir_end_glob_crop]
    z2_fine = param_dict['z2coord_ghst'][k_st_glob - ncrop:k_end_glob_crop]    

    dim_size_max = max((x2_fine.size, y2_fine.size, z2_fine.size))    

    ngrids_max = min(int(np.log(dim_size_max) / np.log(2)), int(param_dict['ngrids_max']))
    
    grid_edges_lists = define_coarsening(x2_fine, y2_fine, z2_fine, rank, dim_size_threshold=5, ngrids_max=ngrids_max)

    x2_list = grid_edges_lists[0]
    y2_list = grid_edges_lists[1]
    z2_list = grid_edges_lists[2]        
    ngrids = len(z2_list)

    vols1d_eff_lev = [vols_eff_sub[fld_inds_n1_c]]
    vols_lev = [vols_sub[fld_inds_n1_c].reshape(shp_n1_c)]
    vols_eff_lev = [vols_eff_sub[fld_inds_n1_c].reshape(shp_n1_c)]
    areas_eff_lev = [ops.put_1d([
                                    area_effx_sub[fld_inds_n1_u].reshape(shp_n1_u), 
                                    area_effy_sub[fld_inds_n1_v].reshape(shp_n1_v), 
                                    area_effz_sub[fld_inds_n1_w].reshape(shp_n1_w)
                                ])]
    dginv_lev = [ops.put_1d([
                                dginvx_sub[fld_inds_n1_u].reshape(shp_n1_u), 
                                dginvy_sub[fld_inds_n1_v].reshape(shp_n1_v), 
                                dginvz_sub[fld_inds_n1_w].reshape(shp_n1_w)
                            ])]
    areas = ops.put_1d([
                           areax_sub[fld_inds_n1_u].reshape(shp_n1_u), 
                           areay_sub[fld_inds_n1_v].reshape(shp_n1_v), 
                           areaz_sub[fld_inds_n1_w].reshape(shp_n1_w)
                       ])
    vols1d = vols_sub[fld_inds_n1_c]

    area_effx_lev = [area_effx_sub[fld_inds_n1_u].reshape(shp_n1_u)]
    area_effy_lev = [area_effy_sub[fld_inds_n1_v].reshape(shp_n1_v)]
    area_effz_lev = [area_effz_sub[fld_inds_n1_w].reshape(shp_n1_w)]
    dginvx = dginvx_sub[fld_inds_n1_u].reshape(shp_n1_u)
    dginvy = dginvy_sub[fld_inds_n1_v].reshape(shp_n1_v)
    dginvz = dginvz_sub[fld_inds_n1_w].reshape(shp_n1_w)
    dsurfdx_lev = [dsurfdx_sub[fld_inds_n1_u].reshape(shp_n1_u)]
    dsurfdy_lev = [dsurfdy_sub[fld_inds_n1_v].reshape(shp_n1_v)]
    dginv_comp_lev = [[dginvx, dginvx, dginvy, dginvy, dginvz, dginvz]]

    nr_lev = [nr]
    nc_lev = [nc]
    nz_lev = [nz]    

    pids_lev = [pids]

    npr_lev = [npr]
    npc_lev = [npc]

    nri_lev = [nri]
    ncj_lev = [ncj]

    # Define the multigrid coarse-levels

    nr_f, nc_f, nz_f = nr, nc, nz
    nr_c, nc_c, nz_c = nr, nc, nz

    for n in range(ngrids - 1):

        nc_c = x2_list[n + 1].size - 1 - 2 * ncrop
        nr_c = y2_list[n + 1].size - 1 - 2 * ncrop
        nz_c = z2_list[n + 1].size - 1 - 2 * ncrop

        nri_c, ncj_c, npr_c, npc_c, pids_c = ddcp.coarse_decomp(
                                                                   y2_list[n][ncrop:y2_list[n].size - ncrop], 
                                                                   x2_list[n][ncrop:x2_list[n].size - ncrop], 
                                                                   y2_list[n + 1][ncrop:y2_list[n + 1].size - ncrop], 
                                                                   x2_list[n + 1][ncrop:x2_list[n + 1].size - ncrop], 
                                                                   nri_lev[-1], ncj_lev[-1], npr_lev[-1], npc_lev[-1], pids_lev[-1], ncells_min=5
                                                               )

        nz_lev.append(nz_c)
        nri_lev.append(nri_c)
        ncj_lev.append(ncj_c)
        npr_lev.append(npr_c)
        npc_lev.append(npc_c)
        pids_lev.append(pids_c)

        nr_f, nc_f, nz_f = nri_c[-1], ncj_c[-1], nz_c


    # Use discretized coarse grid approximation as long as possible (until any of the grid sizes becomes odd)
    # and proceed with Galerkin coarse grid approximation
    n_DCA = 0
    for n in range(ngrids - 1):
        if all((nz_lev[n] % 2  == 0, (nri_lev[n][-1] -  nri_lev[n][0]) % 2  == 0, (ncj_lev[n][-1] -  ncj_lev[n][0]) % 2  == 0)):
            n_DCA += 1
        else:
            break

    # Organize the subdomains

    subdomain_from_lev = []
    subdomain_same = []
    subdomain_to_lev = []

    X_tmp_shp = []
    X_tmp = []
    dom_bnds_lev = []

    for n in range(ngrids - 1):

        out_lsts = ddcp.org_subdomains(
                                          mpicomm, 
                                          y2_list[n][ncrop:y2_list[n].size - ncrop], 
                                          x2_list[n][ncrop:x2_list[n].size - ncrop], 
                                          y2_list[n + 1][ncrop:y2_list[n + 1].size - ncrop], 
                                          x2_list[n + 1][ncrop:x2_list[n + 1].size - ncrop], 
                                          nri_lev[n], ncj_lev[n], nz_lev[n], nz_lev[n + 1], nri_lev[n + 1], ncj_lev[n + 1], 
                                          pids_lev[n], pids_lev[n + 1]
                                      )

        flds_s, flds_r, fld_same, field_tmp, dom_bnds = out_lsts[:]

        subdomain_to_lev.append(flds_r)
        subdomain_from_lev.append(flds_s)
        subdomain_same.append(fld_same)
        X_tmp.append(field_tmp)        
        dom_bnds_lev.append(dom_bnds)

        if isinstance(field_tmp, np.ndarray):
            X_tmp.append(field_tmp.flatten())
            X_tmp_shp.append(field_tmp.shape)

    # Setup new MPI communicators, which exclude iddle processes

    MPICOMM_lev = []

    for n in range(ngrids):
        group = mpicomm.Get_group()
        newgroup = group.Incl(pids_lev[n])
        newcomm = mpicomm.Create(newgroup)
        MPICOMM_lev.append(newcomm)  

    nlev = 0

    for n in range(0, ngrids):
        if rank in pids_lev[n]:
            ind_plev = pids_lev[n].index(rank)
            ind_prlev = int(ind_plev / npc_lev[n])
            ind_pclev = ind_plev - ind_prlev * npc_lev[n]
            nlev += 1
            if n > 0:
                nr_lev.append(nri_lev[n][ind_prlev + 1] - nri_lev[n][ind_prlev])
                nc_lev.append(ncj_lev[n][ind_pclev + 1] - ncj_lev[n][ind_pclev])

    # These are the original grid fields all located on the root node
    if rank == 0:
      
        grid_flds = [fld[k_st_glob - ncrop:k_end_glob_crop, ir_st_glob - ncrop:ir_end_glob_crop, jc_st_glob - ncrop:jc_end_glob_crop] for fld in grid_flds]
        ffx, ffy, ffz = deepcopy(grid_flds[:3])

        fvol = grid_flds[3]
        area_x, area_y, area_z = grid_flds[4:7]
        vols = grid_flds[7]
        dginv_x, dginv_y, dginv_z = grid_flds[8:11]
        dhsurfdx, dhsurfdy = grid_flds[-2:]

        ff1d_f = ops.put_1d([ffx, ffy, ffz])
        fvol1d_f = fvol.flatten()
        areas1d_f = ops.put_1d([area_x, area_y, area_z])
        areas_eff1d_f = ff1d_f * areas1d_f
        vols1d_f = vols.flatten()
        vols_eff1d_f = fvol1d_f * vols1d_f
        dsurf1d_f = ops.put_1d([dhsurfdx, dhsurfdy, np.zeros_like(area_z)])

        dginv_x0, dginv_y0, dginv_z0 = dginv_x, dginv_y, dginv_z
        area_effx_0, area_effy_0, area_effz_0 = ops.put_3d(areas_eff1d_f, vols.shape)
        vols_eff_0 = vols_eff1d_f.reshape(vols.shape)
        dinv_fine_comp = [dginv_x0, dginv_x0, dginv_y0, dginv_y0, dginv_z0, dginv_z0]

    for n in range(1, n_DCA + 1):   
        if rank == 0: 
            nr_f = nri_lev[n - 1][-1]
            nc_f = ncj_lev[n - 1][-1]
            nz_f = nz_lev[n - 1]
            nr_c = nri_lev[n][-1]
            nc_c = ncj_lev[n][-1]
            nz_c = nz_lev[n]
            grid_fields_f = [areas_eff1d_f, vols_eff1d_f, dsurf1d_f, vols1d_f]
            edges_fine = [x2_list[n - 1], y2_list[n - 1], z2_list[n - 1]]
            edges_coarse = [x2_list[n], y2_list[n], z2_list[n]]
            grid_fields_c = coarse_grid_fields(grid_fields_f, edges_fine, edges_coarse)
            areas_eff1d_c, vols_eff1d_c, dsurf1d_c, vols1d_c = grid_fields_c[:]                    
            vols_u = ops.vol_stag(vols_eff1d_c, 'u', nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop)
            vols_v = ops.vol_stag(vols_eff1d_c, 'v', nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop)
            vols_w = ops.vol_stag(vols_eff1d_c, 'w', nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop)
            vols_stag = vols_u + vols_v + vols_w

            dinv_coarse_comp = coarse_gradient_coeff(dinv_fine_comp, vols_eff1d_f, vols_eff1d_c, areas_eff1d_f, areas_eff1d_c, edges_fine, edges_coarse)
            dginvx_l_c, dginvx_r_c, dginvy_l_c, dginvy_r_c, dginvz_l_c, dginvz_r_c = dinv_coarse_comp[:]

            area_effx_c, area_effy_c, area_effz_c = ops.put_3d(areas_eff1d_c, [nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop])
            dsurfdx_c, dsurfdy_c, dsurfdz_c = ops.put_3d(dsurf1d_c, [nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop])
 
            vols_eff_c = vols_eff1d_c.reshape(nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop)
            vols_c = vols1d_c.reshape(nz_c + 2 * ncrop, nr_c + 2 * ncrop, nc_c + 2 * ncrop)
            grid_flds_c = [
                              area_effx_c, area_effy_c, area_effz_c, vols_eff_c, vols_c, dginvx_l_c, dginvx_r_c, dginvy_l_c, 
                              dginvy_r_c, dginvz_l_c, dginvz_r_c, dsurfdx_c, dsurfdy_c
                          ]

        else:
  
            grid_flds_c = [ None for m in range(13) ]

        if rank in pids_lev[n]:

            ind_plev = pids_lev[n].index(rank)
            ind_prlev = int(ind_plev / npc_lev[n])
            ind_pclev = ind_plev - ind_prlev * npc_lev[n]

            ncj_st = ncj_lev[n][:-1]
            ncj_end = [nc_ + 2 * ncrop for nc_ in ncj_lev[n][1:]]
            nri_st = nri_lev[n][:-1]
            nri_end = [nr_ + 2 * ncrop for nr_ in nri_lev[n][1:]]
                
            nr_sub_tmp = nri_lev[n][ind_prlev + 1] - nri_lev[n][ind_prlev] + 2 * ncrop
            nc_sub_tmp = ncj_lev[n][ind_pclev + 1] - ncj_lev[n][ind_pclev] + 2 * ncrop
            nz_sub_tmp = nz_lev[n] + 2 * ncrop

            grid_flds_sub_lev = []
           
            area_eff_x = np.ones([nz_sub_tmp , nr_sub_tmp, nc_sub_tmp + 1])
            area_eff_y = np.ones([nz_sub_tmp, nr_sub_tmp + 1, nc_sub_tmp])
            area_eff_z = np.ones([nz_sub_tmp + 1, nr_sub_tmp, nc_sub_tmp])
            vols_eff = np.ones([nz_sub_tmp, nr_sub_tmp, nc_sub_tmp])
            vols = np.ones([nz_sub_tmp, nr_sub_tmp, nc_sub_tmp])
            dginvx_l = np.ones([nz_sub_tmp, nr_sub_tmp, nc_sub_tmp + 1])
            dginvy_l = np.ones([nz_sub_tmp, nr_sub_tmp + 1, nc_sub_tmp])
            dginvz_l = np.ones([nz_sub_tmp + 1, nr_sub_tmp, nc_sub_tmp])
            dginvx_r = np.ones([nz_sub_tmp, nr_sub_tmp, nc_sub_tmp + 1])
            dginvy_r = np.ones([nz_sub_tmp, nr_sub_tmp + 1, nc_sub_tmp])
            dginvz_r = np.ones([nz_sub_tmp + 1, nr_sub_tmp, nc_sub_tmp])
            dsurfdx = np.ones([nz_sub_tmp, nr_sub_tmp, nc_sub_tmp + 1])
            dsurfdy = np.ones([nz_sub_tmp, nr_sub_tmp + 1, nc_sub_tmp])
            
            grid_flds_sub_lev = [
                                     area_eff_x, area_eff_y, area_eff_z, vols_eff, vols, 
                                     dginvx_l, dginvx_r, dginvy_l, dginvy_r, dginvz_l, dginvz_r, dsurfdx, dsurfdy
                                ]

            for k, fld in enumerate(grid_flds_c):
                ddcp.distribute_data(mpicomm, fld, grid_flds_sub_lev[k], nri_st, nri_end, ncj_st, ncj_end, pids_lev[n])

            areas_eff_lev.append(ops.put_1d([area_eff_x, area_eff_y, area_eff_z]))
            vols_eff_lev.append(vols_eff)
            vols1d_eff_lev.append(vols_eff.flatten())
            vols_lev.append(vols)
            area_effx_lev.append(area_eff_x)
            area_effy_lev.append(area_eff_y)
            area_effz_lev.append(area_eff_z)
            dginv_comp_lev.append([dginvx_l, dginvx_r, dginvy_l, dginvy_r, dginvz_l, dginvz_r])
            dsurfdx_lev.append(dsurfdx)
            dsurfdy_lev.append(dsurfdy)

        if rank == 0:

            vols_eff1d_f = vols_eff1d_c
            vols1d_f = vols1d_c
            areas_eff1d_f = areas_eff1d_c
            dsurf1d_f = dsurf1d_c
            dinv_fine_comp = dinv_coarse_comp

    # make the operators and related objects
    
    MAT_lev = []
    MAT_ghost_lev = []
    DIV_lev = []
    DIV_contra_lev = []
    GRAD_lev = []
    
    SMO_lev = []

    Dinv_lev = []
    RDBL_MAT_lev = []
    RD_lev = []
    BL_lev = []

    X = []
    R = []
    B = []


    # Make the prolongation and restriction operators for cell-centred fields

    PROL_lev = []
    REST_lev = []

    for n in range(min(nlev, ngrids - 1)):
        edges_fine = [
                          x2_list[n][ncrop:x2_list[n].size - ncrop], 
                          y2_list[n][ncrop:y2_list[n].size - ncrop], 
                          z2_list[n][ncrop:z2_list[n].size - ncrop]
                     ]
        edges_coarse = [
                           x2_list[n + 1][ncrop:x2_list[n + 1].size - ncrop], 
                           y2_list[n + 1][ncrop:y2_list[n + 1].size - ncrop], 
                           z2_list[n + 1][ncrop:z2_list[n + 1].size - ncrop]
                       ] 
 
        SCAL = ops.REST_T_weight_scaling(edges_fine, edges_coarse, dom_bnds_lev[n])
        REST = ops.make_restr(edges_fine, edges_coarse, dom_bnds_lev[n], order='linear')
        REST_T = (ops.make_restr(edges_fine, edges_coarse, dom_bnds_lev[n], order='linear')).transpose(copy=True)
        PROL = REST_T * SCAL

        REST_lev.append(REST)
        PROL_lev.append(PROL)

    #Construct discretization coarse-grid approximation
    for n in range(min(n_DCA + 1, nlev)):

        ind_plev = pids_lev[n].index(rank)
        ind_prlev = int(ind_plev / npc_lev[n])
        ind_pclev = ind_plev - ind_prlev * npc_lev[n]
        

        grid_fields_sub = [
                              vols_eff_lev[n], area_effx_lev[n], area_effy_lev[n], area_effz_lev[n], 
                              dginv_comp_lev[n], dsurfdx_lev[n], dsurfdy_lev[n]
                          ]

        bnd_cond_neumann_lst = [False for side in range(6)]
        if bnd_pres_x == 'neumann' and ind_pclev == 0:
            bnd_cond_neumann_lst[0] = True
        if bnd_pres_x == 'neumann' and ind_pclev == npc_lev[n] - 1:
            bnd_cond_neumann_lst[1] = True
        if bnd_pres_y == 'neumann' and ind_prlev == 0:
            bnd_cond_neumann_lst[2] = True
        if bnd_pres_y == 'neumann' and ind_prlev == npr_lev[n] - 1:
            bnd_cond_neumann_lst[3] = True
        if bnd_pres_z == 'neumann':
            bnd_cond_neumann_lst[4] = True
            bnd_cond_neumann_lst[5] = True

 
        OP_list = ops.make_div_grad(grid_fields_sub, z2_list, bnd_cond_neumann_lst, level=n)
        DIV, GRAD, DIV_contra, GRAD_metr = OP_list[:]

        if n == 0:
            GRAD_dirich = ops.implement_neumann_bc_pcorr(GRAD_metr.copy(), vols_eff_lev[n].shape, bnd_cond_neumann_lst, nhalo=ncrop)


        MAT_ghost = DIV_contra * GRAD_metr
        if n <= n_DCA:
             MAT_ghost_lev.append(MAT_ghost)


        DIV_lev.append(DIV)
        GRAD_lev.append(GRAD_metr)
        DIV_contra_lev.append(DIV_contra)
    
    #Proceed with Galerkin coarse-grid approximation
    for n in range(n_DCA, min(nlev, ngrids - 1)):

        return_lst = ddcp.subdomain_to_subtomain_galerkin(
                                                             mpicomm, 
                                                             y2_list[n], x2_list[n], y2_list[n + 1], x2_list[n + 1],
                                                             nz_lev[n] + 2 * ncrop, nri_lev[n], ncj_lev[n], 
                                                             nz_lev[n + 1] + 2 * ncrop, nri_lev[n + 1], ncj_lev[n + 1],
                                                             pids_lev[n], pids_lev[n + 1]
                                                         )

        bounds_send, bounds_recv, fld_part_same  = return_lst[:]
            
        MAT_tmp = ddcp.gather_galerkin_coarse_op(mpicomm, MAT_ghost_lev[-1], bounds_send, fld_part_same, bounds_recv)


        if n < nlev - 1:

            ind_plev = pids_lev[n + 1].index(rank)
            ind_prlev = int(ind_plev / npc_lev[n + 1])
            ind_pclev = ind_plev - ind_prlev * npc_lev[n + 1]

            jc_cst = ncj_lev[n + 1][ind_pclev]
            jc_cend = ncj_lev[n + 1][ind_pclev + 1] + 2 * ncrop
            ir_cst = nri_lev[n + 1][ind_prlev]
            ir_cend = nri_lev[n + 1][ind_prlev + 1] + 2 * ncrop

            jc_fst = max(np.argwhere(x2_list[n + 1][jc_cst] == x2_list[n])[0][0] - 1, 0)
            jc_fend = min(np.argwhere(x2_list[n + 1][jc_cend] == x2_list[n])[0][0] + 1, x2_list[n].size - 1)
            ir_fst = max(np.argwhere(y2_list[n + 1][ir_cst] == y2_list[n])[0][0] - 1, 0)
            ir_fend = min(np.argwhere(y2_list[n + 1][ir_cend] == y2_list[n])[0][0] + 1, y2_list[n].size - 1)                

            edges_fine = [x2_list[n], y2_list[n], z2_list[n]]
            edges_coarse = [x2_list[n + 1], y2_list[n + 1], z2_list[n + 1]]

            dom_bnds = [[[ir_cst, ir_cend], [jc_cst, jc_cend]], [[ir_fst, ir_fend], [jc_fst, jc_fend]]]
                
            SCAL = ops.REST_T_weight_scaling(edges_fine, edges_coarse, dom_bnds)
            REST_MAT = ops.make_restr(edges_fine, edges_coarse, dom_bnds, order='constant', type='Galerkin')
            oc_fac = 1.0 / 1.90
            PROL_MAT = REST_MAT.transpose() * SCAL
            MAT_ghost = oc_fac * REST_MAT * MAT_tmp * PROL_MAT
            MAT_ghost_lev.append(MAT_ghost)
    
    for n in range(nlev):

        pids_c = pids_lev[n]
        npc = npc_lev[n]
        npr = npr_lev[n]
        pid = pids_c.index(rank)
        ind_pr = pid / npc
        ind_pc = pid - ind_pr * npc

        nz = nz_lev[n]
        nr_sub = nri_lev[n][ind_pr + 1] - nri_lev[n][ind_pr]
        nc_sub = ncj_lev[n][ind_pc + 1] - ncj_lev[n][ind_pc]


        X.append(np.zeros([nc_sub * nr_sub * nz]))
        R.append(np.zeros([nc_sub * nr_sub * nz]))
        B.append(np.zeros([nc_sub * nr_sub * nz]))

        # remove the halo layer from the differential operators

        CROP_CELL, CROP_FACE = ops.crop_operators([nz + 2 * ncrop, nr_sub + 2 * ncrop, nc_sub + 2 * ncrop], ncrop=ncrop)
        MAT = CROP_CELL * MAT_ghost_lev[n] * CROP_CELL.transpose()

        if n == 0:            

            DIV_lev[0] = CROP_CELL * DIV_lev[0] * CROP_FACE.transpose()
            DIV_contra_lev[0] = CROP_CELL * DIV_contra_lev[0] * CROP_FACE.transpose()
            GRAD_lev[0] = CROP_FACE * GRAD_lev[0] * CROP_CELL.transpose()
           
        MAT_lev.append(MAT)

        RD, BL = ops.make_checkerboard_point_pattern(nri_lev[n], ncj_lev[n], nz, ind_pr, ind_pc, bnd_pres_x, bnd_pres_y, bnd_pres_z)
        
        RD_lev.append(RD)
        BL_lev.append(BL)            


        # make smoother

        if param_dict['smoother'] == 'ILU':
            smooth_type = 'SPAI'
            ILU = ops.ILU(MAT)
            SMO_lev.append(ILU)

        if param_dict['smoother'] == 'SPAI': # sparse approximate inverse
            smooth_type = 'SPAI'
            if param_dict['comp_smoother']:
                SMO = ops.make_SPAI_smoother(MAT, [nz, nr_lev[n], nc_lev[n]], param_dict)
            if param_dict['save_smoother']:
                scipy.sparse.save_npz('./SMOOTHER/SMO_{}_{}.npz'.format(n, rank), SMO)
            if param_dict['load_smoother']:
                SMO = scipy.sparse.load_npz('./SMOOTHER/SMO_{}_{}.npz'.format(n, rank))
            SMO_lev.append(param_dict['omega_or'] * SMO)

        elif param_dict['smoother'] == 'RJAC': # relaxed jacobi iterations
            smooth_type = 'Relaxed_Jacobi'
            diag = MAT.diagonal()
            diag_inv = 1.0 / (diag + 1e-100) * np.array(np.array(diag, dtype=bool), dtype=float)
            Dinv_lev.append(diag_inv)
            
            global rjac_params 
            rjac_params = param_dict['RJAC_params']


        elif param_dict['smoother'] == 'LOR': # lexicographic overrelaxation
            smooth_type = 'Lexicographic'
            diag = MAT.diagonal()
            diag_inv = param_dict['omega_or'] / (diag + 1e-100) * np.array(np.array(diag, dtype=bool), dtype=float)
            Dinv_lev.append(diag_inv)

        elif param_dict['smoother'] == 'LSOR': # lexicographic-symmetric overrelaxation
            smooth_type = 'Symmetric'
            diag = MAT.diagonal()
            diag_inv = param_dict['omega_or'] / (diag + 1e-100) * np.array(np.array(diag, dtype=bool), dtype=float)
            Dinv_lev.append(diag_inv)

        elif param_dict['smoother'] == 'RBOR': # red-black overrelaxation
            smooth_type = 'Red_Black'
            diag = MAT.diagonal()

            if param_dict['omega_or_field']:
                omega_or = calc_omega_or_ub(MAT_ghost_lev[n], (nz, nr_sub, nc_sub), ncrop)
            else:
                omega_or = param_dict['omega_or']

            diag_inv = omega_or  / (diag + 1e-100) * np.array(np.array(diag, dtype=bool), dtype=float)
            Dinv = diags(diag_inv, format='csr')
            Dinv_lev.append([(RD * Dinv).diagonal(), (BL * Dinv).diagonal()])
            RDBL_MAT_lev.append([RD * MAT, BL * MAT])


# Intra-domain and cyclic boundary conditions 

    bounds_s_lev = []
    bounds_r_lev = []
    bounds_r_tmp_lev = []
    bounds_rd_s_lev = []
    bounds_rd_r_lev = []
    bounds_rd_r_tmp_lev = []
    bounds_bl_s_lev = []
    bounds_bl_r_lev = []
    bounds_bl_r_tmp_lev = []

    for n in range(nlev):
        pids_c = pids_lev[n]
        npc = npc_lev[n]
        npr = npr_lev[n]
        pid = pids_c.index(rank)
        ind_pr = pid / npc
        ind_pc = pid - ind_pr * npc

        nz = nz_lev[n]
        ny = nri_lev[n][ind_pr + 1] - nri_lev[n][ind_pr]
        nx = ncj_lev[n][ind_pc + 1] - ncj_lev[n][ind_pc]

        w_fld = np.zeros([nz, ny, nx])
        w_fld[:, :, 0] = 1.0
        w_fld_rd = (RD_lev[n] * w_fld.flatten()).reshape(nz, ny, nx)
        w_fld_bl = (BL_lev[n] * w_fld.flatten()).reshape(nz, ny, nx)
        w_inds_rd = np.where(w_fld_rd.flatten())
        w_inds_bl = np.where(w_fld_bl.flatten())
        w_inds = np.where(w_fld.flatten())
        e_fld = np.zeros([nz, ny, nx])
        e_fld[:, :, -1] = 1.0
        e_fld_rd = (RD_lev[n] * e_fld.flatten()).reshape(nz, ny, nx)
        e_fld_bl = (BL_lev[n] * e_fld.flatten()).reshape(nz, ny, nx)
        e_inds_rd = np.where(e_fld_rd.flatten())
        e_inds_bl = np.where(e_fld_bl.flatten())
        e_inds = np.where(e_fld.flatten())
        s_fld = np.zeros([nz, ny, nx])
        s_fld[:, 0] = 1.0
        s_fld_rd = (RD_lev[n] * s_fld.flatten()).reshape(nz, ny, nx)
        s_fld_bl = (BL_lev[n] * s_fld.flatten()).reshape(nz, ny, nx)
        s_inds_rd = np.where(s_fld_rd.flatten())
        s_inds_bl = np.where(s_fld_bl.flatten())
        s_inds = np.where(s_fld.flatten())
        n_fld = np.zeros([nz, ny, nx])
        n_fld[:, -1] = 1.0
        n_fld_rd = (RD_lev[n] * n_fld.flatten()).reshape(nz, ny, nx)
        n_fld_bl = (BL_lev[n] * n_fld.flatten()).reshape(nz, ny, nx)
        n_inds_rd = np.where(n_fld_rd.flatten())
        n_inds_bl = np.where(n_fld_bl.flatten())
        n_inds = np.where(n_fld.flatten())
        b_fld = np.zeros([nz, ny, nx])        
        b_fld[0] = 1.0
        b_fld_rd = (RD_lev[n] * b_fld.flatten()).reshape(nz, ny, nx)
        b_fld_bl = (BL_lev[n] * b_fld.flatten()).reshape(nz, ny, nx)
        b_inds_rd = np.where(b_fld_rd.flatten())
        b_inds_bl = np.where(b_fld_bl.flatten())
        b_inds = np.where(b_fld.flatten())
        t_fld = np.zeros([nz, ny, nx])
        t_fld[-1] = 1.0
        t_fld_rd = (RD_lev[n] * t_fld.flatten()).reshape(nz, ny, nx)
        t_fld_bl = (BL_lev[n] * t_fld.flatten()).reshape(nz, ny, nx)
        t_inds_rd = np.where(t_fld_rd.flatten())
        t_inds_bl = np.where(t_fld_bl.flatten())
        t_inds = np.where(t_fld.flatten())
        sw_fld = np.zeros([nz, ny, nx])
        sw_fld[:, 0, 0] = 1.0
        sw_fld_rd = (RD_lev[n] * sw_fld.flatten()).reshape(nz, ny, nx)
        sw_fld_bl = (BL_lev[n] * sw_fld.flatten()).reshape(nz, ny, nx)
        sw_inds_rd = np.where(sw_fld_rd.flatten())
        sw_inds_bl = np.where(sw_fld_bl.flatten())
        sw_inds = np.where(sw_fld.flatten())
        se_fld = np.zeros([nz, ny, nx])
        se_fld[:, 0, -1] = 1.0
        se_fld_rd = (RD_lev[n] * se_fld.flatten()).reshape(nz, ny, nx)
        se_fld_bl = (BL_lev[n] * se_fld.flatten()).reshape(nz, ny, nx)
        se_inds_rd = np.where(se_fld_rd.flatten())
        se_inds_bl = np.where(se_fld_bl.flatten())
        se_inds = np.where(se_fld.flatten())
        nw_fld = np.zeros([nz, ny, nx])
        nw_fld[:, -1, 0] = 1.0
        nw_fld_rd = (RD_lev[n] * nw_fld.flatten()).reshape(nz, ny, nx)
        nw_fld_bl = (BL_lev[n] * nw_fld.flatten()).reshape(nz, ny, nx)
        nw_inds_rd = np.where(nw_fld_rd.flatten())
        nw_inds_bl = np.where(nw_fld_bl.flatten())
        nw_inds = np.where(nw_fld.flatten())
        ne_fld = np.zeros([nz, ny, nx])
        ne_fld[:, -1, -1] = 1.0
        ne_fld_rd = (RD_lev[n] * ne_fld.flatten()).reshape(nz, ny, nx)
        ne_fld_bl = (BL_lev[n] * ne_fld.flatten()).reshape(nz, ny, nx)
        ne_inds_rd = np.where(ne_fld_rd.flatten())
        ne_inds_bl = np.where(ne_fld_bl.flatten())
        ne_inds = np.where(ne_fld.flatten())


        bounds_s = []
        bounds_r = []
        bounds_r_tmp = []
        bounds_rd_s = []
        bounds_rd_r = []
        bounds_rd_r_tmp = []
        bounds_bl_s = []
        bounds_bl_r = []
        bounds_bl_r_tmp = []

        ind_p = pids_lev[n].index(rank)


        if ind_pc > 0 or ind_pc == 0 and bnd_pres_x == 'cyclic':

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, ncrop:ny + ncrop, 0] = 1.0
            fld_out[ncrop:nz + ncrop, ncrop:ny + ncrop, 1] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten()) 
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(w_fld_rd[:, :, 0].flatten())
            Exp_bl = ops.make_expansion_op(w_fld_bl[:, :, 0].flatten())

            if ind_pc > 0:
                pid_dest = pids_lev[n][ind_p - 1]
            else:
                pid_dest = pids_lev[n][ind_p + npc_lev[n] - 1]
            tag_com = pid_dest * rank
            bounds_rd_s.append(ddcp.bound(np.zeros([len(w_inds_bl[0])]), w_inds_bl, pid_dest, 'w', tag=tag_com + 300000))
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(w_inds_rd[0])]), w_inds, pid_dest, 'w', tag=tag_com + 600000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(w_inds[0])]), w_inds, pid_dest, 'w', tag=tag_com + 600000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(w_inds_rd[0])]), w_inds_rd, pid_dest, 'w', tag=tag_com + 10300000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(w_inds_bl[0])]), w_inds, pid_dest, 'w', tag=tag_com + 10600000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(w_inds[0])]), w_inds, pid_dest, 'w', tag=tag_com + 10600000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(w_inds[0])]), w_inds, pid_dest, 'w', tag=tag_com + 300000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(w_inds[0])]), w_inds, pid_dest, 'w', tag=tag_com + 600000))
            bounds_r.append(ddcp.bound(np.zeros([len(w_inds[0])]), w_inds, pid_dest, 'w', tag=tag_com + 600000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if ind_pc < npc_lev[n] - 1 or ind_pc == npc_lev[n] - 1 and bnd_pres_x == 'cyclic':

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, ncrop:ny + ncrop, -1] = 1.0
            fld_out[ncrop:nz + ncrop, ncrop:ny + ncrop, -2] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(e_fld_rd[:, :, -1].flatten())
            Exp_bl = ops.make_expansion_op(e_fld_bl[:, :, -1].flatten())
            if ind_pc < npc_lev[n] - 1:
                pid_dest = pids_lev[n][ind_p + 1]
            else:
                pid_dest = pids_lev[n][ind_p - npc_lev[n] + 1]
            tag_com = pid_dest * rank
            bounds_rd_s.append(ddcp.bound(np.zeros([len(e_inds_bl[0])]), e_inds_bl, pid_dest, 'e', tag=tag_com + 600000))            
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(e_inds_rd[0])]), e_inds, pid_dest, 'e', tag=tag_com + 300000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(e_inds[0])]), e_inds, pid_dest, 'e', tag=tag_com + 300000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(e_inds_rd[0])]), e_inds_rd, pid_dest, 'e', tag=tag_com + 10600000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(e_inds_bl[0])]), e_inds, pid_dest, 'e', tag=tag_com + 10300000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(e_inds[0])]), e_inds, pid_dest, 'e', tag=tag_com + 10300000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(e_inds[0])]), e_inds, pid_dest, 'e', tag=tag_com + 600000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(e_inds[0])]), e_inds, pid_dest, 'e', tag=tag_com + 300000))
            bounds_r.append(ddcp.bound(np.zeros([len(e_inds[0])]), e_inds, pid_dest, 'e', tag=tag_com + 300000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if ind_pr > 0  or ind_pr == 0 and bnd_pres_y == 'cyclic':

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, 0, ncrop:nx + ncrop] = 1.0
            fld_out[ncrop:nz + ncrop, 1, ncrop:nx + ncrop] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(s_fld_rd[:, 0].flatten())
            Exp_bl = ops.make_expansion_op(s_fld_bl[:, 0].flatten())

            if ind_pr > 0 :
                pid_dest = pids_lev[n][ind_p - npc_lev[n]]
            else:
                pid_dest = pids_lev[n][ind_p + (npr_lev[n] - 1) * npc_lev[n]]
            tag_com = pid_dest * rank
            bounds_rd_s.append(ddcp.bound(np.zeros([len(s_inds_bl[0])]), s_inds_bl, pid_dest, 's', tag=tag_com + 900000))
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(s_inds_rd[0])]), s_inds, pid_dest, 's', tag=tag_com + 1200000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(s_inds[0])]), s_inds, pid_dest, 's', tag=tag_com + 1200000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(s_inds_rd[0])]), s_inds_rd, pid_dest, 's', tag=tag_com + 10900000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(s_inds_bl[0])]), s_inds, pid_dest, 's', tag=tag_com + 11200000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(s_inds[0])]), s_inds, pid_dest, 's', tag=tag_com + 11200000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(s_inds[0])]), s_inds, pid_dest, 's', tag=tag_com + 900000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(s_inds[0])]), s_inds, pid_dest, 's', tag=tag_com + 1200000))
            bounds_r.append(ddcp.bound(np.zeros([len(s_inds[0])]), s_inds, pid_dest, 's', tag=tag_com + 1200000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if ind_pr < npr - 1 or ind_pr == npr - 1 and bnd_pres_y == 'cyclic':

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, -1, ncrop:nx + ncrop] = 1.0
            fld_out[ncrop:nz + ncrop, -2, ncrop:nx + ncrop] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(n_fld_rd[:, -1].flatten())
            Exp_bl = ops.make_expansion_op(n_fld_bl[:, -1].flatten())

            if ind_p < npr - 1:
                pid_dest = pids_lev[n][ind_p + npc_lev[n]]
            else:
                pid_dest = pids_lev[n][ind_p - (npr_lev[n] - 1) * npc_lev[n]]
            tag_com = pid_dest * rank
            bounds_rd_s.append(ddcp.bound(np.zeros([len(n_inds_bl[0])]), n_inds_bl, pid_dest, 'n', tag=tag_com + 1200000))
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(n_inds_rd[0])]), n_inds, pid_dest, 'n', tag=tag_com + 900000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(n_inds[0])]), n_inds, pid_dest, 'n', tag=tag_com + 900000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(n_inds_rd[0])]), n_inds_rd, pid_dest, 'n', tag=tag_com + 11200000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(n_inds_bl[0])]), n_inds, pid_dest, 'n', tag=tag_com + 10900000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(n_inds[0])]), n_inds, pid_dest, 'n', tag=tag_com + 10900000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(n_inds[0])]), n_inds, pid_dest, 'n', tag=tag_com + 1200000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(n_inds[0])]), n_inds, pid_dest, 'n', tag=tag_com + 900000))
            bounds_r.append(ddcp.bound(np.zeros([len(n_inds[0])]), n_inds, pid_dest, 'n', tag=tag_com + 900000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)


        if ind_pc > 0 and ind_pr > 0:
            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, 0, 0] = 1.0
            fld_out[ncrop:nz + ncrop, 1, 1] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(sw_fld_rd[:, 0, 0].flatten())
            Exp_bl = ops.make_expansion_op(sw_fld_bl[:, 0, 0].flatten())

            pid_dest = pids_lev[n][ind_p - 1 - npc_lev[n]] 
            
            tag_com = pid_dest * rank
            if len(sw_inds_rd[0]): 
                bounds_rd_s.append(ddcp.bound(np.zeros([len(sw_inds_bl[0])]), sw_inds_bl, pid_dest, 'sw', tag=tag_com + 11500000))            
                bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(sw_inds_bl[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 11800000))
                bounds_rd_r.append(ddcp.bound(np.zeros([len(sw_inds[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 11800000))
                bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            if len(sw_inds_bl[0]):
                bounds_bl_s.append(ddcp.bound(np.zeros([len(sw_inds_rd[0])]), sw_inds_rd, pid_dest, 'sw', tag=tag_com + 1500000))
                bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(sw_inds_rd[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 1800000))
                bounds_bl_r.append(ddcp.bound(np.zeros([len(sw_inds[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 1800000))
                bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
        
            bounds_s.append(ddcp.bound(np.zeros([len(sw_inds[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 1500000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(sw_inds[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 1800000))
            bounds_r.append(ddcp.bound(np.zeros([len(sw_inds[0])]), sw_inds, pid_dest, 'sw', tag=tag_com + 1800000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if ind_pc < npc - 1 and ind_pr < npr - 1:
            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, -1, -1] = 1.0
            fld_out[ncrop:nz + ncrop, -2, -2] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(ne_fld_rd[:, -1, -1].flatten())
            Exp_bl = ops.make_expansion_op(ne_fld_bl[:, -1, -1].flatten())

            pid_dest = pids_lev[n][ind_p + 1 + npc_lev[n]]

            tag_com = pid_dest * rank

            if len(ne_inds_rd[0]):
                bounds_rd_s.append(ddcp.bound(np.zeros([len(ne_inds_bl[0])]), ne_inds_bl, pid_dest, 'ne', tag=tag_com + 11800000))
                bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(ne_inds_bl[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 11500000))
                bounds_rd_r.append(ddcp.bound(np.zeros([len(ne_inds[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 11500000))
                bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            if len(ne_inds_bl[0]):
                bounds_bl_s.append(ddcp.bound(np.zeros([len(ne_inds_rd[0])]), ne_inds_rd, pid_dest, 'ne', tag=tag_com + 1800000))
                bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(ne_inds_rd[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 1500000))
                bounds_bl_r.append(ddcp.bound(np.zeros([len(ne_inds[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 1500000))
                bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)

            bounds_s.append(ddcp.bound(np.zeros([len(ne_inds[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 1800000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(ne_inds[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 1500000))
            bounds_r.append(ddcp.bound(np.zeros([len(ne_inds[0])]), ne_inds, pid_dest, 'ne', tag=tag_com + 1500000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)
   

        if ind_pc > 0 and ind_pr < npr - 1:
            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, -1, 0] = 1.0
            fld_out[ncrop:nz + ncrop, -2, 1] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(nw_fld_rd[:, -1, 0].flatten())
            Exp_bl = ops.make_expansion_op(nw_fld_bl[:, -1, 0].flatten())

            pid_dest = pids_lev[n][ind_p - 1 + npc_lev[n]]

            tag_com = pid_dest * rank

            if len(nw_inds_rd[0]):
                bounds_rd_s.append(ddcp.bound(np.zeros([len(nw_inds_bl[0])]), nw_inds_bl, pid_dest, 'nw', tag=tag_com + 12100000))
                bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(nw_inds_bl[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 12400000))
                bounds_rd_r.append(ddcp.bound(np.zeros([len(nw_inds[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 12400000))
                bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            if len(nw_inds_bl[0]):
                bounds_bl_s.append(ddcp.bound(np.zeros([len(nw_inds_rd[0])]), nw_inds_rd, pid_dest, 'nw', tag=tag_com + 2100000))
                bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(nw_inds_rd[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 2400000))
                bounds_bl_r.append(ddcp.bound(np.zeros([len(nw_inds[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 2400000))
                bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)

            bounds_s.append(ddcp.bound(np.zeros([len(nw_inds[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 2100000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(nw_inds[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 2400000))
            bounds_r.append(ddcp.bound(np.zeros([len(nw_inds[0])]), nw_inds, pid_dest, 'nw', tag=tag_com + 2400000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if ind_pc < npc - 1 and ind_pr > 0:
            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[ncrop:nz + ncrop, 0, -1] = 1.0
            fld_out[ncrop:nz + ncrop, 1, -2] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(se_fld_rd[:, 0, -1].flatten())
            Exp_bl = ops.make_expansion_op(se_fld_bl[:, 0, -1].flatten())

            pid_dest = pids_lev[n][ind_p + 1 - npc_lev[n]]

            tag_com = pid_dest * rank
            if len(se_inds_rd[0]):
                bounds_rd_s.append(ddcp.bound(np.zeros([len(se_inds_bl[0])]), se_inds_bl, pid_dest, 'se', tag=tag_com + 12400000))
                bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(se_inds_bl[0])]), se_inds, pid_dest, 'se', tag=tag_com + 12100000))
                bounds_rd_r.append(ddcp.bound(np.zeros([len(se_inds[0])]), se_inds, pid_dest, 'se', tag=tag_com + 12100000))
                bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            if len(se_inds_bl[0]):
                bounds_bl_s.append(ddcp.bound(np.zeros([len(se_inds_rd[0])]), se_inds_rd, pid_dest, 'se', tag=tag_com + 2400000))
                bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(se_inds_rd[0])]), se_inds, pid_dest, 'se', tag=tag_com + 2100000))
                bounds_bl_r.append(ddcp.bound(np.zeros([len(se_inds[0])]), se_inds, pid_dest, 'se', tag=tag_com + 2100000))
                bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)

            bounds_s.append(ddcp.bound(np.zeros([len(se_inds[0])]), se_inds, pid_dest, 'se', tag=tag_com + 2400000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(se_inds[0])]), se_inds, pid_dest, 'se', tag=tag_com + 2100000))
            bounds_r.append(ddcp.bound(np.zeros([len(se_inds[0])]), se_inds, pid_dest, 'se', tag=tag_com + 2100000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        if bnd_pres_z == 'cyclic':

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[0, ncrop:ny + ncrop, ncrop:nx + ncrop] = 1.0
            fld_out[1, ncrop:ny + ncrop, ncrop:nx + ncrop] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(b_fld_rd[0].flatten())
            Exp_bl = ops.make_expansion_op(b_fld_bl[0].flatten())

            tag_com = rank ** 2
            pid_dest = rank
            bounds_rd_s.append(ddcp.bound(np.zeros([len(b_inds_bl[0])]), b_inds_bl, pid_dest, 'b', tag=tag_com + 2700000))
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(b_inds_rd[0])]), b_inds, pid_dest, 'b', tag=tag_com + 3000000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(b_inds[0])]), b_inds, pid_dest, 'b', tag=tag_com + 3000000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(b_inds_rd[0])]), b_inds_rd, pid_dest, 'b', tag=tag_com + 12700000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(b_inds_bl[0])]), b_inds, pid_dest, 'b', tag=tag_com + 13000000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(b_inds[0])]), b_inds, pid_dest, 'b', tag=tag_com + 13000000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(b_inds[0])]), b_inds, pid_dest, 'b', tag=tag_com + 2700000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(b_inds[0])]), b_inds, pid_dest, 'b', tag=tag_com + 3000000))
            bounds_r.append(ddcp.bound(np.zeros([len(b_inds[0])]), b_inds, pid_dest, 'b', tag=tag_com + 3000000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

            fld_in = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_out = np.zeros([nz + 2 * ncrop, ny + 2 * ncrop, nx + 2 * ncrop])
            fld_in[-1, ncrop:ny + ncrop, ncrop:nx + ncrop] = 1.0
            fld_out[-2, ncrop:ny + ncrop, ncrop:nx + ncrop] = 1.0
            Exp_bnd_mat = ops.make_expansion_op(fld_in.flatten())
            Compr_bnd_mat = (ops.make_expansion_op(fld_out.flatten())).transpose()

            Exp_rd = ops.make_expansion_op(t_fld_rd[-1].flatten())
            Exp_bl = ops.make_expansion_op(t_fld_bl[-1].flatten())
            bounds_rd_s.append(ddcp.bound(np.zeros([len(t_inds_bl[0])]), t_inds_bl, pid_dest, 't', tag=tag_com + 3000000))            
            bounds_rd_r_tmp.append(ddcp.bound(np.zeros([len(t_inds_rd[0])]), t_inds, pid_dest, 't', tag=tag_com + 2700000))
            bounds_rd_r.append(ddcp.bound(np.zeros([len(t_inds[0])]), t_inds, pid_dest, 't', tag=tag_com + 2700000))
            bounds_rd_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_rd)
            bounds_bl_s.append(ddcp.bound(np.zeros([len(t_inds_rd[0])]), t_inds_rd, pid_dest, 't', tag=tag_com + 13000000))
            bounds_bl_r_tmp.append(ddcp.bound(np.zeros([len(t_inds_bl[0])]), t_inds, pid_dest, 't', tag=tag_com + 12700000))
            bounds_bl_r.append(ddcp.bound(np.zeros([len(t_inds[0])]), t_inds, pid_dest, 't', tag=tag_com + 12700000))
            bounds_bl_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat * Exp_bl)
            bounds_s.append(ddcp.bound(np.zeros([len(t_inds[0])]), t_inds, pid_dest, 't', tag=tag_com + 3000000))
            bounds_r_tmp.append(ddcp.bound(np.zeros([len(t_inds[0])]), t_inds, pid_dest, 't', tag=tag_com + 2700000))
            bounds_r.append(ddcp.bound(np.zeros([len(t_inds[0])]), t_inds, pid_dest, 't', tag=tag_com + 2700000))
            bounds_r[-1].add_op(Compr_bnd_mat * MAT_ghost_lev[n] * Exp_bnd_mat)

        bounds_rd_s_lev.append(bounds_rd_s)
        bounds_rd_r_lev.append(bounds_rd_r)
        bounds_rd_r_tmp_lev.append(bounds_rd_r_tmp)
        bounds_bl_s_lev.append(bounds_bl_s)
        bounds_bl_r_lev.append(bounds_bl_r)
        bounds_bl_r_tmp_lev.append(bounds_bl_r_tmp)
        bounds_s_lev.append(bounds_s)
        bounds_r_lev.append(bounds_r)
        bounds_r_tmp_lev.append(bounds_r_tmp)

    set_zeros = []
    if rank == 0:
        set_zeros.append(np.full([1], nc_sub * nr_sub + nc_sub + 1))     





def bicgstab_solve(comm, pres, b, max_tol = 1e-06, niter_max = 10000, nsmooth_pre = 2, nsmooth_post = 2):
    """
    A stabilized biconjugated gradients solver 
    using the same domain decomposition
    technique as the parallel multigrid solver.
    Additional communications arise in the dot products.
    For preconditioning, it uses two V-cycles in each iteration.

    comm... ddcp.communicator
    pres... pressure field
    b... source field (rhs)
    max_tol... residual tolerance in max norm
    niter_max... maximum number of iterations
    nsmooth_pre, nsmooth_post... smoothing sweeps for pre-smoothing and post-smoothing in V-cycle preconditioner
    """

    global MAT_lev
    global bounds_ext
    global fld_inds_c
    global shp_c

    MAT = MAT_lev[0]

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()

    npr = comm.npr
    npc = comm.npc
    nz = comm.nz

    pids = comm.pids

    if rank == 0:
        rmax_vec = [np.empty([1]) for id in pids]
        buffs = [np.zeros([1]) for id in pids[1:]]

    else:
        buff = np.zeros([1])

    x = pres[fld_inds_c]
    b = b.flatten()

    st_time = time()
    r = b - matmul_dirichlet(mpicomm, MAT, x, 0)
    r_tilde = r.copy()

    rho = 1.0
    alpha = 1.0
    om = 1.0

    p = b.copy()
    v = b.copy()
    y = b.copy()
    h = b.copy()
    s = b.copy()
    z = b.copy()
    t = b.copy()
    u = b.copy()
    r_test = b.copy()

    y.fill(0.0)
    z.fill(0.0)
    u.fill(0.0)

    for n in range(niter_max):
        rho_tilde = ddcp.dot_para(comm, r, r_tilde)
        if n == 0:
            p[:] = r.copy()
        else:
            beta = rho_tilde / rho * (alpha / om)
            p[:] = r + beta * (p - om * v)

        y.fill(0.0)
        y[:] = V_cycle_parallel(comm, p, y, nsmooth_pre=nsmooth_pre, nsmooth_post=nsmooth_post)[0]
        v[:] = matmul_dirichlet(mpicomm, MAT, y, 0)
        alpha = rho_tilde / ddcp.dot_para(comm, r_tilde, v)
        s[:] = r - alpha * v
        z.fill(0.0)
        z[:] = V_cycle_parallel(comm, s, z, nsmooth_pre=nsmooth_pre, nsmooth_post=nsmooth_post)[0]
        t[:] = matmul_dirichlet(mpicomm, MAT, z, 0)
        om = ddcp.dot_para(comm, t, s) / ddcp.dot_para(comm, t, t)
        x[:] = x + om * z + alpha * y
        r[:] = s - om * t
        rmax = np.max(np.absolute(r))

        if rank == 0:
            rmax_vec[0] = rmax
            ddcp.gather_point(mpicomm, rmax_vec[1:], pids)
            rmax = np.max(np.array(rmax_vec))
            if rmax < max_tol or n == niter_max - 1:
#                print rmax
                buffs = [np.ones([1]) for id in pids[1:] ]
                ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)
#                print 'number of iterations {}'.format(n + 1)
                break
            else:
                ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)
        else:
            req = mpicomm.Isend(rmax, dest=0)
            req.wait()
            req = mpicomm.Irecv(buff, source=0)
            req.wait()
            exit_stat = buff[0]
            if exit_stat == 1:
                break
            req.wait()
            exit_stat = buff[0]
            if exit_stat == 1:
                break

        rho = rho_tilde

    p_new = pres.copy()
    p_new[fld_inds_c] = x

    return p_new


def mg_solve(comm, pres, b, max_tol = 1e-06, niter_max = 10000, nsmooth_pre = 2, nsmooth_post = 2):
    """
    A parallel multigrid solver
    using domain decomposition.
    Prior to using the solver it 
    must be initialized by calling
    the init_multigrid function.
    
    As matrix, the discretized Laplace operator 
    constructed by the init_multigrid function is taken.
    
    comm... general communicator object
    pres... pressure field
    b...  source field (rhs)
    max_tol... residual tolerance in max norm
    niter_max... maximum number of iterations
    nsmooth_pre, nsmooth_post... smoothing sweeps for pre-smoothing and post-smoothing in V-cycle
    """

    global MAT_lev
    global bounds_ext
    global fld_inds_c
    global shp_c

    mpicomm = comm.mpicomm

    rank = mpicomm.Get_rank()

    npr = comm.npr
    npc = comm.npc
    nz = comm.nz

    pids = comm.pids

    if rank == 0:
        rmax_vec = [np.empty([1]) for id in pids]
        buffs = [np.zeros([1]) for id in pids[1:]]
    else:
        buff = np.zeros([1])

    x = pres[fld_inds_c]
    b = b.flatten()

    rmax = 1.0

    for n in range(niter_max):
        x, b_dir, r = V_cycle_parallel(comm, b, x, nsmooth_pre, nsmooth_post)

        rmax_loc = np.max(np.absolute(r))
        
        if rank == 0:
            rmax_vec[0] = rmax_loc
            ddcp.gather_point(mpicomm, rmax_vec[1:], pids)
            rmax_new = np.max(np.array(rmax_vec))
#            print rmax_new, rmax_new / rmax
            rmax = rmax_new
            if rmax < max_tol or n == niter_max - 1:
#                print rmax
                buffs = [ np.ones([1]) for id in pids[1:] ]
                ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)
                print 'number of multigrid iterations {}'.format(n + 1)
                break
            else:
                ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)
        else:
            req = mpicomm.Isend(rmax_loc, dest=0)
            req.wait()
            req = mpicomm.Irecv(buff, source=0)
            req.wait()
            exit_stat = buff[0]
            if exit_stat == 1:
                break

    p_new = pres.copy()
    p_new[fld_inds_c] = x

    return p_new


def divergence(u, v, w, rho):
    """
    Derives the divergence of a velocity field
    u, v, w... component-wise velocity field
    rho... reference density

    div... mass-flux divergence
    """

    global DIV_contra_lev
    global fld_inds_c, fld_inds_u, fld_inds_v, fld_inds_w
    global shp_u, shp_v, shp_w, shp_c

    vel = ops.put_1d([u[fld_inds_u].reshape(shp_u), v[fld_inds_v].reshape(shp_v), w[fld_inds_w].reshape(shp_w)])
    div = DIV_contra_lev[0] * vel * rho[fld_inds_c]
    
    return div


def norm_p(comm, p):
    """
    Sets the potential to zero indside lower-left grid cell

    p... pressure
    """

    global ng

    pids = comm.pids
    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    if rank == pids[0]:
        p_val = p[ng + 1, ng + 1, ng + 1]
        buffs = [ np.full([1], p_val) for id in pids[1:] ]
        ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)
        p[:] = p - p_val
    else:
        buff = np.zeros([1])
        req = mpicomm.Irecv(buff, source=pids[0])
        req.wait()
        p_val = buff[0]
        p[:] = p - p_val


def correct_vel_incomp(u, v, w, rho_0, p):
    """ 
    Applies the pressure correction
    to a velocity field to satisfy mass conservation.
    
    u, v, w... component-wise velocity field
    rho_0... horizontally averaged density
    p... pressure
    """

    global GRAD_dirich

    global fld_inds_u, fld_inds_v, fld_inds_w
    global shp_u, shp_v, shp_w
    global sl_c
    global fld_inds_n1_c, fld_inds_n1_u, fld_inds_n1_v, fld_inds_n1_w
    global shp_n1_c, shp_n1_u, shp_n1_v, shp_n1_w

    corr_vel = GRAD_dirich * p[fld_inds_n1_c]
    p_x, p_y, p_z = ops.put_3d(corr_vel, shp_n1_c)

    k_st, k_end = sl_c[0]
    ir_st, ir_end = sl_c[1]
    jc_st, jc_end = sl_c[2]

    u[fld_inds_u] = u[fld_inds_u] - (
                                        p_x[1:-1, 1:-1, 1:-1] /
                                        (0.5 * (rho_0[k_st:k_end, ir_st:ir_end, jc_st:jc_end + 1] + rho_0[k_st:k_end, ir_st:ir_end, jc_st - 1:jc_end]))
                                    ).flatten()

    v[fld_inds_v] = v[fld_inds_v] - (
                                        p_y[1:-1, 1:-1, 1:-1] /
                                        (0.5 * (rho_0[k_st:k_end, ir_st:ir_end + 1, jc_st:jc_end] + rho_0[k_st:k_end, ir_st - 1:ir_end, jc_st:jc_end]))
                                    ).flatten()
    w[fld_inds_w] = w[fld_inds_w] - (
                                        p_z[1:-1, 1:-1, 1:-1] /
                                        (0.5 * (rho_0[k_st:k_end + 1, ir_st:ir_end, jc_st:jc_end] + rho_0[k_st - 1:k_end, ir_st:ir_end, jc_st:jc_end]))
                                    ).flatten()



def grad_p(p):
    """ 
    Similar, than correct_vel_incomp,
    but returns the perturbation pressure gradient.
    
    p... pressure
    gpx, gpy, gpz... returned gradient components 
    """

    global GRAD_dirich
    global ng
    global shp_n1_c, shp_n1_u, shp_n1_v, shp_n1_w
    global fld_inds_u, fld_inds_v, fld_inds_w
    global fld_inds_n1_c, fld_inds_n1_u, fld_inds_n1_v, fld_inds_n1_w

    nz, nr, nc = p[ng:-ng, ng:-ng, ng:-ng].shape

    corr_vel = GRAD_dirich * p[fld_inds_n1_c]

    p_x, p_y, p_z = ops.put_3d(corr_vel, shp_n1_c)

    gpx = np.zeros([nz + 2 * ng, nr + 2 * ng, nc + 2 * ng + 1])
    gpy = np.zeros([nz + 2 * ng, nr + 2 * ng + 1, nc + 2 * ng])
    gpz = np.zeros([nz + 2 * ng + 1, nr + 2 * ng, nc + 2 * ng])

    gpx[fld_inds_u] = p_x[1:-1, 1:-1, 1:-1].flatten()
    gpy[fld_inds_v] = p_y[1:-1, 1:-1, 1:-1].flatten()
    gpz[fld_inds_w] = p_z[1:-1, 1:-1, 1:-1].flatten()

    return (gpx, gpy, gpz)



def output(x, fvol, ff1d, shape, lev, rank, string):
    """
    Output function used only for test and debugging.
    """

    nz, ny, nx = shape[:]

    data_out = Dataset(string + '_' + str(lev) + '_' + str(rank).zfill(2) + '.nc', 'w', type='NETCDF4')

    xdim = data_out.createDimension('x', nx)
    ydim = data_out.createDimension('y', ny)
    zdim = data_out.createDimension('z', nz)
    x2dim = data_out.createDimension('x2', nx + 1)
    y2dim = data_out.createDimension('y2', ny + 1)
    z2dim = data_out.createDimension('z2', nz + 1)

    xvar = data_out.createVariable('X', np.float, ('z', 'y', 'x'))
    fvolvar = data_out.createVariable('fvol', np.float, ('z', 'y', 'x'))
    ffxvar = data_out.createVariable('ffx', np.float, ('z', 'y', 'x2'))
    ffyvar = data_out.createVariable('ffy', np.float, ('z', 'y2', 'x'))
    ffzvar = data_out.createVariable('ffz', np.float, ('z2', 'y', 'x'))

    x = x.reshape(nz, ny, nx)
    areasx, areasy, areasz = ops.put_3d(ff1d, [nz, ny, nx])
    fvol = fvol.reshape(nz, ny, nx)

    xvar[:] = x
    fvolvar[:] = fvol
    ffxvar[:] = areasx
    ffyvar[:] = areasy
    ffzvar[:] = areasz

    data_out.close()


def V_cycle_parallel(comm, b, x, nsmooth_pre, nsmooth_post):
    """
    A multigrid V-cycle in parallel.
    It uses domain decomposition on each level 
    for parallel computation. The number of processes
    involved in the computation decreases at the very 
    coarse grids to balance communication overload of
    parallel smoothing. 

    comm... ddcp.communicator
    b... source field (rhs)
    x... solution field
    nsmooth_pre, nsmooth_post... smoothing sweeps for pre-smoothing and post-smoothing in V-cycle 
    """

#    if len(pids_lev[-1]) == 1:
#        n_direct = 1
#    else:
#        n_direct = 3

    mpicomm = comm.mpicomm
    nz = comm.nz
    rank = mpicomm.Get_rank()
    depth = len(MAT_lev)

    smooth_fac = 1
    nsmooth_max = 5

    X = []
    X_tmp = []
    B = []

    for n, MAT in enumerate(MAT_lev):
        X.append(np.empty([MAT.shape[0]]))
        B.append(np.empty([MAT.shape[0]]))

    for shp in X_tmp_shp:
        X_tmp.append(np.empty([np.prod(shp)]))

    X[0][:] = x
    B[0][:] = b
    for n in xrange(1, depth):
        X[n].fill(0.0)

    for n in xrange(0, depth - 1, 1):                
        X[n][:], b_dir, r = smooth(mpicomm, B[n], X[n], n, min(nsmooth_pre + n * smooth_fac, nsmooth_max))
        X_tmp[n] = REST_lev[n] * r
        inds1, inds2 = subdomain_same[n].inds1, subdomain_same[n].inds2
        B[n + 1].fill(0.0)
        B[n + 1][inds1] = X_tmp[n][inds2]
        ddcp.cptobounds(X_tmp[n], subdomain_to_lev[n])
        ddcp.exchange_fields(mpicomm, subdomain_to_lev[n], subdomain_from_lev[n])
        ddcp.cpfrombounds(B[n + 1], subdomain_from_lev[n], mode='add')
    
    if depth == ngrids:
#        X[depth - 1][:], b_dir, r = direct(mpicomm, B[depth - 1], X[depth - 1], depth - 1, n_iter=3)
        X[depth - 1][:], b_dir, r = smooth(mpicomm, B[depth - 1], X[depth - 1], depth - 1, 10)

    else:
        X[depth - 1][:], b_dir, r = smooth(mpicomm, B[depth - 1], X[depth - 1], depth - 1, min(nsmooth_pre + (depth - 1) * smooth_fac, nsmooth_max))
        X_tmp[depth - 1] = REST_lev[depth - 1] * r
        ddcp.cptobounds(X_tmp[depth - 1], subdomain_to_lev[depth - 1])
        ddcp.exchange_fields(mpicomm, subdomain_to_lev[depth - 1], subdomain_from_lev[depth - 1])
        ddcp.exchange_fields(mpicomm, subdomain_from_lev[depth - 1], subdomain_to_lev[depth - 1])
        ddcp.cpfrombounds(X_tmp[depth - 1], subdomain_to_lev[depth - 1], mode='repl')
        X[depth - 1][:] += PROL_lev[depth - 1] * X_tmp[depth - 1]
        X[depth - 1][:], b_dir, r = smooth(mpicomm, B[depth - 1], X[depth - 1], depth - 1, min(nsmooth_post + (depth - 1) * smooth_fac, nsmooth_max))

    for n in xrange(depth - 2, -1, -1):
        inds1, inds2 = subdomain_same[n].inds1, subdomain_same[n].inds2
        X_tmp[n][inds2] = X[n + 1][inds1]
        ddcp.cptobounds(X[n + 1], subdomain_from_lev[n])
        ddcp.exchange_fields(mpicomm, subdomain_from_lev[n], subdomain_to_lev[n])
        ddcp.cpfrombounds(X_tmp[n], subdomain_to_lev[n], mode='repl')
        X[n][:] += PROL_lev[n] * X_tmp[n]
        X[n][:], b_dir, r = smooth(mpicomm, B[n], X[n], n, min(nsmooth_post + n * smooth_fac, nsmooth_max))

    return X[0], b_dir, r


def smooth(mpicomm, b, x, lev, nsmooth):
    """ 
    This is the core of the multigrid solver. Smoothing is applied
    to the local system of equations defined on the subdomain.
    Boundary-value exchanges across a Dirichtlet-Dirichtlet boundary
    approximate the solution on the full-size domain.
    If Red-Black overrelaxation is selected as the smoother, the algorithm is fully parallelized
    by the RB ordering and does not differ from the sequential version. 
    If SPAI is selected as the smoother, smoothing has more
    "Jacobi" character at the subdomain boundaries and the parallel version 
    is thus a little less efficient than the sequential algorithm.

    mpicomm... mpi communicator
    b... right-hand side vector
    x... solution vector
    lev... multigrid level
    nsmooth... number of smoothing sweeps.
    r... residual vector
    """

    global smooth_type

    rank = mpicomm.Get_rank()
    MAT = MAT_lev[lev]

    x[:], r = globals()[smooth_type + '_sweeps'](mpicomm, MAT, x, b, lev, nsmooth)

    return x, b, r


def direct(mpicomm, b, x, lev, n_iter):
    """ 
    Instead of point smoothing, this approximation
    uses a direct solver on each subdomain.
    It can be regarded as a Jacobi block smoother.

    mpicomm... mpi communicator
    b... right-hand side vector
    x... solution vector
    lev... multigrid level
    n_iter... number of iterations (should be only one in case of a single subdomain) 
    r... returned residual vector (zeros as for direct solver)
    b_dir... returned right-hand side with added (Dirichlet) boundary values
    """

    rank = mpicomm.Get_rank()
    MAT = MAT_lev[lev]

    for n in range(n_iter):

        update_bounds(mpicomm, x, bounds_rd_s_lev[lev], bounds_rd_r_lev[lev], bounds_rd_r_tmp_lev[lev])
        update_bounds(mpicomm, x, bounds_bl_s_lev[lev], bounds_bl_r_lev[lev], bounds_bl_r_tmp_lev[lev])
        b_dir = b.copy()
        ddcp.cpfrombounds(b_dir, bounds_rd_r_lev[lev], mode='sub')
        ddcp.cpfrombounds(b_dir, bounds_bl_r_lev[lev], mode='sub')
        x[:] = scipy.sparse.linalg.spsolve(MAT + eye(MAT.shape[0]) * 1e-200, b_dir)
#        x[:] = scipy.sparse.linalg.lsqr(MAT + eye(MAT.shape[0]) * 1e-200, b_dir, iter_lim=50)[0]

    return x, b_dir, np.zeros_like(x)


def update_bounds(mpicomm, x, bounds_s, bounds_r, bounds_r_tmp):
    """ 
    Used to update boundary values for parallel relaxation

    mpicomm... mpi communicator
    x... solution vector
    bounds_s... list of sending objects
    bounds_r_tmp... list of temporary receiving objects
    bounds_r... list of boundary objects to store the received values
    """ 

    ddcp.cptobounds(x, bounds_s)
    ddcp.exchange_fields(mpicomm, bounds_s, bounds_r_tmp)
    for n, bound in enumerate(bounds_r_tmp):
        bounds_r[n].data[:] = bounds_r[n].op * bound.data


def matmul_dirichlet(mpicomm, MAT, x, lev):
    """
    Matrix-vector multiplication with Dirichtlet boundary 
    conditions.

    mpicomm... mpi communicator
    MAT... matrix
    x... vector
    lev... multigrid level
    
    matx... returned matrix-vector product
    """

    update_bounds(mpicomm, x, bounds_s_lev[lev], bounds_r_lev[lev], bounds_r_tmp_lev[lev])
    matx = MAT * x
    ddcp.cpfrombounds(matx, bounds_r_lev[lev], mode='add')

    return matx


def SPAI_sweeps(mpicomm, MAT, x, b, lev, n):
    """
    Smoothing sweeps using a smoother in sparse matrix form
    which approximates the inverse of A. 

    mpicomm... mpi communicator
    MAT... differential matrix
    x... solution vector
    b... right-hand side vector
    lev... multigrid level
    n... number of smoothing sweeps
   
    r... returned residual
    """

    global SMO_lev

    SMO = SMO_lev[lev]

    for i in range(n):
        matx = matmul_dirichlet(mpicomm, MAT, x, lev)
        r = b - matx
        x += SMO * r
    matx = matmul_dirichlet(mpicomm, MAT, x, lev)
    r = b - matx

    return x, r


def Red_Black_sweeps(mpicomm, MAT, x, b, lev, n):
    """
    Gauss Seidel smoothing sweeps in red-black ordering.
    It essentially is composed of two Jacobi sweeps
    treating all red nodes, followed by all black nodes.

    mpicomm... mpi communicator
    MAT... differential matrix
    x... solution vector
    b... right-hand side vector
    lev... multigrid level
    n... number of smoothing sweeps
   
    r... returned residual
    """

    global Dinv_lev
    global RDBL_MAT_lev
    global RD_lev, BL_lev
    global RD_inds_lev, BL_inds_lev

    Dinv_RD, Dinv_BL = Dinv_lev[lev][:]
    RD_MAT, BL_MAT = RDBL_MAT_lev[lev][:]
    
    update_bounds(mpicomm, x, bounds_bl_s_lev[lev], bounds_bl_r_lev[lev], bounds_bl_r_tmp_lev[lev]) 

    for i in range(n):
        update_bounds(mpicomm, x, bounds_rd_s_lev[lev], bounds_rd_r_lev[lev], bounds_rd_r_tmp_lev[lev])
        matx = RD_MAT * x
        ddcp.cpfrombounds(matx, bounds_rd_r_lev[lev], mode='add')
        ddcp.cpfrombounds(matx, bounds_bl_r_lev[lev], mode='add')
        r = b - matx
        x += Dinv_RD * r
        update_bounds(mpicomm, x, bounds_bl_s_lev[lev], bounds_bl_r_lev[lev], bounds_bl_r_tmp_lev[lev])
        matx = BL_MAT * x
        ddcp.cpfrombounds(matx, bounds_bl_r_lev[lev], mode='add')
        ddcp.cpfrombounds(matx, bounds_rd_r_lev[lev], mode='add')
        r = b - matx
        x += Dinv_BL * r

    update_bounds(mpicomm, x, bounds_rd_s_lev[lev], bounds_rd_r_lev[lev], bounds_rd_r_tmp_lev[lev])
    matx = MAT * x
    ddcp.cpfrombounds(matx, bounds_rd_r_lev[lev], mode='add')
    ddcp.cpfrombounds(matx, bounds_bl_r_lev[lev], mode='add')

    r = b - matx

    return x, r


def Relaxed_Jacobi_sweeps(mpicomm, MAT, x, b, lev, n):
    """
    A sequence of 3 progressively under-relaxed Jacobi
    iterations, starting with a heavy over-relaxation can
    give a powerful smoother (Yang and Mittal, 2017).

    mpicomm... mpi communicator
    MAT... differential matrix
    x... solution vector
    b... right-hand side vector
    lev... multigrid level
    n... number of smoothing sweeps
   
    r... returned residual
    """

    global Dinv
    global rjac_params

    Dinv = Dinv_lev[lev]
     

    for i in range(n):
        matx = matmul_dirichlet(mpicomm, MAT, x, lev)
        r = b - matx
        x += rjac_params[0] * Dinv * r
        matx = matmul_dirichlet(mpicomm, MAT, x, lev)
        r = b - matx
        x += rjac_params[1] * Dinv * r
        matx = matmul_dirichlet(mpicomm, MAT, x, lev)
        r = b - matx
        x += rjac_params[2] * Dinv * r

    matx = matmul_dirichlet(mpicomm, MAT, x, lev)
    r = b - matx

    return x, r


def Lexicographic_sweeps(mpicomm, MAT, x, b, lev, n):
    """
    Sweeps of Gauss Seidel performed in standard
    lexicographic ordering.

    mpicomm... mpi communicator
    MAT... differential matrix
    x... solution vector
    b... right-hand side vector
    lev... multigrid level
    n... number of smoothing sweeps
   
    r... returned residual
    """
    
    global Dinv

    Dinv = Dinv_lev[lev]
    b_dir = b.copy()

    for i in range(n):
        update_bounds(mpicomm, x, bounds_s_lev[lev], bounds_r_lev[lev], bounds_r_tmp_lev[lev])
        ddcp.cpfrombounds(b_dir, bounds_r_lev[lev], mode='sub')
        csr_gauss_seidel(MAT.shape[0], MAT.indptr, MAT.indices, MAT.data, Dinv, b_dir, x)
        b_dir[:] = b

    matx = matmul_dirichlet(mpicomm, MAT, x, lev)
    r = b - matx

    return x, r


def Symmetric_sweeps(mpicomm, MAT, x, b, lev, n):
    """
    Sweeps of Gauss Seidel performed in reverse
    ordering.

    mpicomm... mpi communicator
    MAT... differential matrix
    x... solution vector
    b... right-hand side vector
    lev... multigrid level
    n... number of smoothing sweeps
   
    r... returned residual
    """

    global Dinv

    Dinv = Dinv_lev[lev]

    b_dir = b.copy()

    for i in range(n):
        update_bounds(mpicomm, x, bounds_s_lev[lev], bounds_r_lev[lev], bounds_r_tmp_lev[lev])
        ddcp.cpfrombounds(b_dir, bounds_r_lev[lev], mode='sub')
        csr_gauss_seidel(MAT.shape[0], MAT.indptr, MAT.indices, MAT.data, Dinv, b_dir, x)
        b_dir[:] = b
        update_bounds(mpicomm, x, bounds_s_lev[lev], bounds_r_lev[lev], bounds_r_tmp_lev[lev])
        ddcp.cpfrombounds(b_dir, bounds_r_lev[lev], mode='sub')
        csr_gauss_seidel_reverse(MAT.shape[0], MAT.indptr, MAT.indices, MAT.data, Dinv, b_dir, x)
        b_dir[:] = b

    matx = matmul_dirichlet(mpicomm, MAT, x, lev)
    r = b - matx

    return x, r


def define_coarsening(x2_fine, y2_fine, z2_fine, rank, dim_size_threshold = 3, ngrids_max = 100):
    """
    This function defines the number and structure of all
    coarse grids. It uses simultaneous conditional semi-coarsening in all
    3 dimensions. One dimension is  coarsened as often as the dimension size
    threshold is not reached and the idealized smoothing factor threshold
    is not surpassed for all grid planes.

    x2_fine... input fine grid plane coordinates of first dimension
    y2_fine... input fine grid plane coordinates of second dimension
    z2_fine... input fine grid plane coordinates of third dimension 
    dim_size_threshold... minimum dimension size of the coarsest grid
    ngrids_max... maximum number of allowed grids

    x2_list, y2_list, z2_list... the returned grid-plane coordinates of all grids    
    """

    nz = z2_fine.size - 1
    nr = y2_fine.size - 1
    nc = x2_fine.size - 1

    x2_list = [x2_fine]
    y2_list = [y2_fine]
    z2_list = [z2_fine]

    for n in range(ngrids_max - 1):

        coarse = False

        dx_fine = x2_fine[1:] - x2_fine[:-1]
        dy_fine = y2_fine[1:] - y2_fine[:-1]
        dz_fine = z2_fine[1:] - z2_fine[:-1]

        mu_x, mu_y, mu_z = compute_idealized_approx_smoothfac(dx_fine, dy_fine, dz_fine)


        space_min = np.min(dx_fine)

        if x2_fine.size - 1 <= dim_size_threshold:
            x2_coarse = x2_fine
        else:
            x2_coarse = remove_meshlines(x2_fine, mu_x[0, 0, :])
        if x2_coarse.size != x2_fine.size:
            coarse = True

        x2_fine = x2_coarse
        dx_fine = x2_fine[1:] - x2_fine[:-1]

        space_min = np.min(dy_fine)

        if y2_fine.size - 1 <= dim_size_threshold:
            y2_coarse = y2_fine
        else:
            y2_coarse = remove_meshlines(y2_fine, mu_y[0, :, 0])
        if y2_coarse.size != y2_fine.size:
            coarse = True

        y2_fine = y2_coarse
        dy_fine = y2_fine[1:] - y2_fine[:-1]

        space_min = np.min(dz_fine)

        if z2_fine.size - 1 <= dim_size_threshold:
            z2_coarse = z2_fine
        else:
            z2_coarse = remove_meshlines(z2_fine, mu_z[:, 0, 0])
        if z2_coarse.size != z2_fine.size:
            coarse = True

        z2_fine = z2_coarse
        dz_fine = z2_fine[1:] - z2_fine[:-1]

        if coarse == False:

            return [x2_list, y2_list, z2_list]

        x2_list.append(x2_coarse)
        y2_list.append(y2_coarse)
        z2_list.append(z2_coarse)

    return [x2_list, y2_list, z2_list]



def remove_meshlines(x2_fine, mu_x, mu_lim=1.00, coarse_odd=False):
    """
    This function applies conditional semi-coarsening adapted from 
    Larrson et al. (2005). 
    Coarsening is carried out by removing planes of 
    meshlines. A plane of meshlines is only removed,
    if the idealized local smoothing factor is below the
    limit mu_lim. 
    If it is encountered  an odd grid size, either the
    plane with the worst (highest) smoothing factor can be
    left, or an additional plane is removed with the best smoothing
    factor, in order to merge three cells into one.  

    x2_fine... input fine grid plane coordinates in one dimension
    mu_x... approximately idealized smoothing factors in the given dimension (for RB-SOR)
    mu_lim... threshold smoothing factor, above which planes are not removed
    coarse_odd... Decide whether to keep or remove an additional plane in the 
                  case of uneven grid spacings

    x2_coarse... returned grid-plane coordinates in the given dimension of the coarsened grid
    """

    l = 0

    x2_coarse = [x2_fine[0], x2_fine[1]]
    if len(x2_fine[2:-2]) % 2 == 1:
        for k, edge in enumerate(x2_fine[2:-2]):
            if k % 2 == 0 and mu_x[k] < mu_lim:
                pass
            else:
                x2_coarse.append(edge)
    else:
        if coarse_odd:
            arg_mu_best = np.argmin(mu_x)
            if arg_mu_best % 2 == 0:
                pass
            else:
                arg_mu_best -= 1
            for k, edge in enumerate(x2_fine[2:-2]):
                if all((k % 2 == 0, mu_x[k] < mu_lim)):
                    pass
                elif k >= arg_mu_best:
                    l = 3 + k
                    if l < x2_fine.size - 2: 
                        x2_coarse.append(x2_fine[l])
                    else:
                        pass
                    l += 1
                    break
                else:
                    x2_coarse.append(edge)

        else:
            arg_mu_worst = np.argmax(mu_x)            
            if arg_mu_worst % 2 == 0:
                pass
            else:
                arg_mu_worst -= 1
            for k, edge in enumerate(x2_fine[2:-2]):
                if all((k % 2 == 0, k != arg_mu_worst, mu_x[k] < mu_lim)):
                    pass
                elif  k == arg_mu_worst:
                    x2_coarse.append(edge)
                    break
                else:
                    x2_coarse.append(edge)
            l = 3 + arg_mu_worst

        if l > 0:
            for k, edge in enumerate(x2_fine[l:-2]):
                if k % 2 == 0 and mu_x[k + l - 1] < mu_lim:
                    pass
                else:
                    x2_coarse.append(edge)


    x2_coarse.append(x2_fine[-2])
    x2_coarse.append(x2_fine[-1])

    return np.array(x2_coarse)



def coarse_grid_fields(grid_fields_fine, edges_fine, edges_coarse):
    """
    This routine coarsens the grid fields like the 
    effective cell volumes and areas, as well as the metric 
    derivatives used in DCA. Fields defined on the cell egdes are coarsened 
    by injection, if the cell face normal is in direction of
    the coarsened dimension. Otherwise coarsening is achieved
    by agglomeration and weighted averaging for the metric derivative.
    
    grid_fields_fine... grid geometry fields of the fine grid
    edges_fine... grid-plane coordinates of the fine grid
    edges_coarse... grid-plane coordinates of the coarse grid

    grid_fields_coarse...  returned coarsened grid geometry fields 
    """

    ncrop = 0

    area_eff1d_fine, vols_eff1d_fine, dh1d_fine, vols_1d_fine= grid_fields_fine[:]

    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]

    nx_coarse = x2_coarse.size - 1
    ny_coarse = y2_coarse.size - 1
    nz_coarse = z2_coarse.size - 1

    nx_fine = x2_fine.size - 1
    ny_fine = y2_fine.size - 1
    nz_fine = z2_fine.size - 1    

    area_eff_x_fine, area_eff_y_fine, area_eff_z_fine = ops.put_3d(area_eff1d_fine, [nz_fine, ny_fine, nx_fine])
    dhdx_fine, dhdy_fine, zeros = ops.put_3d(dh1d_fine, [nz_fine, ny_fine, nx_fine])
    vols_eff_fine = vols_eff1d_fine.reshape(nz_fine, ny_fine, nx_fine)
    vols_fine = vols_1d_fine.reshape(nz_fine, ny_fine, nx_fine)

    # x-coarsening
    
    area_eff_x_coarse_x = np.zeros([nz_fine, ny_fine, nx_coarse + 1])
    area_eff_y_coarse_x = np.zeros([nz_fine, ny_fine + 1, nx_coarse])
    area_eff_z_coarse_x = np.zeros([nz_fine + 1, ny_fine, nx_coarse])
    vols_eff_coarse_x = np.zeros([nz_fine, ny_fine, nx_coarse])
    vols_coarse_x = np.zeros([nz_fine, ny_fine, nx_coarse])
    dhdx_coarse_x = np.zeros([nz_fine, ny_fine, nx_coarse + 1])
    dhdy_coarse_x = np.zeros([nz_fine, ny_fine + 1, nx_coarse])

    area_eff_x_coarse_x[:, :, :ncrop + 1] = area_eff_x_fine[:, :, :ncrop + 1]
    dhdx_coarse_x[:, :, :ncrop + 1] = dhdx_fine[:, :, :ncrop + 1]
    ind_prev = 0

    for n, edge in enumerate(x2_coarse[1:]):
        ind = np.argwhere(x2_fine == edge)[0][0]
        area_eff_x_coarse_x[:, :, n + 1 + ncrop] = area_eff_x_fine[:, :, ind + ncrop]
        dhdx_coarse_x[:, :, n + 1 + ncrop] = dhdx_fine[:, :, ind + ncrop]

        for m in range(ind_prev, ind):
            area_eff_y_coarse_x[:, :, n + ncrop] += area_eff_y_fine[:, :, m + ncrop]
            area_eff_z_coarse_x[:, :, n + ncrop] += area_eff_z_fine[:, :, m + ncrop]
            vols_eff_coarse_x[:, :, n + ncrop] += vols_eff_fine[:, :, m + ncrop]
            vols_coarse_x[:, :, n + ncrop] += vols_fine[:, :, m + ncrop]
            dhdy_coarse_x[:, :, n + ncrop] += dhdy_fine[:, :, m + ncrop] * (x2_fine[m + 1] - x2_fine[m])
        ind_prev = ind
    u = dhdy_coarse_x.shape[2] - ncrop 
    dhdy_coarse_x[:, :, ncrop:u] = dhdy_coarse_x[:, :, ncrop:u] / (x2_coarse[1:] - x2_coarse[:-1]).reshape(1, 1, nx_coarse - 2 * ncrop)

    # y-coarsening

    area_eff_x_coarse_xy = np.zeros([nz_fine, ny_coarse, nx_coarse + 1])
    area_eff_y_coarse_xy = np.zeros([nz_fine, ny_coarse + 1, nx_coarse])
    area_eff_z_coarse_xy = np.zeros([nz_fine + 1, ny_coarse, nx_coarse])
    vols_eff_coarse_xy = np.zeros([nz_fine, ny_coarse, nx_coarse])
    vols_coarse_xy = np.zeros([nz_fine, ny_coarse, nx_coarse])
    dhdx_coarse_xy = np.zeros([nz_fine, ny_coarse, nx_coarse + 1])
    dhdy_coarse_xy = np.zeros([nz_fine, ny_coarse + 1, nx_coarse])

    area_eff_y_coarse_xy[:, :ncrop + 1] = area_eff_y_coarse_x[:, :ncrop + 1]
    dhdy_coarse_xy[:, :ncrop + 1] = dhdy_coarse_x[:, :ncrop + 1]
    ind_prev = 0

    for n,edge in enumerate(y2_coarse[1:]):
        ind = np.argwhere(y2_fine == edge)[0][0]
        area_eff_y_coarse_xy[:, n + 1 + ncrop] = area_eff_y_coarse_x[:, ind + ncrop]
        dhdy_coarse_xy[:, n + 1 + ncrop] = dhdy_coarse_x[:, ind + ncrop]

        for m in range(ind_prev, ind):
            area_eff_x_coarse_xy[:, n + ncrop] += area_eff_x_coarse_x[:, m + ncrop]
            area_eff_z_coarse_xy[:, n + ncrop] += area_eff_z_coarse_x[:, m + ncrop]
            vols_eff_coarse_xy[:, n + ncrop] += vols_eff_coarse_x[:, m + ncrop]
            vols_coarse_xy[:, n + ncrop] += vols_coarse_x[:, m + ncrop]
            dhdx_coarse_xy[:, n + ncrop] += dhdx_coarse_x[:, m + ncrop] * (y2_fine[m + 1] - y2_fine[m])
        ind_prev = ind

    u = dhdx_coarse_xy.shape[1] - ncrop
    dhdx_coarse_xy[:, ncrop:u] = dhdx_coarse_xy[:, ncrop:u] / (y2_coarse[1:] - y2_coarse[:-1]).reshape(1, ny_coarse - 2 * ncrop, 1)

    # z-coarsening

    area_eff_x_coarse = np.zeros([nz_coarse, ny_coarse, nx_coarse + 1])
    area_eff_y_coarse = np.zeros([nz_coarse, ny_coarse + 1, nx_coarse])
    area_eff_z_coarse = np.zeros([nz_coarse + 1, ny_coarse, nx_coarse])
    vols_eff_coarse = np.zeros([nz_coarse, ny_coarse, nx_coarse])
    vols_coarse = np.zeros([nz_coarse, ny_coarse, nx_coarse])
    dhdx_coarse = np.zeros([nz_coarse, ny_coarse, nx_coarse + 1])
    dhdy_coarse = np.zeros([nz_coarse, ny_coarse + 1, nx_coarse])

    area_eff_z_coarse[:ncrop + 1] = area_eff_z_coarse_xy[:ncrop + 1]
    ind_prev = 0

    for n,edge in enumerate(z2_coarse[1:]):
        ind = np.argwhere(z2_fine == edge)[0][0]
        area_eff_z_coarse[n + 1 + ncrop] = area_eff_z_coarse_xy[ind + ncrop]

        for m in range(ind_prev, ind):
            area_eff_x_coarse[n + ncrop] += area_eff_x_coarse_xy[m + ncrop]
            area_eff_y_coarse[n + ncrop] += area_eff_y_coarse_xy[m + ncrop]
            vols_eff_coarse[n + ncrop] += vols_eff_coarse_xy[m + ncrop]
            vols_coarse[n + ncrop] += vols_coarse_xy[m + ncrop]
            dhdx_coarse[n + ncrop] += dhdx_coarse_xy[m + ncrop] * (z2_fine[m + 1] - z2_fine[m])
            dhdy_coarse[n + ncrop] += dhdy_coarse_xy[m + ncrop] * (z2_fine[m + 1] - z2_fine[m])
        ind_prev = ind

    u = dhdx_coarse.shape[0] - ncrop
    dhdx_coarse[ncrop:u] = dhdx_coarse[ncrop:u] / (z2_coarse[1:] - z2_coarse[:-1]).reshape(nz_coarse - 2 * ncrop, 1, 1)
    dhdy_coarse[ncrop:u] = dhdy_coarse[ncrop:u] / (z2_coarse[1:] - z2_coarse[:-1]).reshape(nz_coarse - 2 * ncrop, 1, 1)

    area_eff1d_coarse = ops.put_1d([area_eff_x_coarse, area_eff_y_coarse, area_eff_z_coarse])
    dh1d_coarse = ops.put_1d([dhdx_coarse, dhdy_coarse, np.zeros_like(area_eff_z_coarse)])    
    vols_eff1d_coarse = vols_eff_coarse.flatten()
    vols_1d_coarse = vols_coarse.flatten()

    grid_fields_coarse = [area_eff1d_coarse, vols_eff1d_coarse, dh1d_coarse, vols_1d_coarse]

    return grid_fields_coarse


def coarse_gradient_comp(
                            dinv_fine, vols_eff_shift_fine, vols_eff_shift_coarse, area_eff_fine, area_eff_coarse, 
                            edges_fine, edges_coarse, comp, side
                        ):
    """
    An alternative approch to derive the coarse-grid coefficients of the gradient operator
    is not to use arithmetic averaging of the volumes of two adjacent grid cells,
    but instead use only the left- and right-hand side volume to derive
    pairs of different coefficients. This approach produces more consistent
    DCA when using semi-coarsening.

    dinv_fine... one-sided coefficients of the fine-grid gradient component
    vols_eff_shift_fine... one-sided effective volume of the fine grid
    vols_eff_shift_coarse... one-sided effective volume of the coarse grid
    area_eff_fine... effective area at gradient face of the fine grid
    area_eff_coarse... effective area at gradient face of the coarse grid
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid
    comp... gradient component (can be either 'u', 'v' or 'w')
    side... side of the coefficient (can be either 'l' or 'r')

    dginv_c... returned one-sided coefficients of the coarse-grid gradient component
    """


    ncrop = 0

    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]

    nx_coarse = x2_coarse.size - 1
    ny_coarse = y2_coarse.size - 1
    nz_coarse = z2_coarse.size - 1

    nx_fine = x2_fine.size - 1
    ny_fine = y2_fine.size - 1
    nz_fine = z2_fine.size - 1

    
    vol_grad_fine = 1.0 / (dinv_fine + 1e-200) * area_eff_fine


    if comp == 'u':
        shape_fine = dinv_fine.shape
        
        shape_coarse_z = [nz_coarse, ny_fine, nx_fine + 1]
        vols_grad_coarse_z = np.zeros(shape_coarse_z)
        vols_eff_shift_coarse_z = np.zeros(shape_coarse_z)

        ind_prev = 0
        for n,edge in enumerate(z2_coarse[1:]):
            ind = np.argwhere(z2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_z[n + ncrop] += vols_eff_shift_fine[m + ncrop]
                vols_grad_coarse_z[n + ncrop] += vol_grad_fine[m + ncrop]
            ind_prev = ind

        vol_grad_fine = vols_grad_coarse_z
        vols_eff_shift_fine = vols_eff_shift_coarse_z 

        shape_coarse_y = [nz_coarse, ny_coarse, nx_fine + 1]
        vols_grad_coarse_y = np.zeros(shape_coarse_y)
        vols_eff_shift_coarse_y = np.zeros(shape_coarse_y)
 
        ind_prev = 0
        for n,edge in enumerate(y2_coarse[1:]):
            ind = np.argwhere(y2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_y[:, n + ncrop] += vols_eff_shift_fine[:, m + ncrop]
                vols_grad_coarse_y[:, n + ncrop] += vol_grad_fine[:, m + ncrop]
            ind_prev = ind 

        vols_eff_shift_fine = vols_eff_shift_coarse_y

        shape_coarse = [nz_coarse, ny_coarse, nx_coarse + 1]
        vols_grad_coarse = np.zeros(shape_coarse)        

        for n, edge in enumerate(x2_coarse[:]):
            ind = np.argwhere(x2_fine == edge)[0][0]
            vols_grad_coarse[:, :, n] = vols_grad_coarse_y[:, :, ind] + 0.5 * (vols_eff_shift_coarse[:, :, n] - vols_eff_shift_fine[:, :, ind])

            if all((n > 0, n < x2_coarse.size - 1, side == 'l')):
                if x2_coarse[n - 1] != x2_fine[ind - 1]:
                    vols_grad_coarse[:, :, n] += 0.5 * (vols_eff_shift_coarse[:, :, n + 1] - vols_eff_shift_fine[:, :, ind + 1])
            if all((n > 0, n < x2_coarse.size - 1, side == 'r')):
                if x2_coarse[n + 1] != x2_fine[ind + 1]:
                    vols_grad_coarse[:, :, n] += 0.5 * (vols_eff_shift_coarse[:, :, n - 1] - vols_eff_shift_fine[:, :, ind - 1])



    elif comp == 'v':
        shape_fine = dinv_fine.shape
        shape_coarse_z = [nz_coarse, ny_fine + 1, nx_fine]
        vols_grad_coarse_z = np.zeros(shape_coarse_z)
        vols_eff_shift_coarse_z = np.zeros(shape_coarse_z)

        ind_prev = 0
        for n,edge in enumerate(z2_coarse[1:]):
            ind = np.argwhere(z2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_z[n + ncrop] += vols_eff_shift_fine[m + ncrop]
                vols_grad_coarse_z[n + ncrop] += vol_grad_fine[m + ncrop]
            ind_prev = ind

        vol_grad_fine = vols_grad_coarse_z
        vols_eff_shift_fine = vols_eff_shift_coarse_z

        shape_coarse_x = [nz_coarse, ny_fine + 1, nx_coarse]
        vols_grad_coarse_x = np.zeros(shape_coarse_x)
        vols_eff_shift_coarse_x = np.zeros(shape_coarse_x)

        ind_prev = 0
        for n,edge in enumerate(x2_coarse[1:]):
            ind = np.argwhere(x2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_x[:, :, n + ncrop] += vols_eff_shift_fine[:, :, m + ncrop]
                vols_grad_coarse_x[:, :, n + ncrop] += vol_grad_fine[:, :, m + ncrop]
            ind_prev = ind

        vols_eff_shift_fine = vols_eff_shift_coarse_x

        shape_coarse = [nz_coarse, ny_coarse + 1, nx_coarse]
        vols_grad_coarse = np.zeros(shape_coarse)

        for n, edge in enumerate(y2_coarse[:]):
            ind = np.argwhere(y2_fine == edge)[0][0]
            vols_grad_coarse[:, n] = vols_grad_coarse_x[:, ind] + 0.5 * (vols_eff_shift_coarse[:, n] - vols_eff_shift_fine[:, ind])
            if all((n > 0, n < y2_coarse.size - 1, side == 'l')):
                if y2_coarse[n - 1] != y2_fine[ind - 1]:
                    vols_grad_coarse[:, n] += 0.5 * (vols_eff_shift_coarse[:, n + 1] - vols_eff_shift_fine[:, ind + 1])
            if all((n > 0, n < y2_coarse.size - 1, side == 'r')):
                if y2_coarse[n + 1] != y2_fine[ind + 1]:
                    vols_grad_coarse[:, n] += 0.5 * (vols_eff_shift_coarse[:, n - 1] - vols_eff_shift_fine[:, ind - 1])

    elif comp == 'w':
        shape_fine = dinv_fine.shape
        shape_coarse_y = [nz_fine + 1, ny_coarse, nx_fine]
        vols_grad_coarse_y = np.zeros(shape_coarse_y)
        vols_eff_shift_coarse_y = np.zeros(shape_coarse_y)

        ind_prev = 0
        for n,edge in enumerate(y2_coarse[1:]):
            ind = np.argwhere(y2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_y[:, n + ncrop] += vols_eff_shift_fine[:, m + ncrop]
                vols_grad_coarse_y[:,  n + ncrop] += vol_grad_fine[:,  m + ncrop]
            ind_prev = ind

        vol_grad_fine = vols_grad_coarse_y
        vols_eff_shift_fine = vols_eff_shift_coarse_y

        shape_coarse_x = [nz_fine + 1, ny_coarse, nx_coarse]
        vols_grad_coarse_x = np.zeros(shape_coarse_x)
        vols_eff_shift_coarse_x = np.zeros(shape_coarse_x)

        ind_prev = 0
        for n,edge in enumerate(x2_coarse[1:]):
            ind = np.argwhere(x2_fine == edge)[0][0]
            for m in range(ind_prev, ind):
                vols_eff_shift_coarse_x[:, :, n + ncrop] += vols_eff_shift_fine[:, :, m + ncrop]
                vols_grad_coarse_x[:, :, n + ncrop] += vol_grad_fine[:, :, m + ncrop]
            ind_prev = ind

        vols_eff_shift_fine = vols_eff_shift_coarse_x

        shape_coarse = [nz_coarse + 1, ny_coarse, nx_coarse]
        vols_grad_coarse = np.zeros(shape_coarse)


        for n, edge in enumerate(z2_coarse[:]):
            ind = np.argwhere(z2_fine == edge)[0][0]
            vols_grad_coarse[n] = vols_grad_coarse_x[ind] + 0.5 * (vols_eff_shift_coarse[n] - vols_eff_shift_fine[ind])
            if all((n > 0, n < z2_coarse.size - 1, side == 'l')):
                if z2_coarse[n - 1] != z2_fine[ind - 1]:
                    vols_grad_coarse[n] += 0.5 * (vols_eff_shift_coarse[n + 1] - vols_eff_shift_fine[ind + 1])
            if all((n > 0, n < z2_coarse.size - 1, side == 'r')):
                if z2_coarse[n + 1] != z2_fine[ind + 1]:
                    vols_grad_coarse[n] += 0.5 * (vols_eff_shift_coarse[n - 1] - vols_eff_shift_fine[ind - 1])


    dginv_c = 1.0 * area_eff_coarse / (vols_grad_coarse + 1e-200)
   
    
    return dginv_c


def coarse_gradient_coeff(
                              dinv_fine_comp, vols_eff1d_fine, vols_eff1d_coarse, 
                              area_eff1d_fine, area_eff1d_coarse, edges_fine, edges_coarse
                         ):
    """
    An alternative approch to derive the coarse-grid coefficients of the gradient operator
    is not to use arithmetic averaging of the volumes of two adjacent grid cells,
    but instead use only the left- and right-hand side volume to derive
    pairs of different coefficients. This approach produces more consistent
    DCA when using semi-coarsening.

    dinv_fine_comp... list of one-sided coefficients of the fine-grid gradient components
    vols_eff_shift_fine... one-sided effective volume of the fine grid
    vols_eff_shift_coarse... one-sided effective volume of the coarse grid
    area_eff_fine... effective area at gradient face of the fine grid
    area_eff_coarse... effective area at gradient face of the coarse grid
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid

    dinv_coarse_comp... returned list of one-sided coefficients of the coarse-grid gradient components
    """

    ncrop = 0

    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]

    nx_coarse = x2_coarse.size - 1
    ny_coarse = y2_coarse.size - 1
    nz_coarse = z2_coarse.size - 1
    shape_coarse = [nz_coarse, ny_coarse, nx_coarse]

    nx_fine = x2_fine.size - 1
    ny_fine = y2_fine.size - 1
    nz_fine = z2_fine.size - 1
    shape_fine = [nz_fine, ny_fine, nx_fine]
   
    vols_eff_fine = vols_eff1d_fine.reshape(shape_fine)
    area_eff_x_fine, area_eff_y_fine, area_eff_z_fine = ops.put_3d(area_eff1d_fine, shape_fine)
    vols_eff_coarse = vols_eff1d_coarse.reshape(shape_coarse)
    area_eff_x_coarse, area_eff_y_coarse, area_eff_z_coarse = ops.put_3d(area_eff1d_coarse, shape_coarse)

    dinvx_l_fine, dinvx_r_fine, dinvy_l_fine, dinvy_r_fine, dinvz_l_fine, dinvz_r_fine = dinv_fine_comp[:]
     

    vols_eff_shift_fine = np.zeros([nz_fine, ny_fine, nx_fine + 1])
    vols_eff_shift_coarse = np.zeros([nz_coarse, ny_coarse, nx_coarse + 1])
    vols_eff_shift_fine[:, :, 1:] = vols_eff_fine    
    vols_eff_shift_fine[:, :, 0] = vols_eff_fine[:, :, 0]
    vols_eff_shift_coarse[:, :, 1:] = vols_eff_coarse
    vols_eff_shift_coarse[:, :, 0] = vols_eff_coarse[:, :, 0]          
    dinvx_l_coarse = coarse_gradient_comp(
                                             dinvx_l_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_x_fine, area_eff_x_coarse, edges_fine, edges_coarse, 'u', 'l'
                                         )

    vols_eff_shift_fine[:, :, :-1] = vols_eff_fine
    vols_eff_shift_fine[:, :, -1] = vols_eff_fine[:, :, -1]
    vols_eff_shift_coarse[:, :, :-1] = vols_eff_coarse
    vols_eff_shift_coarse[:, :, -1] = vols_eff_coarse[:, :, -1]
    dinvx_r_coarse = coarse_gradient_comp(
                                             dinvx_r_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_x_fine, area_eff_x_coarse, edges_fine, edges_coarse, 'u', 'r'
                                         )
 
    vols_eff_shift_fine = np.zeros([nz_fine, ny_fine + 1, nx_fine])
    vols_eff_shift_coarse = np.zeros([nz_coarse, ny_coarse + 1, nx_coarse])
    vols_eff_shift_fine[:, 1:] = vols_eff_fine
    vols_eff_shift_fine[:, 0] = vols_eff_fine[:, 0]
    vols_eff_shift_coarse[:, 1:] = vols_eff_coarse
    vols_eff_shift_coarse[:, 0] = vols_eff_coarse[:, 0]
    dinvy_l_coarse = coarse_gradient_comp(
                                             dinvy_l_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_y_fine, area_eff_y_coarse, edges_fine, edges_coarse, 'v', 'l'
                                         )
   
    vols_eff_shift_fine[:, :-1] = vols_eff_fine
    vols_eff_shift_fine[:, -1] = vols_eff_fine[:, -1]
    vols_eff_shift_coarse[:, :-1] = vols_eff_coarse
    vols_eff_shift_coarse[:, -1] = vols_eff_coarse[:, -1]
    dinvy_r_coarse = coarse_gradient_comp(
                                             dinvy_r_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_y_fine, area_eff_y_coarse, edges_fine, edges_coarse, 'v', 'r'
                                         )
 
    vols_eff_shift_fine = np.zeros([nz_fine + 1, ny_fine, nx_fine])
    vols_eff_shift_coarse = np.zeros([nz_coarse + 1, ny_coarse, nx_coarse])
    vols_eff_shift_fine[1:] = vols_eff_fine
    vols_eff_shift_fine[0] = vols_eff_fine[0]
    vols_eff_shift_coarse[1:] = vols_eff_coarse
    vols_eff_shift_coarse[0] = vols_eff_coarse[0]
    dinvz_l_coarse = coarse_gradient_comp(
                                             dinvz_l_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_z_fine, area_eff_z_coarse, edges_fine, edges_coarse, 'w', 'l'
                                         )

    vols_eff_shift_fine[:-1] = vols_eff_fine
    vols_eff_shift_fine[-1] = vols_eff_fine[-1]
    vols_eff_shift_coarse[:-1] = vols_eff_coarse
    vols_eff_shift_coarse[-1] = vols_eff_coarse[-1]
    dinvz_r_coarse = coarse_gradient_comp(
                                             dinvz_r_fine, vols_eff_shift_fine, vols_eff_shift_coarse, 
                                             area_eff_z_fine, area_eff_z_coarse, edges_fine, edges_coarse, 'w', 'r'
                                         )

    dinv_coarse_comp = [dinvx_l_coarse, dinvx_r_coarse, dinvy_l_coarse, dinvy_r_coarse, dinvz_l_coarse, dinvz_r_coarse]
    return dinv_coarse_comp


def compute_idealized_approx_smoothfac(dx, dy, dz):
    """
    Derives the approximately idealized smoothing factor for
    Red-Black over-relaxation originally presented in Yavneh (1996)
    and reused in Larrson et al. (2005). It is derived from the coefficient
    anisotropy reflecting grid stretching only (no discontinuities).
    
    dx... grid spacings in first dimension
    dy... grid spacings in second dimension
    dz... grid spacings in third dimension
   
    mu_x, mu_y, mu_z... returns the tuple of idealized smoothing factors in each
                        dimension

    """

    areas_x = dy.reshape(1, dy.size, 1) * dz.reshape(dz.size, 1, 1)
    areas_y = dx.reshape(1, 1, dx.size) * dz.reshape(dz.size, 1, 1)
    areas_z = dx.reshape(1, 1, dx.size) * dy.reshape(1, dy.size, 1)
    volumes = dx.reshape(1, 1, dx.size) * dy.reshape(1, dy.size, 1) * dz.reshape(dz.size, 1, 1)

    a_x = areas_x[1:-1, 1:-1] ** 2 * np.sqrt(0.5 * (
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[1:-1, 1:-1, :-2]) * volumes[1:-1, 1:-1, 1:-1])) ** 2 +
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[1:-1, 1:-1, 2:]) * volumes[1:-1, 1:-1, 1:-1])) ** 2
                                                   )
                                            )
    
    a_y = areas_y[1:-1, :, 1:-1] ** 2 * np.sqrt(0.5 * (
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[1:-1, :-2, 1:-1]) * volumes[1:-1, 1:-1, 1:-1])) ** 2 +
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[1:-1, 2:, 1:-1]) * volumes[1:-1, 1:-1, 1:-1])) ** 2
                                                   )
                                            )
    a_z = areas_z[:, 1:-1, 1:-1] ** 2 * np.sqrt(0.5 * (
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[:-2, 1:-1, 1:-1]) * volumes[1:-1, 1:-1, 1:-1])) ** 2 +
                                                       (1.0 / (0.5 * (volumes[1:-1, 1:-1, 1:-1] + volumes[2:, 1:-1, 1:-1]) * volumes[1:-1, 1:-1, 1:-1])) ** 2
                                                   )
                                            )
    a_norm = a_x + a_y + a_z

    a_x = a_x / a_norm
    a_y = a_y / a_norm
    a_z = a_z / a_norm  

    mu_x = (1.0 - a_x) ** 2
    mu_y = (1.0 - a_y) ** 2
    mu_z = (1.0 - a_z) ** 2
    

    return mu_x, mu_y, mu_z


def calc_omega_or_ub(MAT_ghost, shape, nhalo):
    """
    Derives the approximately optimal over-relaxation parameter
    for RB-SOR based on coefficient anisotropy (Yavneh, 1996; Larsson et al., 2005)

    MAT_ghost... matrix with nhalo number of ghost layers
    shape... 3d-shape of grid (number of cells)
    nhalo... number of halo layers

    omega_or... returned array of local over-relaxation factors
    """

    nz, ny, nx = shape[:]
    nz_g, ny_g, nx_g = nz + 2, ny + 2, nx + 2

    data = MAT_ghost.data.tolist()
    indices = MAT_ghost.indices.tolist()
    indptr = MAT_ghost.indptr.tolist()

    size = nz * ny * nx
    omega_or = []

    for m in range(size):
        k, i, j = ops.retrieve_kij(nz, ny, nx, m)
        k_g, i_g, j_g = k + 1, i + 1, j + 1
        m_g = k_g * nx_g * ny_g + i_g * nx_g + j_g
        indices_sub = indices[indptr[m_g]:indptr[m_g + 1]]
        data_sub = data[indptr[m_g]:indptr[m_g + 1]]

        left_x = k_g * nx_g * ny_g + i_g * nx_g + j_g - 1
        right_x = k_g * nx_g * ny_g + i_g * nx_g + j_g + 1
        left_y = k_g * nx_g * ny_g + (i_g - 1) * nx_g + j_g
        right_y = k_g * nx_g * ny_g + (i_g + 1) * nx_g + j_g
        left_z = (k_g - 1) * nx_g * ny_g + i_g * nx_g + j_g
        right_z = (k_g + 1) * nx_g * ny_g + i_g * nx_g + j_g

        try:
            c_lx = data_sub[indices_sub.index(left_x)]
        except:
            c_lx = 0.0
        try:
            c_rx = data_sub[indices_sub.index(right_x)]
        except:
            c_rx = 0.0
        try:
            c_ly = data_sub[indices_sub.index(left_y)]
        except:
            c_ly = 0.0
        try:
            c_ry = data_sub[indices_sub.index(right_y)]
        except:
            c_ry = 0.0
        try:
            c_lz = data_sub[indices_sub.index(left_z)]
        except:
            c_lz = 0.0
        try:
            c_rz = data_sub[indices_sub.index(right_z)]
        except:
            c_rz = 0.0

        c_x = 0.5 * (abs(c_lx)  + abs(c_rx))
        c_y = 0.5 * (abs(c_ly)  + abs(c_ry))
        c_z = 0.5 * (abs(c_lz)  + abs(c_rz))

        norm = c_x + c_y + c_z
        c_x = c_x / (norm + 1e-100)
        c_y = c_y / (norm + 1e-100)
        c_z = c_z / (norm + 1e-100)
        cmax = 1.0 - min(min(c_x, c_y), c_z)
        if cmax == 1.0:
            omega_or.append(1.50)
        else:
            omega_or.append(min(2.0 / (1.0 + np.sqrt(1.0 - cmax ** 2)), 1.70))

    omega_or = np.array(omega_or)

    return omega_or


def empirical_optimize_omega_or(comm, x, stepsize, n_iter, param_dict):
    """
    Optimizes the over-relaxation parameter by performing a fixed number of 
    multigrid iterations with the over-relaxation parameter increased at
    each sequence. Stops when the residual norm starts to increase and chooses
    the last optimal values. 

    comm... communicator object
    x... solution vector (zeros)
    b... right-hand side vector
    stepsize... incremend at which over-relaxation parameter is increased
    n_iter... number of V-cycle iterations 
    param_dict... parameter dictionary
    """

    global Dinv_lev
    global smooth_type
    global MAT_lev
    global fld_inds_c, fld_inds_n1_c
    global vols_eff_lev

    omega_or_ref = param_dict['omega_or']

    shp_c = x.shape

    u_rand = np.random.randn(shp_c[0], shp_c[1], shp_c[2] + 1)
    v_rand = np.random.randn(shp_c[0], shp_c[1] + 1, shp_c[2])
    w_rand = np.random.randn(shp_c[0] + 1, shp_c[1], shp_c[2])

    vols_eff = np.zeros_like(x)
    vols_eff[fld_inds_n1_c] = (vols_eff_lev[0]).flatten()

    div_sum_loc = np.sum(divergence(u_rand, v_rand, w_rand, vols_eff))
    div_sum = ddcp.sum_para(comm.mpicomm, div_sum_loc, comm.pids[1:], comm.pids[0])
    volume = ddcp.sum_para(comm.mpicomm, np.sum(vols_eff[fld_inds_c]), comm.pids[1:], comm.pids[0])
     
    b = divergence(u_rand, v_rand, w_rand, np.ones_like(vols_eff)) -  div_sum / volume

    b[vols_eff[fld_inds_c] <= 1e-40] = 0.0

    x.fill(0.0)

    x = mg_solve(comm, x, b, max_tol = 1e-13, niter_max = n_iter, nsmooth_pre = 2, nsmooth_post = 2)
    matx = matmul_dirichlet(comm.mpicomm, MAT_lev[0], (x[fld_inds_c]).flatten(), 0)
    r = b - matx
    r_max_ref = ddcp.max_para(comm.mpicomm, np.max(np.absolute(r)), comm.pids[1:], comm.pids[0])    

    while True:
        omega_or = omega_or_ref + stepsize
        for n, D in enumerate(Dinv_lev):
            if smooth_type == 'Red_Black':
                Dinv_lev[n] = D[0] * omega_or / omega_or_ref, D[1] * omega_or / omega_or_ref
            else:
                Dinv_lev[n] = D * omega_or / omega_or_ref
        for n, SMO in enumerate(SMO_lev):
            SMO_lev[n] = SMO_lev[n] * omega_or / omega_or_ref
        param_dict.update({'omega_or':omega_or})
        x.fill(0.0)
        x = mg_solve(comm, x, b, max_tol = 1e-13, niter_max = n_iter, nsmooth_pre = 2, nsmooth_post = 2)
        matx = matmul_dirichlet(comm.mpicomm, MAT_lev[0], (x[fld_inds_c]).flatten(), 0)
        r = b - matx
        r_max = ddcp.max_para(comm.mpicomm, np.max(np.absolute(r)), comm.pids[1:], comm.pids[0])

        if r_max > r_max_ref:                 
             for n, D in enumerate(Dinv_lev):
                 if smooth_type == 'Red_Black':
                     Dinv_lev[n] = D[0] * omega_or_ref / omega_or, D[1] * omega_or_ref / omega_or  
                 else:
                     Dinv_lev[n] = D * omega_or_ref / omega_or
             for n, SMO in enumerate(SMO_lev):
                 SMO_lev[n] = SMO_lev[n] * omega_or_ref / omega_or
             param_dict.update({'omega_or':omega_or_ref})

             return None
        else:
            r_max_ref = r_max
            omega_or_ref = omega_or
