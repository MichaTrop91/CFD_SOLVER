# Namelist file of a simulation 
# Upon function call, the pre-edited parameters
# are stored in a dictionary.


from collections import OrderedDict

def namelist():
    '''
    Namelist file of a simulation. 
    Upon function call, the pre-edited parameters
    are stored in a dictionary.

    param_dict... parameter dictionary
    '''


    param_dict = OrderedDict()

    # general settings
    ##################

    # simulation name after which the grid file, boundary files, emission file and output files are named
    param_dict["simulation_name"] = "play_ground"

    # number of processes per column for parallel computing
    param_dict["npc"] = 3

    # number of processes per row for parallel computing
    param_dict["npr"] = 3 

    # start time in seconds from reference time in first boundary file
    param_dict["st_time"] = 0.0

    # simulation time in seconds
    param_dict["sim_time"] = 3600.0
                
    # compute tracer dispersion 
    param_dict["with_emissfile"] = True

    # output time stepping in seconds
    param_dict["dt_out"] = 10.0

    # variable names of output 
    # full list: 'U', 'V', 'W', 'Pp', 'THETAv', 'QV', 'DPX', 'DPY', 'DPZ', 'VORTX', 'VORTY', 'VORTZ', 'U_SGS', 'V_SGS', 'W_SGS'
    # Tracer variables as they appear in the emission file are appended automatically

    param_dict["output_fields"] = 'U', 'V', 'W', 'THETAv', 'Pp', 'VORTZ'

    #remap output from height over terrain to z=const levels
    param_dict["conv_output_z"] = False

             
    #lateral boundary conditions 
    ############################

    #for velocity
    #(dirichlet, radiation, cyclic)
    param_dict["bnd_xl"] = "radiation"
    param_dict["bnd_xr"] = "radiation"
    param_dict["bnd_yl"] = "radiation"
    param_dict["bnd_yr"] = "radiation"
    param_dict["bnd_zl"] = "dirichlet"
    param_dict["bnd_zr"] = "dirichlet"

    # for tracers
    # (None, dirichlet, cyclic)
    # None is equivalent to homogeneous-zero Dirichlet
    param_dict["bnd_chem_xl"] = None
    param_dict["bnd_chem_xr"] = None
    param_dict["bnd_chem_yl"] = None
    param_dict["bnd_chem_yr"] = None
    param_dict["bnd_chem_zl"] = None                 
    param_dict["bnd_chem_zr"] = None

    # for pressure solver 
    # (neummann or cyclic; has to be compatible with velocity bc)
    param_dict["bnd_pres_x"] = "neumann"
    param_dict["bnd_pres_y"] = "neumann"
    param_dict["bnd_pres_z"] = "neumann" 


    # top domain damping for velocity components
                
    # relaxation length in model length units
    param_dict["ldamp"] = 0.0

    # relaxation time in seconds
    param_dict["taudamp"] = 1000.0   


    # pressure solver
     #################
              
    #type of pressure solver (multigrid or multigrid-bicg)
    param_dict["pres_solver"] = "multigrid"

    # residual tolerance
    param_dict["pres_res_tol"] = 1e-5
               
    # maximum number of iterations
    param_dict["pres_niter_max"] = 100
                
    # limit the number of multigrid levels                 
    param_dict["ngrids_max"] = 100

    # number of smoothing sweeps
    param_dict["nsmooth_pre"] = 2
    param_dict["nsmooth_post"] = 2

    # which type of smoother (RBOR: Red-black overrelaxation, LOR: Lexicographic overrelaxation,
    #                         LSOR: Lexicographic symmetric overrelaxation, SPAI: Sparse approximate inverse,
    #                         RJAC: Scheduled relaxed Jacobi)
                    
    param_dict["smoother"] = "RBOR"

    # overrelaxation parameter
    param_dict["omega_or"] = 1.0

    # empirically optimize overrelaxation parameter by a sequence of multigrid iterations
    param_dict["optimize_omega_or"] = True

    # RBOR ONLY: pre-calculate spatially varying optimal overrelaxation parameter (pre-defined omega_or is overwritten)
    param_dict["omega_or_field"] = True

    # SPAI ONLY: quality of smoother (potency of sparsity pattern of Poisson matrix)
#    param_dict["SPAI_q"] = 2

    # SPAI ONLY: constrain number of non-zeros in each row of the smoother
#    param_dict["SPAI_nfillmax"] = 30

    # SPAI ONLY: drop additional indices with absolute value below this tolerance                                  
#    param_dict["SPAI_drop_tol"] = 1e-6

    # SPAI ONLY: compute the smoother for the multigrid solver
#    param_dict["comp_smoother"] = True 

    # SPAI ONLY: save the computed smoother                        
#    param_dict["save_smoother"] = True 

    # SPAI ONLY: load a pre-computed smooother     
#    param_dict["load_smoother"] = False        

    # RJAC ONLY: values of the three relaxation parameters
#    param_dict["RJAC_params"] = 2.1, 0.88, 0.52

    #time integration
    #################
 
    # order   
    param_dict["expl_order"] = 3

    # maximum CFL number
    param_dict["cmax"] = 0.7                  

    # constrain time step
    param_dict["dt_max"] = 2.0
              
    # lower constrain to integration time step below which a stability error is raised
    param_dict["dt_min"] = 1e-7
                      
    # minimum free-volume fraction; below this value, grid cells are treated as closed cells
    param_dict["lim_fv"] = 0.05         


    # advection
    ###########

    # Type of advection scheme: (upwind: locally flux-limited linear upwind scheme, ENO, WENO)
    param_dict["adv_scheme"] = "upwind"

    # order of reconstructions (
    #                              upwind:  adv_order >=3 and adv_order%2=1 
    #                              WENO: adv_order {3, 5} 
    #                              ENO: adv_order {3, 5}; true orders are only {2, 3}
    #                          )
    param_dict["adv_order"] = 5

    # number of ghost cells (has to be int(adv_order / 2) + 1, or increased by +1 for adv_order +2)
    param_dict["n_ghost"] = 3
 
    # diffusion
    ###########
  
    # Smagorinsky constant
    param_dict["c_smag"] = 0.15

    # factor to increase vertical mixing length
    param_dict["mag_vdiff"] = 1.0


    # surface fluxes
    ################

    # compute surface fluxes for potential temperature
    param_dict["src_theta"] = True

    # compute surface fluxes for specific humidity
    param_dict["src_qv"] = False

    # compute fluxes from vertical surfaces
    param_dict["hor_fluxes"] = True

    # compute fluxes from elevated surfaces
    param_dict["elev_fluxes"] = True


    # large scale phenomena
    #######################

    # compute Coriolis term
    param_dict["coriolis"] = False

    # Coriolis frequency
    param_dict["omega_coriolis"] = 7.2921e-5
   
    # latitude in degrees_north
    param_dict["latitude"] = 45.0
                  
    # large-scale pressure gradient computed from geostrophic approximation
    param_dict["lsc_pres"] = False
                 

    # initial and lateral boundary turbulence generation
    ####################################################
    
    # use random theta fluctuations to initialize turbulence
    param_dict["seed_turb"] = True    

    # use turbulence recycling schemes at the horizontal domain boundaries
    param_dict["rec_turb"] = True

    # lateral boundaries where turbulence generation is employed
    param_dict["rec_turb_sides"] = 'w', 'e', 's', 'n'

    # distances in model units of turbulence recycling planes from inflow boundaries ordered in 'w', 'e', 's', 'n'
    param_dict["rec_turb_plane_dists"] = 380.0, 380.0, 380.0, 380.0

    # horizontal length scale in model units of filter employed in turbulence recycling scheme
    param_dict["turb_lscale_h"] = 120.0 

    # invert the turbulent fluctuations to prevent excessive correlation of turbulence
    param_dict["rec_turb_sign"] = 1

    # maximum amplification factor for turbulence at the inflow boundary
    param_dict["rec_turb_maxfac"] = 1.5


    return param_dict
