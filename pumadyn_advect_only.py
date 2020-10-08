#Michael Weger
#weger@tropos.de
#Permoserstrasse 15
#04318 Leipzig                   
#Germany
#Last modified: 10.10.2020

#This is the advection-only routine of the LES dispersion model Cidispy.
#First a potantial-flow solution to an obstalce configuration and initial conditions
#is calculated. The tracer fields are then advected using this potential flow solution.

#Add searching paths
from sys import path
path.append('./MODEL_SRC/') # path to model code


#Import (from) python packages
import numpy as np
import scipy
import os
from mpi4py import MPI
from time import time
from copy import deepcopy
from netCDF4 import Dataset


#Import modules and functions part of the model code
import domain_decomp as ddcp              #routines and objects for domain decomposition and communication
import multigrid as mgmpi                 #routines for the parallel multigrid pressure solver
import explicit_schemes as expl           #routines to calculate all the explicit tendendies in the Navier-Stokes equations
from rk_tables import rk_coef             #coefficients table for the explicit time scheme
import output_fields as out               #routines to deal with model output
import namelist                           #read in the pre-edited namelist file (./namelist.py)
from init_grid import init_grid           #read in the grid information 
from init_emiss import get_emissions      #read in the emission file (optionally)
import coupling as cpl                    #routines for mesoscale coupling and lateral boundary conditions


#MPI communicator
mpicomm = MPI.COMM_WORLD
nproc = mpicomm.Get_size()
rank = mpicomm.Get_rank()

#read namelist (default path is ./namelist)
param_dict = namelist.namelist()
if rank == 0:
    print "SIMULATION SETTINGS:"
    print ""
    for param in param_dict:
        print param + ': ' + str(param_dict[param])
    print ""
    print "Simulation initializing..."

npc = int(param_dict['npc'])
npr = int(param_dict['npr'])

ng = int(param_dict['n_ghost'])
ng1 = ng - 1

if nproc != npc * npr:
    if rank == 0:
        print "Number of processes {} does not match {} * {} = {} spedified in namelist".format(nproc, npc, npr, npc * npr)
    raise ValueError

pids = range(npc * npr)
comm = ddcp.communicator(mpicomm, npr, npc, pids[:])


#read grid fields with geometry information (will be located on root processing node)
nz, ny, nx, grid_flds, fld_tps = init_grid(mpicomm, param_dict)

if rank == 0:
   print "Domain size is {} x {} x {}".format(nz, ny, nx)


#partition domain for parallel computing
nri, ncj = ddcp.partition_domain(ny, nx, npr, npc)
comm.set_dombnds(nz, nri, ncj)


npr = comm.npr
npc = comm.npc
pids = comm.pids
pid = pids.index(rank)
pid_r = int(pid / npc)
pid_c = pid - pid_r * npc


file_path = "./INPUT/"
bnd_files = sorted([file for file in list(os.walk(file_path))[0][2] if all(('.nc' in file, param_dict['simulation_name'] in file, 'bnd' in file))])

#read the passive-tracer fields
emissions, emiss_names = get_emissions(comm, param_dict)

param_dict.update({'ntracer':len(emiss_names)})
param_dict.update({'tracers':emiss_names})


#initialize the schemes for the explicit tendencies
grid_flds_sub, int_ops = expl.init_schemes(comm, grid_flds, param_dict)

#grid_flds_sub are the distributed grid fields
area_eff_x, area_eff_y, area_eff_z = grid_flds_sub[:3]
vols_eff = grid_flds_sub[6]
dginv_x, dginv_y, dginv_z = grid_flds_sub[8:11]
surf_flds = grid_flds_sub[11:13]

#initialize the turbulence recycling scheme
if param_dict['rec_turb']:
    cpl.init_turbrec_scheme(comm, param_dict)

#initialize the multigrid pressure solver
mgmpi.init_multigrid_pressure_solver(comm, grid_flds, grid_flds_sub, param_dict)

#initialize the model fields (distributed on all processing nodes)
flds_sub = expl.init_fields(comm, bnd_files[0], param_dict)
u, v, w = flds_sub[:3]
p_per = flds_sub[3]
rho = flds_sub[4]
thetav = flds_sub[5]
qv = flds_sub[6]
tr_flds = flds_sub[7:-5]
urms, vrms, wrms = flds_sub[-3:]

#list of field type (staggered or cell-centred)
fld_tps = ['u', 'v', 'w', 'c', 'c', 'c', 'c', 'c', 'c', 'c']
for name in emiss_names:
    fld_tps.append('c')

dp = np.zeros_like(p_per)
vort_x, vort_y, vort_z = expl.vorticity(u, v, w)
dp_x, dp_y, dp_z = mgmpi.grad_p(p_per)
turbu = np.zeros_like(u)
turbv = np.zeros_like(v)
turbw = np.zeros_like(w)

#subdomain dimension sizes
nz_sub, ny_sub, nx_sub = thetav[ng:-ng, ng:-ng, ng:-ng].shape


if int(param_dict['seed_turb']):
    cpl.seed_random_fluct(thetav, [u, v, w], param_dict)
    expl.update_bnds(mpicomm, thetav, 'c')

#start and end time  of integration
p_time = param_dict['st_time']
end_time = param_dict['sim_time'] + p_time

#wall-clock times of individual tasks
time_press = 0.0
time_pred = 0.0
turbrec_tim = 0.0
time_adv = 0.0
time_buoy = 0.0


#organize model output
nt = int(end_time / param_dict['dt_out']) + 1
output_times = np.linspace(0, (nt - 1) * param_dict['dt_out'], nt).tolist()
out_step = 0

default_all_output_flds = [u, v, w, p_per, thetav, qv, dp_x, dp_y, dp_z, vort_x, vort_y, vort_z, turbu, turbv, turbw]
all_output_flds = default_all_output_flds + tr_flds

out.organize_outfields(all_output_flds, param_dict, emiss_names)

#coefficient tables of the Runge-Kutta time integration schemes
expl_order = int(param_dict['expl_order'])
coef_sub, corr_coef, pres_solves = rk_coef(expl_order)

#select pressure solver
if param_dict['pres_solver'] == 'multigrid':        # stand-allone multigrid
    solve = mgmpi.mg_solve
elif param_dict['pres_solver'] == 'multigrid-bicg': # multigrid-preconditioned stabilized biconjugate gradient
    solve = mgmpi.bicgstab_solve

#pressure fields at previous time steps for first-guess extrapolation
p_prev_sub = [np.zeros_like(p_per) for k in range(expl_order)]
p_sub = [np.zeros_like(p_per) for k in range(expl_order)]

#velocity fields at previous time steps for radiation boundary condition
vel_flds = [u, v, w]
vel_flds_prev = [u, v, w]
vel_rms = [urms, vrms, wrms]


#START THE SIMULATION
if rank == 0:
    print ""    
    print "Simulation started using {} cores".format(npc * npr)

#update lateral boundary conditions of the global domain for first time
cpl.set_bnds(comm, param_dict, flds_sub, fld_tps, p_time, bnd_files)
cpl.update_vel(comm, param_dict, vel_flds, vel_flds_prev, vel_rms, rho, surf_flds, 0.1, 0.1)
expl.free_slip(u, v, w, area_eff_x, area_eff_y, area_eff_z)    


#do first projection
rho_div = mgmpi.divergence(u, v, w, rho)
p_per[:] = solve(
                    comm, p_per, rho_div,  max_tol=param_dict['pres_res_tol'], niter_max=int(param_dict['pres_niter_max']), 
                    nsmooth_pre=int(param_dict['nsmooth_pre']), nsmooth_post=int(param_dict['nsmooth_post'])
                )
expl.update_bnds(mpicomm, p_per, 'p')
mgmpi.correct_vel_incomp(u, v, w, rho, p_per)
expl.update_bnds(mpicomm, u, 'u')
expl.update_bnds(mpicomm, v, 'v')
expl.update_bnds(mpicomm, w, 'w')

#output the potential-flow solution
out.write_output_fields(comm, all_output_flds, p_time, 99999, param_dict, int_ops)

eps = 1e-10
expl.avg_vel_flds_stag(mpicomm, u, v, w, param_dict)
dt = min(expl.dt_CFL(comm, param_dict), param_dict['dt_max'])

expl.strain(param_dict)

while p_time + eps < end_time:

    # integration of all tracer scalar tendencies

    st_time = time()
    
    for k, fld in enumerate(flds_sub[7:-5]):

        df_int_lst = []
        fld_int = fld.copy()

        dsrc = expl.emit(emissions[k], p_time - int(param_dict['st_time']))
        bnd_type = 'chem'

        ddiff = expl.diffusion(fld_int, fld_tps[7 + k], ktype='heat')[0]

        for i in range(expl_order):
            adv_st_time = time()
            dadv = expl.advect_scalar(fld_int, fluxcon=False)
            time_adv += time() - adv_st_time
            
            df = dadv + ddiff + dsrc
            
            expl.update_bnds(mpicomm, df, bnd_type)

            df_int_lst.append(df)

            fld_int[:] = fld
         
            for l, df in enumerate(df_int_lst):
                fld_int += dt * coef_sub[i][l] * df

        fld[:] = fld_int 

    time_pred += time() - st_time 

    p_time += dt
    if rank == 0:
        print "physical time  {}s".format(p_time - int(param_dict['st_time']))
 
    if p_time + eps >= output_times[out_step]:
        #write output
        if rank == 0:
            print "Write output"

        out.write_output_fields(comm, all_output_flds, p_time, out_step,  param_dict, int_ops)        
        out_step +=1

#END OF SIMULATION
if rank == 0:
    print "Simulation terminated as scheduled!"
    print "Total computing time for pressure correction {}s".format(time_press)
    print "Total computing time for explicit tendencies {}s".format(time_pred)
    print "Total computing time for turbulence recycling {}s".format(turbrec_tim)
    print "Total computing time for advection (part of explicit tendencies) {}s".format(time_adv)
