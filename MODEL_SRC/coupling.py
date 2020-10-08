# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# last modified: 10.10.2020

# This script contains 
# ... all the routines for the organization of global domain boundaries and the data transfer
# ... a turbulence recycling scheme to provide transient inflow conditions

# load external python packages
import numpy as np
from netCDF4 import Dataset
from copy import deepcopy

# load model specific *py files
import domain_decomp as ddcp
import filtering as filt

class bound_glob(ddcp.bound):
   """
   These are boundary objects residing
   on each node that share a part of the global
   domain boundary. Two different time steps are
   stored and the actual values are calculated using
   linear interpolation.
   """

   def __init__(self, fld, inds, pid, side, st_time, tag=None):
       """
       Function to initialize a boundary field.
       
       fld... global boundary part needed on  subdomain
       inds... indices refering  to the boundary  values of subdomain
       pid... process id of subdomain
       st_time... initial time in seconds
       """

       self.inds = inds
       self.pid = pid
       self.tag = tag
       self.t_next = st_time
       self.t_prev = st_time
       self.data_next = fld.copy()
       self.data_prev = fld.copy()
       self.side = side
       shp = [len(set(inds_comp.tolist())) for inds_comp in inds]
       self.shape = shp
       sl = [(np.min(array), np.max(array) + 1) for array in inds]
       self.slice = sl

   def update(self, data):
       """
       Function to update data after loading a new boundary file
       data... data of the next time step
       """

       self.data_prev[:] = self.data_next.copy()
       self.data_next[:] = data.copy()

   def update_bndtime(self, time_next):
       """
       Function to update the time step after loading a new boundary file
       time_next... time in seconds of the next time step
       """

       self.t_prev = float(self.t_next)
       self.t_next = float(time_next)

   def interpolate(self, time):
       """
       Function to interpolate between the two stored time steps.
       time... model time in seconds
       """

       if time > self.t_next or time < self.t_prev:
           print "Warning: Invalid boundary times!"
       self.data = self.data_prev + (self.data_next - self.data_prev) / (self.t_next - self.t_prev) * (time - self.t_prev)

   def add(self, fld):
       """
       Function to add a field.
       fld... field to add
       """

       self.data[:] = self.data + fld

   def mul(self, fld):
       """
       Function to multiply with a field.
       fld... field to multiply
       """

       self.data[:] = self.data * fld



def make_gobal_bnds(comm, param_dict, bnds, type='c'):
    """
    Creates global boundary objects to communicate and store boundary values.
  
    comm... ddcp.communicator object
    param_dict... parameter dictionary
    bnds... types of lateral boundary conditions
    type... field type ('c' for volume centred, 'u','v','w' for area centred)
    """

    global ng

    mpicomm = comm.mpicomm
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids

    rank = mpicomm.Get_rank()

    bnd_xl, bnd_xr, bnd_yl, bnd_yr, bnd_zl, bnd_zr = bnds[:]

    ng1 = ng - 1
    bnds_global_s = []

    if rank == 0:
        for ind_p_s, pid_s in enumerate(pids):
            ind_pr_s =  ind_p_s / (len(ncj) - 1)
            ind_pc_s = ind_p_s - ind_pr_s * (len(ncj) - 1)

            if ind_pc_s == 0 and bnd_xl in ['dirichlet', 'radiation']:
            # western boundary
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                jc_st = 0
                jc_end = ng
                k_st = ng
                k_end = nz + ng
                if type == 'v':
                    ir_end += 1
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'w', tag + 1000000))

            if ind_pc_s == npc - 1 and bnd_xr in ['dirichlet', 'radiation']:
            # eastern boundary
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                jc_st = ncj[-1] + ng
                jc_end = ncj[-1] + 2 * ng
                k_st = ng
                k_end = nz + ng
                if type == 'u':
                    jc_st += 1
                    jc_end += 1
                if type == 'v':
                    ir_end += 1
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'e', tag + 1000000))

            if ind_pr_s == 0 and bnd_yl in ['dirichlet', 'radiation']:
            # southern boundary
                ir_st = 0
                ir_end = ng
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = ng
                k_end = nz + ng
                if type == 'u':
                    jc_end += 1
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 's', tag + 1000000))

            if ind_pr_s == npr - 1 and bnd_yr in ['dirichlet', 'radiation']:
            # northern boundary
                ir_st = nri[-1] + ng
                ir_end = nri[-1] + 2 * ng
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = ng
                k_end = nz + ng
                if type == 'v':
                    ir_st += 1
                    ir_end += 1
                if type == 'u':
                    jc_end += 1
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'n', tag + 1000000))

            if bnd_zl in ['dirichlet', 'radiation']:
            # bottom boundary
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = 0
                k_end = ng
                if type == 'u':
                    jc_end += 1
                if type == 'v':
                    ir_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'b', tag + 1000000))

            if bnd_zr in ['dirichlet', 'radiation']:
            # top boundary
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = nz + ng 
                k_end = nz + 2 * ng 
                if type == 'u':
                    jc_end += 1
                if type == 'v':
                    ir_end += 1
                if type == 'w':
                    k_st += 1
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 't', tag + 1000000))
    
            if all((ind_pc_s == 0, ind_pr_s == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_yl in ['dirichlet', 'radiation'])):
            # south-western edge
                ir_st = 0
                ir_end = ng
                jc_st = 0
                jc_end = ng
                k_st = 0
                k_end = nz + 2 * ng
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'sw', tag + 1000000))

            if all((ind_pc_s == npc - 1, ind_pr_s == 0, bnd_xr in ['dirichlet', 'radiation'] or bnd_yl in ['dirichlet', 'radiation'])):
            # south-eastern edge
                ir_st = 0
                ir_end = ng
                jc_st = ncj[-1] + ng
                jc_end = ncj[-1] + 2 * ng
                k_st = 0
                k_end = nz + 2 * ng
                if type == 'w':
                    k_end += 1
                if type == 'u':
                    jc_st += 1
                    jc_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'se', tag + 1000000))

            if all((ind_pc_s == 0, ind_pr_s == npr - 1, bnd_xl in ['dirichlet', 'radiation'] or bnd_yr in ['dirichlet', 'radiation'])):
            # north-western edge
                ir_st = nri[-1] + ng
                ir_end = nri[-1] + 2 * ng
                jc_st = 0
                jc_end = ng
                k_st = 0
                k_end = nz + 2 * ng
                if type == 'w':
                    k_end += 1 
                if type == 'v':
                    ir_st += 1
                    ir_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1 
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'nw', tag + 1000000))
      
            if all((ind_pc_s == npc - 1, ind_pr_s == npr - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_yr in ['dirichlet', 'radiation'])):
            # north-eastern edge
                ir_st = nri[-1] + ng
                ir_end = nri[-1] + 2 * ng
                jc_st = ncj[-1] + ng
                jc_end = ncj[-1] + 2 * ng
                k_st = 0
                k_end = nz + 2 * ng
                if type == 'u': 
                    jc_st += 1
                    jc_end += 1
                if type == 'v':
                    ir_st += 1
                    ir_end += 1
                if type == 'w':
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'ne', tag + 1000000))

            if all((ind_pc_s == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
            # bottom-western edge
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                k_st = 0
                k_end = ng
                jc_st = 0
                jc_end = ng
                if type == 'v':
                    ir_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'bw', tag + 1000000))     

            if all((ind_pc_s == npc - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
            # bottom-eastern edge
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                k_st = 0
                k_end = ng
                jc_st = ncj[-1] + ng
                jc_end = ncj[-1] + 2 * ng
                if type == 'u':
                    jc_st += 1
                    jc_end += 1
                if type == 'v':
                    ir_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'be', tag + 1000000))
            
            if all((ind_pr_s == 0, bnd_yl in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
            # bottom-southern edge
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = 0
                k_end = ng
                ir_st = 0
                ir_end = ng
                if type == 'u':
                    jc_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'bs', tag + 1000000))
    
            if all((ind_pr_s == npr - 1, bnd_yr in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
            # bottom-northern edge
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = 0
                k_end = ng
                ir_st = nri[-1] + ng
                ir_end = nri[-1] + 2 * ng
                if type == 'v':
                    ir_st += 1
                    ir_end += 1
                if type == 'u':
                    jc_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'bn', tag + 1000000))

            if all((ind_pc_s == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
            # top-western edge
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                k_st = nz + ng
                k_end = nz + 2 * ng
                jc_st = 0
                jc_end = ng
                if type == 'v':
                    ir_end += 1
                if type == 'w':
                    k_st += 1
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'tw', tag + 1000000))

            if all((ind_pc_s == npc - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
            # top-eastern edge
                ir_st = nri[ind_pr_s] + ng
                ir_end = nri[ind_pr_s + 1] + ng
                k_st = nz + ng
                k_end = nz + 2 * ng
                jc_st = ncj[-1] + ng
                jc_end = ncj[-1] + 2 * ng
                if type == 'u':
                    jc_st += 1
                    jc_end += 1
                if type == 'v':
                    ir_end += 1
                if type == 'w':
                    k_st += 1
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'te', tag + 1000000))

            if all((ind_pr_s == 0, bnd_yl in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
            # top-southern edge
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = nz + ng
                k_end = nz + 2 * ng
                ir_st = 0
                ir_end = ng
                if type == 'u':
                    jc_end += 1
                if type == 'w':
                    k_st += 1
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'ts', tag + 1000000))

            if all((ind_pr_s == npr - 1, bnd_yr in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
            # top-northern edge
                jc_st = ncj[ind_pc_s] + ng
                jc_end = ncj[ind_pc_s + 1] + ng
                k_st = nz + ng
                k_end = nz + 2 * ng
                ir_st = nri[-1] + ng
                ir_end = nri[-1] + 2 * ng
                if type == 'v':
                    ir_st += 1
                    ir_end += 1
                if type == 'u':
                    jc_end += 1
                if type == 'w':
                    k_st += 1
                    k_end += 1
                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
                fld = np.zeros([len(inds[0])])
                tag = pid_s + 1
                bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 'tn', tag + 1000000))

    ind_p = pids.index(rank)
    ind_pr = ind_p / (len(ncj) - 1)
    ind_pc = ind_p - ind_pr * (len(ncj) - 1)
    nc_sub = ncj[ind_pc + 1] - ncj[ind_pc]
    nr_sub = nri[ind_pr + 1] - nri[ind_pr]


    bnds_global_r = []
    bnds_global_m = []
    
    if ind_pc == 0 and bnd_xl in ['dirichlet', 'radiation']:
    # western boundary
        ir_st = ng
        ir_end = nr_sub + ng 
        jc_st = 0
        jc_end = ng
        k_st = ng
        k_end = nz + ng
        if type == 'v':
            ir_end += 1
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'w', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'w', tag + 1000000))
    
    if ind_pc == npc - 1 and bnd_xr in ['dirichlet', 'radiation']:
    # eastern boundary
        ir_st = ng
        ir_end = nr_sub + ng 
        jc_st = nc_sub + ng
        jc_end = nc_sub + 2 * ng
        k_st = ng
        k_end = nz + ng
        if type == 'u':
            jc_st += 1
            jc_end += 1
        if type == 'v':
            ir_end += 1
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'e', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'e', tag + 1000000))

    if ind_pr == 0 and bnd_yl in ['dirichlet', 'radiation']:
    # southern boundary
        ir_st = 0
        ir_end = ng
        jc_st = ng
        jc_end = nc_sub + ng
        k_st = ng
        k_end = nz + ng
        if type == 'u':
            jc_end += 1
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 's', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 's', tag + 1000000))

    if ind_pr == npr - 1 and bnd_yr in ['dirichlet', 'radiation']:
    # northern boundary
        ir_st = nr_sub + ng
        ir_end = nr_sub + 2 * ng
        jc_st = ng
        jc_end = nc_sub + ng
        k_st = ng
        k_end = nz + ng
        if type == 'v':
            ir_st += 1
            ir_end += 1
        if type == 'u':
            jc_end += 1
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'n', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'n', tag + 1000000))

    if bnd_zl in ['dirichlet', 'radiation']:
    # bottom boundary
        ir_st = ng
        ir_end = nr_sub + ng
        jc_st = ng
        jc_end = nc_sub + ng
        k_st = 0
        k_end = ng
        if type == 'u':
            jc_end += 1
        if type == 'v':
            ir_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'b', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'b', tag + 1000000))

    if bnd_zr in ['dirichlet', 'radiation']:
    # top boundary
        ir_st = ng
        ir_end = nr_sub + ng
        jc_st = ng
        jc_end = nc_sub + ng
        k_st = nz + ng 
        k_end = nz + 2 * ng
        if type == 'u':
            jc_end += 1
        if type == 'v':
            ir_end += 1
        if type == 'w':
            k_st += 1
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 't', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 't', tag + 1000000))

    if all((ind_pc == 0, ind_pr == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_yl in ['dirichlet', 'radiation'])):
    # south-western edge
        k_st = 0 
        k_end = nz + 2 * ng
        ir_st = 0
        ir_end = ng
        jc_st = 0
        jc_end = ng
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'sw', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'sw', tag + 1000000))

    if all((ind_pc == npc - 1, ind_pr == 0, bnd_xr in ['dirichlet', 'radiation'] or bnd_yl in ['dirichlet', 'radiation'])):
    # south-eastern edge
        k_st = 0
        k_end = nz + 2 * ng
        ir_st = 0
        ir_end = ng
        jc_st = nc_sub + ng
        jc_end = nc_sub + 2 * ng
        if type == 'w':
            k_end += 1
        if type == 'u':
            jc_st += 1
            jc_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'se', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'se', tag + 1000000))

    if all((ind_pc == 0, ind_pr == npr - 1, bnd_xl in ['dirichlet', 'radiation'] or bnd_yr in ['dirichlet', 'radiation'])):
    # north-western edge
        k_st = 0
        k_end = nz + 2 * ng
        ir_st = nr_sub + ng
        ir_end = nr_sub + 2 * ng
        jc_st = 0
        jc_end = ng
        if type == 'w':
            k_end += 1
        if type == 'v':
            ir_st += 1
            ir_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'nw', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'nw', tag + 1000000))

    if all((ind_pc == npc - 1, ind_pr == npr - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_yr in ['dirichlet', 'radiation'])):
    # north-eastern edge
        k_st = 0
        k_end = nz + 2 * ng
        ir_st = nr_sub + ng
        ir_end = nr_sub + 2 * ng
        jc_st = nc_sub + ng
        jc_end = nc_sub + 2 * ng
        if type == 'v':
            ir_st += 1
            ir_end += 1
        if type == 'u':
            jc_st += 1
            jc_end += 1
        if type == 'w':
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'ne', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'ne', tag + 1000000))

    if all((ind_pc == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
    # bottom-western edge
        ir_st = ng
        ir_end = nr_sub + ng
        k_st = 0
        k_end = ng
        jc_st = 0
        jc_end = ng
        if type == 'v':
            ir_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'bw', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'bw', tag + 1000000))

    if all((ind_pc == npc - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
    # bottom-eastern edge
        ir_st = ng
        ir_end = nr_sub + ng
        k_st = 0
        k_end = ng
        jc_st = nc_sub + ng
        jc_end = nc_sub + 2 * ng
        if type == 'u':
            jc_st += 1
            jc_end += 1
        if type == 'v':
            ir_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'be', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'be', tag + 1000000))      

    if all((ind_pr == 0, bnd_yl in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
    # bottom-southern edge
        ir_st = 0 
        ir_end = ng
        k_st = 0
        k_end = ng
        jc_st = ng
        jc_end = nc_sub + ng
        if type == 'u':
            jc_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'bs', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'bs', tag + 1000000))

    if all((ind_pr == npr - 1, bnd_yr in ['dirichlet', 'radiation'] or bnd_zl in ['dirichlet', 'radiation'])):
    # bottom-northern edge
        ir_st = nr_sub + ng
        ir_end = nr_sub + 2 * ng
        k_st = 0
        k_end = ng
        jc_st = ng
        jc_end = nc_sub + ng
        if type == 'v':
            ir_st += 1
            ir_end += 1
        if type == 'u':
            jc_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'bn', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'bn', tag + 1000000))

    if all((ind_pc == 0, bnd_xl in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
    # top-western edge
        ir_st = ng
        ir_end = nr_sub + ng
        k_st = nz + ng
        k_end = nz + 2 * ng
        jc_st = 0
        jc_end = ng
        if type == 'v':
            ir_end += 1
        if type == 'w':
            k_st += 1
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'tw', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'tw', tag + 1000000))

    if all((ind_pc == npc - 1, bnd_xr in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
    # top-eastern edge
        ir_st = ng
        ir_end = nr_sub + ng
        k_st = nz + ng
        k_end = nz + 2 * ng
        jc_st = nc_sub + ng
        jc_end = nc_sub + 2 * ng
        if type == 'u':
            jc_st += 1
            jc_end += 1
        if type == 'v':
            ir_end += 1
        if type == 'w':
            k_st += 1
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'te', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'te', tag + 1000000))

    if all((ind_pr == 0, bnd_yl in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
    # top-southern edge
        ir_st = 0
        ir_end = ng
        k_st = nz + ng
        k_end = nz + 2 * ng
        jc_st = ng
        jc_end = nc_sub + ng
        if type == 'u':
            jc_end += 1
        if type == 'w':
            k_st += 1
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'ts', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'ts', tag + 1000000))

    if all((ind_pr == npr - 1, bnd_yr in ['dirichlet', 'radiation'] or bnd_zr in ['dirichlet', 'radiation'])):
    # top-northern edge
        ir_st = nr_sub + ng
        ir_end = nr_sub + 2 * ng
        k_st = nz + ng
        k_end = nz + 2 * ng
        jc_st = ng
        jc_end = nc_sub + ng
        if type == 'v':
            ir_st += 1
            ir_end += 1
        if type == 'u':
            jc_end += 1
        if type == 'w':
            k_st += 1
            k_end += 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = rank + 1
        bnds_global_m.append(bound_glob(fld, inds, 0, 'tn', int(param_dict['st_time']), tag + 1000000))
        bnds_global_r.append(ddcp.bound(fld, inds, 0, 'tn', tag + 1000000))                    

    return bnds_global_r, bnds_global_s, bnds_global_m


def make_gobal_diag(comm, param_dict, type='c'):
    """
    Creates objects to communicate and store 3d diagnostic fields
    used for surface forcing  and turbulence generation.

    comm... ddcp.communicator object
    param_dict... parameter dictionary
    type... field type ('c' for volume centred, 'u','v','w' for area centred)
    """

    global ng

    mpicomm = comm.mpicomm
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids

    rank = mpicomm.Get_rank()

    bnds_global_s = []

    nc_full = ncj[-1] + 2 * ng
    nr_full = nri[-1] + 2 * ng

    nz_sub = nz + 2 * ng

    if type == 'u':
        nc_full += 1
    if type == 'v':
        nr_full += 1
    if type == 'w':
        nz_sub += 1

    if rank == 0:
        for ind_p_s, pid_s in enumerate(pids):
            ind_pr_s =  ind_p_s / (len(ncj) - 1)
            ind_pc_s = ind_p_s - ind_pr_s * (len(ncj) - 1)
            ir_st = nri[ind_pr_s]
            ir_end = nri[ind_pr_s + 1] + 2 * ng
            jc_st = ncj[ind_pc_s]
            jc_end = ncj[ind_pc_s + 1] + 2 * ng
            k_st =  0
            k_end = nz_sub

            if type == 'u':
                jc_end += 1
            if type == 'v':
                ir_end += 1

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            tag = pid_s + 1
            bnds_global_s.append(ddcp.bound(fld, inds, pid_s, 't', tag + 1000000))

    ind_p = pids.index(rank)
    ind_pr = ind_p / (len(ncj) - 1)
    ind_pc = ind_p - ind_pr * (len(ncj) - 1)
    nc_sub = ncj[ind_pc + 1] - ncj[ind_pc] + 2 * ng
    nr_sub = nri[ind_pr + 1] - nri[ind_pr] + 2 * ng

    if type == 'u':
        nc_sub += 1
    if type == 'v':
        nr_sub += 1

    bnds_global_r = []
    bnds_global_m = []

    ir_st = 0
    ir_end = nr_sub
    jc_st = 0
    jc_end = nc_sub
    k_st = 0
    k_end = nz_sub

    inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
    fld = np.zeros([len(inds[0])])
    tag = rank + 1
    bnds_global_m.append(bound_glob(fld, inds, 0, 't', int(param_dict['st_time']), tag + 1000000))
    bnds_global_r.append(ddcp.bound(fld, inds, 0, 't', tag + 1000000))


    return bnds_global_r, bnds_global_s, bnds_global_m



def make_bnds_global_halos(comm, bnd_x, bnd_y, bnd_z, type='c', ghst_inds = 3):
    """
    Creates boundary objects to exchange the points at the boundary corners  
    which is necessary after applying updating global boundary conditions.
    
    comm... ddcp.communicator
    bnd_x, bnd_y, bnd_z... type of boundary conditions
    type... field type ('c' for volume centred, 'u','v','w' for area centred)
    ghst_inds... number of ghost layers
    """

    mpicomm = comm.mpicomm
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz

    rank = mpicomm.Get_rank()

    npr = len(nri) - 1
    npc = len(ncj) - 1

    ind_p = pids.index(rank)
    ind_pr = ind_p / npc
    ind_pc = ind_p - ind_pr * npc

    c_x = 0
    c_y = 0
    c_z = 0

    if type == 'u':
        c_x = 1
    if type == 'v':
        c_y = 1
    if type == 'w':
        c_z = 1

    c_st = ncj[ind_pc]

    c_end = ncj[ind_pc + 1] + 2 * ghst_inds + c_x
    r_st = nri[ind_pr]
    r_end = nri[ind_pr + 1] + 2 * ghst_inds + c_y
          
    nz_sub = nz + 2 * ghst_inds + c_z
    nc_sub = c_end - c_st
    nr_sub = r_end - r_st   
        
    bnds_halo_s = []
    bnds_halo_r = []

    #halo of top boundary
    
    if (ind_pc > 0): 
    # halo has a west-component
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = ghst_inds
        ir_end = nr_sub - ghst_inds
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        if all([ind_pr == 0, not bnd_y == 'cyclic', type == 'p']):
            ir_st = ir_st - 1
        if all([ind_pr == npr - 1, not bnd_y == 'cyclic', type == 'p']):
            ir_end = ir_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'w', tag + 1000000))
        jc_st = 0
        jc_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'w', tag + 1300000))

        k_st = 0
        k_end = ghst_inds
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'w', tag + 1600000))
        jc_st = 0
        jc_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'w', tag + 1900000))


    if (ind_pc < npc - 1):
    # halo has an east-component
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = ghst_inds
        ir_end = nr_sub - ghst_inds
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        if all([ind_pr == 0, not bnd_y == 'cyclic', type == 'p']):
            ir_st = ir_st - 1
        if all([ind_pr == npr - 1, not bnd_y == 'cyclic', type == 'p']):
            ir_end = ir_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'e', tag + 1300000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'e', tag + 1000000))

        k_st = 0
        k_end = ghst_inds
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'e', tag + 1900000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'e', tag + 1600000))

    if (ind_pr > 0):
    # halo has a south-component
        jc_st = ghst_inds
        jc_end = nc_sub - ghst_inds
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        if all([ind_pc == 0, not bnd_x == 'cyclic', type == 'p']):
            jc_st = jc_st - 1
        if all([ind_pc == npc - 1, not bnd_x == 'cyclic', type == 'p']):
            jc_end = jc_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 's', tag + 2200000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 's', tag + 2500000))

        k_st = 0
        k_end = ghst_inds
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 's', tag + 2800000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 's', tag + 3100000))


    if (ind_pr < npr - 1):
    # halo has a north-component
        jc_st = ghst_inds
        jc_end = nc_sub - ghst_inds
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        if all([ind_pc == 0, not bnd_x == 'cyclic', type == 'p']):
            jc_st = jc_st - 1
        if all([ind_pc == npc - 1, not bnd_x == 'cyclic', type == 'p']):
            jc_end = jc_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'n', tag + 2500000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'n', tag + 2200000))

        k_st = 0
        k_end = ghst_inds
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'n', tag + 3100000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'n', tag + 2800000))


    if ind_pc > 0 and ind_pr > 0:
    # halo has a south-western edge
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc - 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'sw', tag + 3400000))
        jc_st = 0
        jc_end = ghst_inds
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'sw', tag + 3700000))

        k_st = 0
        k_end = ghst_inds
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'sw', tag + 4000000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'sw', tag + 4300000))


    if ind_pc < npc - 1 and ind_pr < npr - 1:
    # halo has a north-eastern edge
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc + 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'ne', tag + 3700000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'ne', tag + 3400000))             

        k_st = 0
        k_end = ghst_inds
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'ne', tag + 4300000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'ne', tag + 4000000))


    if ind_pc < npc - 1 and ind_pr > 0:
    # halo has a south-eastern edge
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc + 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'se', tag + 4600000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'se', tag + 4900000))
        
        k_st = 0
        k_end = ghst_inds
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'se', tag + 5200000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'se', tag + 5500000))

    if ind_pc > 0 and ind_pr < npr - 1:
    # halo has a north-western edge
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc - 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'nw', tag + 4900000))
        jc_st = 0
        jc_end = ghst_inds
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'nw', tag + 4600000))
       
        k_st = 0
        k_end = ghst_inds
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'nw', tag + 5500000))
        jc_st = 0
        jc_end = ghst_inds
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'nw', tag + 5200000))

    if ind_pc == 0 and ind_pr > 0:
        jc_st = 0
        jc_end = ghst_inds
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 's', tag + 5800000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 's', tag + 6100000))

    if ind_pc == 0 and ind_pr < npr - 1:
        jc_st = 0
        jc_end = ghst_inds
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'n', tag + 6100000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'n', tag + 5800000))

    if ind_pc == npc - 1 and ind_pr > 0:
        jc_st = nc_sub - ghst_inds 
        jc_end = nc_sub
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 's', tag + 6400000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 's', tag + 6700000))

    if ind_pc == npc - 1 and ind_pr < npr - 1:
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'n', tag + 6700000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'n', tag + 6400000))

    if ind_pr == 0 and ind_pc > 0:
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = 0
        ir_end = ghst_inds
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'w', tag + 7000000))
        jc_st = 0
        jc_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'w', tag + 7300000))

    if ind_pr == 0 and ind_pc < npc - 1:
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = 0
        ir_end = ghst_inds
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'e', tag + 7300000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'e', tag + 7000000))
    
    if ind_pr == npr - 1 and ind_pc > 0:
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'w', tag + 7600000))
        jc_st = 0
        jc_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'w', tag + 7900000))

    if ind_pr == npr - 1 and ind_pc < npc - 1:
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = nr_sub - ghst_inds 
        ir_end = nr_sub 
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + 1]
        tag = pid * rank
        bnds_halo_s.append(ddcp.bound(fld, inds, pid, 'e', tag + 7900000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_halo_r.append(ddcp.bound(fld, inds, pid, 'e', tag + 7600000))

    return bnds_halo_s, bnds_halo_r



def init_bnds_global(comm, param_dict, nfields):
    """ 
    Creates the boundary objects for communication and  data storage of lateral boundary  values
    for all prognostic and diagnostic fields.
 
    comm... ddcp.communicator
    param_dict... paramter dictionary
    nfields... number of fields in total
    """


    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz
    ind_pr = comm.ind_pr
    ind_pc = comm.ind_pc
    pids = comm.pids
  
    npc = comm.npc
    npr = comm.npr
  
    ind_p = pids.index(rank)

    nc_sub = ncj[ind_pc + 1] - ncj[ind_pc] 
    nr_sub = nri[ind_pr + 1] - nri[ind_pr] 

    global bnds_global_lst
    global bnds_global_halo_u_s
    global bnds_global_halo_u_r
    global bnds_global_halo_v_s
    global bnds_global_halo_v_r
    global bnds_global_halo_w_s
    global bnds_global_halo_w_r
    global bnds_global_halo_c_s
    global bnds_global_halo_c_r

    global bnds_global_u_s
    global bnds_global_u_r
    global bnds_global_v_s
    global bnds_global_v_r
    global bnds_global_w_s
    global bnds_global_w_r
    global bnds_global_c_s
    global bnds_global_c_r
    global bnds_global_cp_r
    global bnds_global_cp_s
    global bnds_global_diag_c_s
    global bnds_global_diag_c_r
    global bnds_global_chem_r
    global bnds_global_chem_s

    global gb_sl_x_up_indtpl_lst
    global gb_sl_x_mid_indtpl_lst
    global gb_sl_x_down_indtpl_lst
    global gb_sl_x_urad_mid_indtpl_lst

    global gb_sl_y_up_indtpl_lst
    global gb_sl_y_mid_indtpl_lst
    global gb_sl_y_down_indtpl_lst
    global gb_sl_y_urad_mid_indtpl_lst

    global gb_sl_z_up_indtpl_lst
    global gb_sl_z_mid_indtpl_lst
    global gb_sl_z_down_indtpl_lst
    global gb_sl_z_urad_mid_indtpl_lst

    global u_rad_lst, v_rad_lst, w_rad_lst
    global u_prefac
    
    global bnd_obj_inds

    global bnds_u_rad_avg_send_lst, bnds_u_rad_avg_recv_lst
    global bnds_v_rad_avg_send_lst, bnds_v_rad_avg_recv_lst
    global bnds_w_rad_avg_send_lst, bnds_w_rad_avg_recv_lst
    
    global ng

    ng = int(param_dict['n_ghost'])
    ng1 = ng - 1

    bnds_vel = [
                   param_dict['bnd_xl'], param_dict['bnd_xr'], 
                   param_dict['bnd_yl'], param_dict['bnd_yr'], 
                   param_dict['bnd_zl'], param_dict['bnd_zr']
               ]
    bnds_pres = [
                    param_dict['bnd_pres_x'], param_dict['bnd_pres_x'], 
                    param_dict['bnd_pres_y'], param_dict['bnd_pres_y'], 
                    param_dict['bnd_pres_z'], param_dict['bnd_pres_z']
                ]
    bnds_chem = [
                    param_dict['bnd_chem_xl'], param_dict['bnd_chem_xr'], 
                    param_dict['bnd_chem_yl'], param_dict['bnd_chem_yr'], 
                    param_dict['bnd_chem_zl'], param_dict['bnd_chem_zr']
                ]

    dx = param_dict['dx']
    dy = param_dict['dy']
    dz = param_dict['dz']


    bnds_global_halo_c_s, bnds_global_halo_c_r = make_bnds_global_halos(comm, bnds_vel[0], bnds_vel[1], bnds_vel[2], type='c', ghst_inds=ng)
    bnds_global_halo_u_s, bnds_global_halo_u_r = make_bnds_global_halos(comm, bnds_vel[0], bnds_vel[1], bnds_vel[2], type='u', ghst_inds=ng)
    bnds_global_halo_v_s, bnds_global_halo_v_r = make_bnds_global_halos(comm, bnds_vel[0], bnds_vel[1], bnds_vel[2], type='v', ghst_inds=ng)
    bnds_global_halo_w_s, bnds_global_halo_w_r = make_bnds_global_halos(comm, bnds_vel[0], bnds_vel[1], bnds_vel[2], type='w', ghst_inds=ng)


    #U-wind
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='u')
    bnds_global_u_r, bnds_global_u_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst = [bnds_global_m]
    #V-wind
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='v')
    bnds_global_v_r, bnds_global_v_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #W-wind
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='w')
    bnds_global_w_r, bnds_global_w_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #Perturbation pressure
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_pres, type='c')
    bnds_global_cp_r, bnds_global_cp_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #Density
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='c')
    bnds_global_c_r, bnds_global_c_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #T_pot virt
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='c')
    bnds_global_c_r, bnds_global_c_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #Water vapor mixing ratio
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='c')
    bnds_global_c_r, bnds_global_c_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    #Tracers

    for n in range(7, nfields):
        bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_chem, type='c')
        bnds_global_chem_r, bnds_global_chem_s, bnds_global_m = bnd_global_lists[:]
        bnds_global_lst.append(bnds_global_m)


    #Surface temperature
    bnds_global_diag_c_r, bnds_global_diag_c_s, bnds_global_m = make_gobal_diag(comm, param_dict, type='c')
    bnds_global_lst.append(bnds_global_m)

    #Surface specific humidity
    bnds_global_diag_c_r, bnds_global_diag_c_s, bnds_global_m = make_gobal_diag(comm, param_dict, type='c')
    bnds_global_lst.append(bnds_global_m)

    #turbulent intensities (rms of u, v, w)
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='u')
    bnds_global_u_r, bnds_global_u_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='v')
    bnds_global_v_r, bnds_global_v_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    bnd_global_lists = make_gobal_bnds(comm, param_dict, bnds_vel, type='w')
    bnds_global_w_r, bnds_global_w_s, bnds_global_m = bnd_global_lists[:]
    bnds_global_lst.append(bnds_global_m)
    
    bnd_obj_inds = []
 
    gb_sl_x_up_indtpl_lst = []
    gb_sl_x_mid_indtpl_lst = []
    gb_sl_x_down_indtpl_lst = []
    gb_sl_x_urad_mid_indtpl_lst = []
    gb_sl_y_up_indtpl_lst = []
    gb_sl_y_mid_indtpl_lst = []
    gb_sl_y_down_indtpl_lst = []
    gb_sl_y_urad_mid_indtpl_lst = []
    gb_sl_z_up_indtpl_lst = []
    gb_sl_z_mid_indtpl_lst = []
    gb_sl_z_down_indtpl_lst = []
    gb_sl_z_urad_mid_indtpl_lst = []
    u_rad_lst = [] 
    v_rad_lst = []
    w_rad_lst = []

    u_prefac = []

    bnds_u_rad_avg_send_lst = []
    bnds_u_rad_avg_recv_lst = []
    bnds_v_rad_avg_send_lst = []
    bnds_v_rad_avg_recv_lst = []
    bnds_w_rad_avg_send_lst = []
    bnds_w_rad_avg_recv_lst = []

    if param_dict['bnd_xl'] == 'radiation' and ind_pc == 0:

        gb_sl_x_urad_mid_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (ng, ng + 1)]) 
        gb_sl_x_down_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (0, ng)])
        gb_sl_x_mid_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (1, ng + 1)])
        gb_sl_x_up_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (2, ng + 2)])

        u_rad_lst.append(np.zeros([nz, nr_sub, ng]))
        v_rad_lst.append(np.zeros([nz, nr_sub + 1, ng]))
        w_rad_lst.append(np.zeros([nz + 1, nr_sub, ng]))

        u_prefac.append(-1.0 / dx[0])

        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 'w':
                bnd_obj_inds.append(n)  


    if param_dict['bnd_xr'] == 'radiation' and ind_pc == len(ncj) - 2:

        gb_sl_x_up_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (-ng - 2, -2)])
        gb_sl_x_urad_mid_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (-ng - 1, -ng)])
        gb_sl_x_mid_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (-ng - 1, -1)])
        gb_sl_x_down_indtpl_lst.append([(ng, nz + ng), (ng, nr_sub + ng), (-ng, None)])

        u_rad_lst.append(np.zeros([nz, nr_sub, ng]))
        v_rad_lst.append(np.zeros([nz, nr_sub + 1, ng]))
        w_rad_lst.append(np.zeros([nz + 1, nr_sub, ng]))

        u_prefac.append(1.0 / dx[-1])

        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 'e':
                bnd_obj_inds.append(n)

    if any((param_dict['bnd_xl'] == 'radiation' and ind_pc == 0, param_dict['bnd_xr'] == 'radiation' and ind_pc == len(ncj) - 2)):

        bnds_send = []
        bnds_recv = []

        inds = inds = list(np.array([(k, i, j) for k in range(0, nz) for i in range(0, 1) for j in range(0, ng)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pr > 0:
            pid_s = (ind_pr - 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 's', tag= 111000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 's', tag= 141000 + tag))

        elif param_dict['bnd_yl'] == 'cyclic':
            pid_s = (npr - 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 's', tag= 111000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 's', tag= 141000 + tag))

        inds = inds = list(np.array([(k, i, j) for k in range(0, nz) for i in range(nr_sub, nr_sub + 1) for j in range(0, ng)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pr < npr - 1:
            pid_s = (ind_pr + 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 141000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 111000 + tag))

        elif param_dict['bnd_yl'] == 'cyclic':
            pid_s = ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 141000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 111000 + tag))

        bnds_v_rad_avg_send_lst.append(bnds_send)
        bnds_v_rad_avg_recv_lst.append(bnds_recv)

        bnds_send = []
        bnds_recv = []

        if param_dict['bnd_zl'] == 'cyclic':

            inds = inds = list(np.array([(k, i, j) for k in range(0, 1) for i in range(0, nr_sub) for j in range(0, ng)]).T)
            fld = np.empty([len(inds[0])])
            bnds_send.append(ddcp.bound(fld, inds, rank, 'b', tag= 171000))
            bnds_recv.append(ddcp.bound(fld, inds, rank, 'b', tag= 201000))
            inds = inds = list(np.array([(k, i, j) for k in range(nz, nz + 1) for i in range(0, nr_sub) for j in range(0, ng)]).T)   
            bnds_send.append(ddcp.bound(fld, inds, rank, 't', tag= 201000))
            bnds_recv.append(ddcp.bound(fld, inds, rank, 't', tag= 171000))

        bnds_w_rad_avg_send_lst.append(bnds_send)
        bnds_w_rad_avg_recv_lst.append(bnds_recv)

        if all((param_dict['bnd_xl'] == 'radiation' and ind_pc == 0, param_dict['bnd_xr'] == 'radiation' and ind_pc == len(ncj) - 2)):
            bnds_v_rad_avg_send_lst.append(deepcopy(bnds_v_rad_avg_send_lst[-1]))
            bnds_v_rad_avg_recv_lst.append(deepcopy(bnds_v_rad_avg_recv_lst[-1]))       
            bnds_w_rad_avg_send_lst.append(deepcopy(bnds_w_rad_avg_send_lst[-1]))
            bnds_w_rad_avg_recv_lst.append(deepcopy(bnds_w_rad_avg_recv_lst[-1]))


    if param_dict['bnd_yl'] == 'radiation' and ind_pr == 0:

        gb_sl_y_down_indtpl_lst.append([(ng, nz + ng), (0, ng), (ng, nc_sub + ng)])
        gb_sl_y_urad_mid_indtpl_lst.append([(ng, nz + ng), (ng, ng + 1), (ng, nc_sub + ng)])
        gb_sl_y_mid_indtpl_lst.append([(ng, nz + ng), (1, ng + 1), (ng, nc_sub + ng)])
        gb_sl_y_up_indtpl_lst.append([(ng, nz + ng), (2, ng + 2), (ng, nc_sub + ng)])

        u_rad_lst.append(np.zeros([nz, ng, nc_sub + 1]))
        v_rad_lst.append(np.zeros([nz, ng, nc_sub]))
        w_rad_lst.append(np.zeros([nz + 1, ng, nc_sub]))

        u_prefac.append(-1.0 / dy[0])

        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 's':
                bnd_obj_inds.append(n)
        
    if param_dict['bnd_yr'] == 'radiation' and ind_pr == len(nri) - 2:

        gb_sl_y_up_indtpl_lst.append([(ng, nz + ng), (-ng - 2, -2), (ng, nc_sub + ng)])
        gb_sl_y_mid_indtpl_lst.append([(ng, nz + ng), (-ng - 1, -1), (ng, nc_sub + ng)])
        gb_sl_y_urad_mid_indtpl_lst.append([(ng, nz + ng), (-ng - 1, -ng), (ng, nc_sub + ng)])
        gb_sl_y_down_indtpl_lst.append([(ng, nz + ng), (-ng, None), (ng, nc_sub + ng)])

        u_rad_lst.append(np.zeros([nz, ng, nc_sub + 1]))
        v_rad_lst.append(np.zeros([nz, ng, nc_sub]))
        w_rad_lst.append(np.zeros([nz + 1, ng, nc_sub]))

        u_prefac.append(1.0 / dy[-1])
        
        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 'n':
                bnd_obj_inds.append(n)

    if any((param_dict['bnd_yl'] == 'radiation' and ind_pr == 0, param_dict['bnd_yr'] == 'radiation' and ind_pr == len(nri) - 2)):

        bnds_send = []
        bnds_recv = []

        inds = inds = list(np.array([(k, i, j) for k in range(0, nz) for i in range(0, ng) for j in range(0, 1)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pc > 0:
            pid_s = ind_pr * npc + ind_pc - 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 231000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 261000 + tag))

        elif param_dict['bnd_xl'] == 'cyclic':
            pid_s = ind_pr * npc + npc - 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 231000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 261000 + tag))

        inds = inds = list(np.array([(k, i, j) for k in range(0, nz) for i in range(0, ng) for j in range(nc_sub, nc_sub + 1)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pc < npc - 1:
            pid_s = ind_pr * npc + ind_pc + 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 261000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 231000 + tag))

        elif param_dict['bnd_xl'] == 'cyclic':
            pid_s = ind_pr * npc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 261000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 231000 + tag))

        bnds_u_rad_avg_send_lst.append(bnds_send)
        bnds_u_rad_avg_recv_lst.append(bnds_recv)

        bnds_send = []
        bnds_recv = []

        if param_dict['bnd_zl'] == 'cyclic':

            inds = inds = list(np.array([(k, i, j) for k in range(0, 1) for i in range(0, ng) for j in range(0, nc_sub)]).T)
            fld = np.empty([len(inds[0])])
    
            bnds_send.append(ddcp.bound(fld, inds, rank, 'b', tag= 291000))
            bnds_recv.append(ddcp.bound(fld, inds, rank, 'b', tag= 321000))
            inds = inds = list(np.array([(k, i, j) for k in range(nz, nz + 1) for i in range(0, ng) for j in range(0, nc_sub)]).T)
            bnds_send.append(ddcp.bound(fld, inds, rank, 't', tag= 321000))
            bnds_recv.append(ddcp.bound(fld, inds, rank, 't', tag= 291000))

        bnds_w_rad_avg_send_lst.append(bnds_send)
        bnds_w_rad_avg_recv_lst.append(bnds_recv)

        if all((param_dict['bnd_yl'] == 'radiation' and ind_pr == 0, param_dict['bnd_yr'] == 'radiation' and ind_pr == len(nri) - 2)):
            bnds_u_rad_avg_send_lst.append(deepcopy(bnds_u_rad_avg_send_lst[-1]))
            bnds_u_rad_avg_recv_lst.append(deepcopy(bnds_u_rad_avg_recv_lst[-1]))
        
            bnds_w_rad_avg_send_lst.append(deepcopy(bnds_w_rad_avg_send_lst[-1]))
            bnds_w_rad_avg_recv_lst.append(deepcopy(bnds_w_rad_avg_recv_lst[-1]))


    if param_dict['bnd_zl'] == 'radiation':

        gb_sl_z_down_indtpl_lst.append([(0, ng), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_mid_indtpl_lst.append([(1, ng + 1), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_urad_mid_indtpl_lst.append([(ng, ng + 1), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_up_indtpl_lst.append([(2, ng + 2), (ng, nr_sub + ng), (ng, nc_sub + ng)])

        u_rad_lst.append(np.zeros([ng, nr_sub, nc_sub + 1]))
        v_rad_lst.append(np.zeros([ng, nr_sub + 1, nc_sub]))
        w_rad_lst.append(np.zeros([ng, nr_sub, nc_sub]))

        u_prefac.append(-1.0 / dz[0])

        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 'b':
                bnd_obj_inds.append(n)

    if param_dict['bnd_zr'] == 'radiation':

        gb_sl_z_up_indtpl_lst.append([(-ng - 2, -2), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_mid_indtpl_lst.append([(-ng - 1, -1), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_urad_mid_indtpl_lst.append([(-ng - 1, -ng), (ng, nr_sub + ng), (ng, nc_sub + ng)])
        gb_sl_z_down_indtpl_lst.append([(-ng, None), (ng, nr_sub + ng), (ng, nc_sub + ng)])


        u_rad_lst.append(np.zeros([ng, nr_sub, nc_sub + 1]))
        v_rad_lst.append(np.zeros([ng, nr_sub + 1, nc_sub]))
        w_rad_lst.append(np.zeros([ng, nr_sub, nc_sub]))

        u_prefac.append(1.0 / dz[-1])

        for n, bnd in enumerate(bnds_global_m):
            if bnd.side == 't':
                bnd_obj_inds.append(n)

    if any((param_dict['bnd_zl'] == 'radiation', param_dict['bnd_zr'] == 'radiation')):

        bnds_send = []
        bnds_recv = []

        inds = inds = list(np.array([(k, i, j) for k in range(0, ng) for i in range(0, nr_sub) for j in range(0, 1)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pc > 0:
            pid_s = ind_pr * npc + ind_pc - 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 351000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 381000 + tag))

        elif param_dict['bnd_xl'] == 'cyclic':
            pid_s = ind_pr * npc + npc - 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 351000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'w', tag= 381000 + tag))

        inds = inds = list(np.array([(k, i, j) for k in range(0, ng) for i in range(0, nr_sub) for j in range(nc_sub, nc_sub + 1)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pc < npc - 1:
            pid_s = ind_pr * npc + ind_pc + 1
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 381000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 351000 + tag))

        elif param_dict['bnd_xl'] == 'cyclic':
            pid_s = ind_pr * npc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 381000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'e', tag= 351000 + tag))

        bnds_u_rad_avg_send_lst.append(bnds_send)
        bnds_u_rad_avg_recv_lst.append(bnds_recv)

        if all((param_dict['bnd_zl'] == 'radiation', param_dict['bnd_zr'] == 'radiation')):
            bnds_u_rad_avg_send_lst.append(bnds_send)
            bnds_u_rad_avg_recv_lst.append(bnds_recv)

        bnds_send = []
        bnds_recv = []

        inds = inds = list(np.array([(k, i, j) for k in range(0, ng) for i in range(0, 1) for j in range(0, nc_sub)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pr > 0:
            pid_s = (ind_pr - 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 's', tag= 411000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 's', tag= 441000 + tag))

        elif param_dict['bnd_yl'] == 'cyclic':
            pid_s = (npr - 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 's', tag= 411000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 's', tag= 441000 + tag))

        inds = inds = list(np.array([(k, i, j) for k in range(0, ng) for i in range(nr_sub, nr_sub + 1) for j in range(0, nc_sub)]).T)
        fld = np.empty([len(inds[0])])
        if ind_pr < npr - 1:
            pid_s = (ind_pr + 1) * npc + ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 441000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 411000 + tag))

        elif param_dict['bnd_yl'] == 'cyclic':
            pid_s = ind_pc
            tag = rank * pid_s
            bnds_send.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 441000 + tag))
            bnds_recv.append(ddcp.bound(fld, inds, pid_s, 'n', tag= 411000 + tag))

        bnds_v_rad_avg_send_lst.append(bnds_send)
        bnds_v_rad_avg_recv_lst.append(bnds_recv)

        if all((param_dict['bnd_zl'] == 'radiation', param_dict['bnd_zr'] == 'radiation')):
            bnds_v_rad_avg_send_lst.append(bnds_send)
            bnds_v_rad_avg_recv_lst.append(bnds_recv)
    
    return bnds_global_lst



def read_bnd(mpicomm, pids, param_dict, bnd_files, file_nr):
    """
    Reads in the next boundary file
    and distributes the data 
    
    mpicomm... mpi communicator
    pids... list of subdomain process ids
    param_dict... parameter dictionary
    bnd_files... list of boundary file names
    file_nr... number of boundary  file to open next
    """

    rank = mpicomm.Get_rank()

    global bnds_global_lst
    global bnds_global_u_s
    global bnds_global_u_r
    global bnds_global_v_s
    global bnds_global_v_r
    global bnds_global_w_s
    global bnds_global_w_r
    global bnds_global_c_s
    global bnds_global_c_r
    global bnds_global_diag_c_s
    global bnds_global_diag_c_r
    global bnds_global_chem_r
    global bnds_global_chem_s

    if rank == 0:
        # check for file availability and abort in case  
        if file_nr >= len(bnd_files):
            print "Simulation aboarded due to missing next boundary time!"
            mpicomm.Abort()
        netcdf_file = Dataset('./INPUT/' + bnd_files[file_nr], 'r')

        #thermodynamics
        theta = netcdf_file.variables['Theta'][0]
        QV = netcdf_file.variables['QV'][0]
        thetav = theta * (1.0 + 0.61 * QV)

        # time
        tim = netcdf_file.variables['time'][0]
        t_units = netcdf_file.variables['time'].units
        param_dict.update({'t_units_ref':t_units})
        time = np.array([1], dtype=int)
        time[0] = tim
        buffs = [time for pid in pids[1:]]
        ddcp.scatter_point(mpicomm, buffs, pids[1:], wait=True)

    else:
        time = np.zeros([1], dtype=int)
        req = mpicomm.Irecv(time, source = 0)
        req.wait()

    if rank == 0:
        #U-wind
        U = netcdf_file.variables['U'][0]
        bnd_fld = U.copy()
        ddcp.cptobounds(bnd_fld, bnds_global_u_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_u_s, bnds_global_u_r)
    for k, bnd in enumerate(bnds_global_u_r):
        bnds_global_lst[0][k].update(bnd.data)
    for bnd in bnds_global_lst[0]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #V-wind
        V = netcdf_file.variables['V'][0]
        bnd_fld = V.copy()
        ddcp.cptobounds(bnd_fld, bnds_global_v_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_v_s, bnds_global_v_r)
    for k, bnd in enumerate(bnds_global_v_r):
        bnds_global_lst[1][k].update(bnd.data)
    for bnd in bnds_global_lst[1]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #W-wind
        bnd_fld = netcdf_file.variables['W'][0]
        ddcp.cptobounds(bnd_fld, bnds_global_w_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_w_s, bnds_global_w_r)
    for k, bnd in enumerate(bnds_global_w_r):
        bnds_global_lst[2][k].update(bnd.data)
    for bnd in bnds_global_lst[2]:
        bnd.update_bndtime(time[0])

    nb = 3

    for bnd in bnds_global_lst[3]:
        bnd.update_bndtime(time[0])
    nb += 1

    if rank == 0:
        #air density
        bnd_fld = netcdf_file.variables['Rho'][0]
        ddcp.cptobounds(bnd_fld, bnds_global_c_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_c_s, bnds_global_c_r)
    for k, bnd in enumerate(bnds_global_c_r):
        bnds_global_lst[nb][k].update(bnd.data)
    for bnd in bnds_global_lst[nb]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        # virtual potential temperature
        bnd_fld = thetav
        ddcp.cptobounds(bnd_fld, bnds_global_c_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_c_s, bnds_global_c_r)
    for k, bnd in enumerate(bnds_global_c_r):
        bnds_global_lst[nb + 1][k].update(bnd.data)
    for bnd in bnds_global_lst[nb + 1]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #water vapour mixing ratio
        bnd_fld = QV
        ddcp.cptobounds(bnd_fld, bnds_global_c_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_c_s, bnds_global_c_r)
    for k, bnd in enumerate(bnds_global_c_r):
        bnds_global_lst[nb + 2][k].update(bnd.data)
    for bnd in bnds_global_lst[nb + 2]:
        bnd.update_bndtime(time[0])

    tr_names = param_dict['tracers']

    for n, name in enumerate(tr_names):
        # Tracers
        if any([param_dict['bnd_chem_xl'] == 'dirichlet', param_dict['bnd_chem_yl'] == 'dirichlet']):            
            if rank == 0:
                bnd_fld = netcdf_file.variables[name][0]
                ddcp.cptobounds(bnd_fld, bnds_global_chem_s, mode='repl')
            ddcp.exchange_fields(mpicomm, bnds_global_chem_s, bnds_global_chem_r)

            for k, bnd in enumerate(bnds_global_chem_r):
                bnds_global_lst[nb + 3 + n][k].update(bnd.data)
            for bnd in bnds_global_lst[nb + 3 + n]:
                 bnd.update_bndtime(time[0])
        else:
            for k, bnd in enumerate(bnds_global_chem_r):
                bnds_global_lst[nb + 3 + n][k].update(np.zeros_like(bnd.data))
            for bnd in bnds_global_lst[nb + 3 + n]:
                 bnd.update_bndtime(time[0])

    if rank == 0:
        #surface potential temperature
        THS = netcdf_file.variables['Th_S'][0]                
        bnd_fld = THS
        ddcp.cptobounds(bnd_fld, bnds_global_diag_c_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_diag_c_s, bnds_global_diag_c_r)

    for k, bnd in enumerate(bnds_global_diag_c_r):
        bnds_global_lst[-5][k].update(bnd.data)
    for bnd in bnds_global_lst[-5]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #surface specific humidity
        QVS = netcdf_file.variables['QV_S'][0]
        bnd_fld = QVS
        ddcp.cptobounds(bnd_fld, bnds_global_diag_c_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_diag_c_s, bnds_global_diag_c_r)

    for k, bnd in enumerate(bnds_global_diag_c_r):
        bnds_global_lst[-4][k].update(bnd.data)
    for bnd in bnds_global_lst[-4]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #U-wind rms (turbulent intensity)
        U_rms = netcdf_file.variables['U_rms'][0]
        bnd_fld = U_rms.copy()
        ddcp.cptobounds(bnd_fld, bnds_global_u_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_u_s, bnds_global_u_r)

    for k, bnd in enumerate(bnds_global_u_r):
        bnds_global_lst[-3][k].update(bnd.data)
    for bnd in bnds_global_lst[-3]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #V-wind rms (turbulent intensity)
        V_rms = netcdf_file.variables['V_rms'][0]
        bnd_fld = V_rms.copy()
        ddcp.cptobounds(bnd_fld, bnds_global_v_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_v_s, bnds_global_v_r)

    for k, bnd in enumerate(bnds_global_v_r):
        bnds_global_lst[-2][k].update(bnd.data)
    for bnd in bnds_global_lst[-2]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        #W-wind rms (turbulent intensity)
        W_rms = netcdf_file.variables['W_rms'][0]
        bnd_fld = W_rms.copy()
        ddcp.cptobounds(bnd_fld, bnds_global_w_s, mode='repl')
    ddcp.exchange_fields(mpicomm, bnds_global_w_s, bnds_global_w_r)

    for k, bnd in enumerate(bnds_global_w_r):
        bnds_global_lst[-1][k].update(bnd.data)
    for bnd in bnds_global_lst[-1]:
        bnd.update_bndtime(time[0])

    if rank == 0:
        netcdf_file.close()


def set_bnds(comm, param_dict, flds, fld_tps, time, bnd_files):
    """
    Sets the lateral boundaries for the
    first time at simulation initialization.

    comm... ddcp.communicator
    param_dict... parameter dictionary
    flds... list of prognostic fields
    fld_tps... list of prognostic field types
    time... start time of simulation
    bnd_files... list of boundary file names
    """

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids

    global bnds_global_lst

    global bnd_obj_inds_u
    global bnd_obj_inds_v

    global bnd_times
    global file_counter

    if len(bnds_global_lst[0]):        
        t_next = bnds_global_lst[0][0].t_next

    if time == int(param_dict['st_time']):
        read_bnd(mpicomm, pids, param_dict, bnd_files, 0)
        read_bnd(mpicomm, pids, param_dict, bnd_files, 1)
    file_counter = 2

    for n, fld in enumerate(flds[:-3]):

       for bnd in bnds_global_lst[n]:
            bnd.interpolate(time)
       ddcp.cpfrombounds(fld, bnds_global_lst[n], mode='repl')
       update_halo_bnds(comm.mpicomm, fld, type=fld_tps[n])
    

def update_bnd_objs(comm, param_dict, dt, time, bnd_files):
    """
    Load the next boundary value and update the boundary  object to store
    the data.
    
    comm... ddcp.communicator
    param_dict... parameter dictionary
    dt... time increment
    time... model  time
    bnd_files... list of boundary file names
    """

    mpicomm = comm.mpicomm
    pids = comm.pids    

    global bnd_times
    global file_counter

    if len(bnds_global_lst[-4]):
        t_next = bnds_global_lst[-4][0].t_next        
    else:
        return None

    if time > t_next:
        read_bnd(mpicomm, pids, param_dict, bnd_files, file_counter)
        file_counter += 1


def update_scalar_bnds(comm, flds, time):
    """
    Updates the lateral boundary values for 
    all scalar fields.

    comm... ddcp.communicator
    flds... list of prognostic fields
    time... simulation time
    """

    global bnds_global_lst

    nb = 3

    for fld in flds[3:-3]:
        for bnd in bnds_global_lst[nb]:
            bnd.interpolate(time)            
        ddcp.cpfrombounds(fld, bnds_global_lst[nb], mode='repl')
        nb += 1

    nb = 4
    for fld in flds[4:-3]:
        update_halo_bnds(comm.mpicomm, fld, type='c')
        nb += 1

def update_vel(comm, param_dict, vel_flds, vel_flds_prev, vel_rms, rho, surf_flds, dt, time):
    """
    This scheme updates the lateral boundary values of the velocity components.
    If the boundary condition is set to "radiation", inflow
    and outflow areas are determined dynamically using an approximate convective velocity.
    For the inflow areas, values are prescribed from the
    external boundary fields with optional superposition of turbulent fluctuations.
    For the outlflow areas, the velocity components are updated using
    the prognostic convection equation with upwind discretization (Miller and Thorpe, 1981).
    The total mass-flux integrated over the area where the radiation condition applies
    is corrected to balance the prescribed Dirichlet mass flux in order 
    to satisfy the integrability criterion of the pressure solver. 

    comm... ddcp.communicator
    param_dict... parameter dictionary
    vel_flds... list of velocity fields
    vel_flds_prev... list of velocity fields at previous simulation time step
    vel_rms... target root mean square of velocity fluctuations
    dt... current integration time increment
    time... simulation time
    """

    global bnds_global_lst
    global comm_bnd

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids
    ind_p = pids.index(rank)
    ind_pr = ind_p / npc
    ind_pc = ind_p - ind_pr * npc 

    speed_rad_lst, rad_true_lst, rad_false_lst, dvel_dn_lst = u_radiation(comm, vel_flds[:3], vel_flds_prev, surf_flds, dt)

    rec_bnd_lst = [[], [], []]

    if int(param_dict['rec_turb']):
        u_rms, v_rms, w_rms = vel_rms[0], vel_rms[1], vel_rms[2]
        u, v, w = vel_flds[0], vel_flds[1], vel_flds[2]
        rec_bnd_lst[:] = recycle_turb(comm, u, v, w, u_rms, v_rms, w_rms, param_dict)
    
    for n, fld in enumerate(vel_flds[:3]):

        #interpolate new Dirichlet values on the boundary objects
        for bnd in bnds_global_lst[n]:
             bnd.interpolate(time)

        #impose optional turbulence using the recycling scheme
        for k, rec_bnd in enumerate(rec_bnd_lst[n]):
            bnds_global_lst[n][bnd_obj_ind_recturb[k]].data += rec_bnd.data
        #Overwrite data with zeros over area, where radiation condition applies instead of Dirichlet 
        for k, rad_false in enumerate(rad_false_lst[n]):
            bnds_global_lst[n][bnd_obj_inds[k]].data[:] = bnds_global_lst[n][bnd_obj_inds[k]].data * rad_false.flatten()
    
    #Update the velocity ghost cells with the new Dirichlet values
    for n, fld in enumerate(vel_flds[:3]):
       for bnd in bnds_global_lst[n]:
           bnd.data[:] -=fld[bnd.inds]
       for k, rad_false in enumerate(rad_false_lst[n]):
           bnds_global_lst[n][bnd_obj_inds[k]].data[:] = bnds_global_lst[n][bnd_obj_inds[k]].data * rad_false.flatten()

       ddcp.cpfrombounds(fld, bnds_global_lst[n], mode='add')

    #apply outflow radiation condition for the rest of the area
    for n, fld in enumerate(vel_flds[:3]):
       for bnd in bnds_global_lst[n]:
           bnd.data.fill(0.0)
    for n, fld in enumerate(vel_flds[:3]):      
       for k, speed_rad in enumerate(speed_rad_lst[n]): 
           bnds_global_lst[n][bnd_obj_inds[k]].data[:] = fld[bnds_global_lst[n][bnd_obj_inds[k]].inds] * rad_true_lst[n][k].flatten()
           bnds_global_lst[n][bnd_obj_inds[k]].data[:] -= (speed_rad * dvel_dn_lst[n][k]).flatten() 
               

    #Update the remaining velocity ghost cells with the new prognostic values from the radiation condition
    for n, fld in enumerate(vel_flds[:3]):
       for k, rad_true in enumerate(rad_true_lst[n]):
           bnds_global_lst[n][bnd_obj_inds[k]].data[:] -= fld[bnds_global_lst[n][bnd_obj_inds[k]].inds]
           bnds_global_lst[n][bnd_obj_inds[k]].data[:] = bnds_global_lst[n][bnd_obj_inds[k]].data * rad_true.flatten()
       ddcp.cpfrombounds(fld, bnds_global_lst[n], mode='add')
  

    # mass correction scheme to satisfy global mass conservation
#    bnd_flux_correct_3d(comm, vel_flds[:3], rho)
    bnd_flux_correct_dimsplit(comm, vel_flds[:3], rho)

    update_halo_bnds(comm.mpicomm, vel_flds[0], type='u')
    update_halo_bnds(comm.mpicomm, vel_flds[1], type='v')
    update_halo_bnds(comm.mpicomm, vel_flds[2], type='w')



def update_halo_bnds(mpicomm, field, type='c'):
    """
    Updates the missing edges and corners (halo)
    of the lateral boundaries.

    mpicomm... MPI communicator
    field... prognostic field to update
    type... field type ('c' cell centred, 'u', 'v', 'w' cell-face centred)
    """

    bnds_global_halo_s = globals()['bnds_global_halo_' + type + '_s']
    bnds_global_halo_r = globals()['bnds_global_halo_' + type + '_r']

    ddcp.cptobounds(field, bnds_global_halo_s)
    ddcp.exchange_fields(mpicomm, bnds_global_halo_s, bnds_global_halo_r)
    ddcp.cpfrombounds(field, bnds_global_halo_r, mode = 'repl')


def u_radiation(comm, vel_flds, vel_flds_prev, surf_flds, dt):
    """
    This routine derives the discretized
    terms for the radiation boundary condition using
    the upwind discretization for the gradient
    (Miller and Thorpe, 1981).
    As the transport velocity, it is taken the boundary-
    normal velocity component.

    comm... ddcp.communicator
    vel_flds... velocity components
    vel_flds_prev... velocity components at previous model time step
    dt... current integration time increment
    """

    mpicomm = comm.mpicomm


    global gb_sl_x_up_indtpl_lst
    global gb_sl_y_up_indtpl_lst
    global gb_sl_z_up_indtpl_lst

    global gb_sl_x_mid_indtpl_lst
    global gb_sl_y_mid_indtpl_lst
    global gb_sl_z_mid_indtpl_lst

    global gb_sl_x_urad_mid_indtpl_lst
    global gb_sl_y_urad_mid_indtpl_lst
    global gb_sl_z_urad_mid_indtpl_lst

    global gb_sl_x_down_indtpl_lst
    global gb_sl_y_down_indtpl_lst
    global gb_sl_z_down_indtpl_lst

    global u_rad_lst, v_rad_lst, w_rad_lst

    global dhsurfdx_lst, dhsurfdy_lst

    u, v, w = vel_flds[:]
    u_prev, v_prev, w_prev = vel_flds_prev[:]
  
    dhsurfdx, dhsurfdy = surf_flds[:]

    u_rad_tmp_lst = []
    v_rad_tmp_lst = []
    w_rad_tmp_lst = []

    u_rad_true_lst = []
    v_rad_true_lst = []
    w_rad_true_lst = []

    u_rad_false_lst = []
    v_rad_false_lst = []
    w_rad_false_lst = []

    du_dn_lst = []
    dv_dn_lst = []
    dw_dn_lst = []
    

    for n, obj in enumerate(gb_sl_x_mid_indtpl_lst):
      
        eps = 1e-40
        k_inds_up, i_inds_up, j_inds_up = gb_sl_x_up_indtpl_lst[n][:]
        k_inds_mid, i_inds_mid, j_inds_mid = gb_sl_x_mid_indtpl_lst[n][:]
        k_inds_urad_mid, i_inds_urad_mid, j_inds_urad_mid = gb_sl_x_urad_mid_indtpl_lst[n][:]
        k_inds_down, i_inds_down, j_inds_down = gb_sl_x_down_indtpl_lst[n][:]

        u_sl_rad_mid = (u[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        v_sl_rad_mid = (v[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1] + 1, j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        w_sl_rad_mid = (w[k_inds_urad_mid[0]:k_inds_urad_mid[1] + 1, i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        
        u_sl_mid_prev = u_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1]]
        v_sl_mid_prev = v_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1] + 1, j_inds_mid[0]:j_inds_mid[1]]
        w_sl_mid_prev = w_prev[k_inds_mid[0]:k_inds_mid[1] + 1 , i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1]]

        u_sl_down_prev = u_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1]]
        v_sl_down_prev = v_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1] + 1, j_inds_down[0]:j_inds_down[1]]
        w_sl_down_prev = w_prev[k_inds_down[0]:k_inds_down[1] + 1, i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1]]

        u_rad_lst[n][:] = np.minimum(np.maximum(u_prefac[n] * u_sl_rad_mid * dt, 0.0), 1.0)
        
        v_rad_lst[n][:, 1:-1] = 0.5 * (u_rad_lst[n][:, 1:] + u_rad_lst[n][:, :-1])
        v_rad_lst[n][:, 0] = u_rad_lst[n][:, 0]
        v_rad_lst[n][:, -1] = u_rad_lst[n][:, -1]
      
        ddcp.cptobounds(0.5 * v_rad_lst[n][:], bnds_v_rad_avg_send_lst[n], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_v_rad_avg_send_lst[n], bnds_v_rad_avg_recv_lst[n])
        ddcp.cptobounds(0.5 * v_rad_lst[n][:], bnds_v_rad_avg_recv_lst[n], mode = 'sub')
        ddcp.cpfrombounds(v_rad_lst[n][:], bnds_v_rad_avg_recv_lst[n], mode = 'add')

        w_rad_lst[n][1:-1] = 0.5 * (u_rad_lst[n][1:] + u_rad_lst[n][:-1])
        w_rad_lst[n][0] = u_rad_lst[n][0]
        w_rad_lst[n][-1] = u_rad_lst[n][-1]
        
        ddcp.cptobounds(0.5 * w_rad_lst[n][:], bnds_w_rad_avg_send_lst[n], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_w_rad_avg_send_lst[n], bnds_w_rad_avg_recv_lst[n])
        ddcp.cptobounds(0.5 * w_rad_lst[n][:], bnds_w_rad_avg_recv_lst[n], mode = 'sub')
        ddcp.cpfrombounds(w_rad_lst[n][:], bnds_w_rad_avg_recv_lst[n], mode = 'add')

        u_rad_false_lst.append(np.logical_not(u_rad_lst[n]))
        v_rad_false_lst.append(np.logical_not(v_rad_lst[n]))
        w_rad_false_lst.append(np.logical_not(w_rad_lst[n]))

        u_rad_true_lst.append(np.array(u_rad_lst[n], dtype=bool))
        v_rad_true_lst.append(np.array(v_rad_lst[n], dtype=bool))
        w_rad_true_lst.append(np.array(w_rad_lst[n], dtype=bool))

        du_dn = u_sl_down_prev - u_sl_mid_prev
        dv_dn = v_sl_down_prev - v_sl_mid_prev
        dw_dn = w_sl_down_prev - w_sl_mid_prev

        du_dn_lst.append(du_dn)
        dv_dn_lst.append(dv_dn)
        dw_dn_lst.append(dw_dn)

    for n, obj in enumerate(gb_sl_y_mid_indtpl_lst):

        eps = 1e-40
        m = n + len(gb_sl_x_mid_indtpl_lst)

        k_inds_up, i_inds_up, j_inds_up = gb_sl_y_up_indtpl_lst[n][:]
        k_inds_mid, i_inds_mid, j_inds_mid = gb_sl_y_mid_indtpl_lst[n][:]
        k_inds_urad_mid, i_inds_urad_mid, j_inds_urad_mid = gb_sl_y_urad_mid_indtpl_lst[n][:]
        k_inds_down, i_inds_down, j_inds_down = gb_sl_y_down_indtpl_lst[n][:]

        u_sl_rad_mid = (u[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1] + 1]).copy()
        v_sl_rad_mid = (v[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        w_sl_rad_mid = (w[k_inds_urad_mid[0]:k_inds_urad_mid[1] + 1, i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()

        u_sl_mid_prev = u_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1] + 1]
        v_sl_mid_prev = v_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1]]
        w_sl_mid_prev = w_prev[k_inds_mid[0]:k_inds_mid[1] + 1, i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1]]

        u_sl_down_prev = u_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1] + 1]
        v_sl_down_prev = v_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1]]
        w_sl_down_prev = w_prev[k_inds_down[0]:k_inds_down[1] + 1, i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1]]

        v_rad_lst[m][:] = np.minimum(np.maximum(u_prefac[m] * v_sl_rad_mid * dt, 0.0), 1.0)

        u_rad_lst[m][:, :, 1:-1] = 0.5 * (v_rad_lst[m][:, :, 1:] + v_rad_lst[m][:, :, :-1])
        u_rad_lst[m][:, :, 0] = v_rad_lst[m][:, :, 0]
        u_rad_lst[m][:, :, -1] = v_rad_lst[m][:, :, -1]
      
        ddcp.cptobounds(0.5 * u_rad_lst[m][:], bnds_u_rad_avg_send_lst[n], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_u_rad_avg_send_lst[n], bnds_u_rad_avg_recv_lst[n])
        ddcp.cptobounds(0.5 * u_rad_lst[m][:], bnds_u_rad_avg_recv_lst[n], mode = 'sub')
        ddcp.cpfrombounds(u_rad_lst[m][:], bnds_u_rad_avg_recv_lst[n], mode = 'add')

        w_rad_lst[m][1:-1] = 0.5 * (v_rad_lst[m][1:] + v_rad_lst[m][:-1])
        w_rad_lst[m][0] = v_rad_lst[m][0]
        w_rad_lst[m][-1] = v_rad_lst[m][-1]
     
        ddcp.cptobounds(0.5 * w_rad_lst[m][:], bnds_w_rad_avg_send_lst[m], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_w_rad_avg_send_lst[m], bnds_w_rad_avg_recv_lst[m])
        ddcp.cptobounds(0.5 * w_rad_lst[m][:], bnds_w_rad_avg_recv_lst[m], mode = 'sub')
        ddcp.cpfrombounds(w_rad_lst[m][:], bnds_w_rad_avg_recv_lst[m], mode = 'add')

        u_rad_false_lst.append(np.logical_not(u_rad_lst[m]))
        v_rad_false_lst.append(np.logical_not(v_rad_lst[m]))
        w_rad_false_lst.append(np.logical_not(w_rad_lst[m]))

        u_rad_true_lst.append(np.array(u_rad_lst[m], dtype=bool))
        v_rad_true_lst.append(np.array(v_rad_lst[m], dtype=bool))
        w_rad_true_lst.append(np.array(w_rad_lst[m], dtype=bool))


        du_dn = u_sl_down_prev - u_sl_mid_prev
        dv_dn = v_sl_down_prev - v_sl_mid_prev
        dw_dn = w_sl_down_prev - w_sl_mid_prev

        du_dn_lst.append(du_dn)
        dv_dn_lst.append(dv_dn)
        dw_dn_lst.append(dw_dn)

    for n, obj in enumerate(gb_sl_z_mid_indtpl_lst):
        
        m = n + len(gb_sl_x_mid_indtpl_lst)
        o = n + len(gb_sl_y_mid_indtpl_lst)
        p = n + len(gb_sl_x_mid_indtpl_lst) + len(gb_sl_y_mid_indtpl_lst) 

        k_inds_up, i_inds_up, j_inds_up = gb_sl_z_up_indtpl_lst[n][:]
        k_inds_urad_mid, i_inds_urad_mid, j_inds_urad_mid = gb_sl_z_urad_mid_indtpl_lst[n][:]
        k_inds_mid, i_inds_mid, j_inds_mid = gb_sl_z_mid_indtpl_lst[n][:]
        k_inds_down, i_inds_down, j_inds_down = gb_sl_z_down_indtpl_lst[n][:]

        w_sl_mid = (w[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        u_sl_mid = (u[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1] + 1]).copy()
        v_sl_mid = (v[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1] + 1, j_inds_urad_mid[0]:j_inds_urad_mid[1]]).copy()
        dhsurfdx_mid = dhsurfdx[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1], j_inds_urad_mid[0]:j_inds_urad_mid[1] + 1]
        dhsurfdy_mid = dhsurfdy[k_inds_urad_mid[0]:k_inds_urad_mid[1], i_inds_urad_mid[0]:i_inds_urad_mid[1] + 1, j_inds_urad_mid[0]:j_inds_urad_mid[1]]

        dhsrfdxu_sl_mid_z = dhsurfdx_mid * u_sl_mid
        dhsrfdxv_sl_mid_z = dhsurfdy_mid * v_sl_mid
        
        w_sl_mid -= 0.5 * (dhsrfdxu_sl_mid_z[:, :, 1:] + dhsrfdxu_sl_mid_z[:, :, :-1])
        w_sl_mid -= 0.5 * (dhsrfdxv_sl_mid_z[:, 1:] + dhsrfdxv_sl_mid_z[:, :-1])

        u_sl_mid_prev = u_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1] + 1]
        v_sl_mid_prev = v_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1] + 1, j_inds_mid[0]:j_inds_mid[1]]
        w_sl_mid_prev = w_prev[k_inds_mid[0]:k_inds_mid[1], i_inds_mid[0]:i_inds_mid[1], j_inds_mid[0]:j_inds_mid[1]]

        u_sl_down_prev = u_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1] + 1]
        v_sl_down_prev = v_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1] + 1, j_inds_down[0]:j_inds_down[1]]
        w_sl_down_prev = w_prev[k_inds_down[0]:k_inds_down[1], i_inds_down[0]:i_inds_down[1], j_inds_down[0]:j_inds_down[1]]

        w_rad_lst[p][:] = np.minimum(np.maximum(u_prefac[m] * w_sl_mid * dt, 0.0), 1.0)        

        u_rad_lst[p][:, :, 1:-1] = 0.5 * (w_rad_lst[p][:, :, 1:] + w_rad_lst[p][:, :, :-1])
        u_rad_lst[p][:, :, 0] = w_rad_lst[p][:, :, 0]
        u_rad_lst[p][:, :, -1] = w_rad_lst[p][:, :, -1]       

        ddcp.cptobounds(0.5 * u_rad_lst[p][:], bnds_u_rad_avg_send_lst[o], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_u_rad_avg_send_lst[o], bnds_u_rad_avg_recv_lst[o])
        ddcp.cptobounds(0.5 * u_rad_lst[p][:], bnds_u_rad_avg_recv_lst[o], mode = 'sub')
        ddcp.cpfrombounds(u_rad_lst[p][:], bnds_u_rad_avg_recv_lst[o], mode = 'add')

        v_rad_lst[p][:, 1:-1] = 0.5 * (w_rad_lst[p][:, 1:] + w_rad_lst[p][:, :-1])
        v_rad_lst[p][:, 0] = w_rad_lst[p][:, 0]
        v_rad_lst[p][:, -1] = w_rad_lst[p][:, -1]

        ddcp.cptobounds(0.5 * v_rad_lst[p][:], bnds_v_rad_avg_send_lst[m], mode = 'repl')
        ddcp.exchange_fields(mpicomm, bnds_v_rad_avg_send_lst[m], bnds_v_rad_avg_recv_lst[m])
        ddcp.cptobounds(0.5 * v_rad_lst[p][:], bnds_v_rad_avg_recv_lst[m], mode = 'sub')
        ddcp.cpfrombounds(v_rad_lst[p][:], bnds_v_rad_avg_recv_lst[m], mode = 'add')

        u_rad_false_lst.append(np.logical_not(u_rad_lst[p]))
        v_rad_false_lst.append(np.logical_not(v_rad_lst[p]))
        w_rad_false_lst.append(np.logical_not(w_rad_lst[p]))

        u_rad_true_lst.append(np.array(u_rad_lst[p], dtype=bool))
        v_rad_true_lst.append(np.array(v_rad_lst[p], dtype=bool))
        w_rad_true_lst.append(np.array(w_rad_lst[p], dtype=bool))

        du_dn = u_sl_down_prev - u_sl_mid_prev
        dv_dn = v_sl_down_prev - v_sl_mid_prev
        dw_dn = w_sl_down_prev - w_sl_mid_prev

        du_dn_lst.append(du_dn)
        dv_dn_lst.append(dv_dn)
        dw_dn_lst.append(dw_dn)

    dvel_dn = [du_dn_lst, dv_dn_lst, dw_dn_lst]

    return [u_rad_lst, v_rad_lst, w_rad_lst], [u_rad_true_lst, v_rad_true_lst, w_rad_true_lst],  [u_rad_false_lst, v_rad_false_lst, w_rad_false_lst],  dvel_dn


def init_neumann_bc(comm, param_dict):
    """
    Initializes lateral Neumann boundary conditions for
    arbitrary prognostic fields.

    comm... communicator
    param_dict... parameter dictionary
    """

    global ng
    global bnd_neumann_transfer

    ng1 = ng - 1

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    npr = comm.npr
    npc = comm.npc
    pids = comm.pids
    ind_p = pids.index(rank)
    ind_pr = ind_p / npc
    ind_pc = ind_p - ind_pr * npc

    bnd_xl = param_dict['bnd_xl']
    bnd_yl = param_dict['bnd_yl']
    bnd_zl = param_dict['bnd_zl']

    bnd_neumann_transfer = []

    if ind_pc == 0 and not bnd_xl == 'cyclic':
        slice_from = [(0, None), (0, None), (ng, ng + 1)]
        slice_to = [(0, None), (0, None), (0, ng)]
        bnd_neumann_transfer.append([slice_from, slice_to])

    if ind_pc == npc - 1 and not bnd_xl == 'cyclic':
        slice_from = [(0, None), (0, None), (-(ng + 1), -ng)]
        slice_to = [(0, None), (0, None), (-ng, None)]
        bnd_neumann_transfer.append([slice_from, slice_to])

    if ind_pr == 0 and not bnd_yl == 'cyclic':
        slice_from = [(0, None), (ng, ng + 1), (0, None)]
        slice_to = [(0, None), (0, ng), (0, None)]
        bnd_neumann_transfer.append([slice_from, slice_to])

    if ind_pr == npr - 1 and not bnd_yl == 'cyclic':
        slice_from = [(0, None), (-(ng + 1), -ng), (0, None)]
        slice_to = [(0, None), (-ng, None), (0, None)]
        bnd_neumann_transfer.append([slice_from, slice_to])

    slice_from = [(ng, ng + 1), (0, None), (0, None)]
    slice_to = [(0, ng), (0, None), (0, None)]
    bnd_neumann_transfer.append([slice_from, slice_to])

    slice_from = [(-(ng + 1), -ng), (0, None), (0, None)]
    slice_to = [(-ng, None), (0, None), (0, None)]
    bnd_neumann_transfer.append([slice_from, slice_to])


def impose_lateral_neumann_bc(field):
    """
    Impose lateral Neumann boundary conditions on
    a prognostic field.

    field... arbitrary prognostic field
    """

    for transfer in bnd_neumann_transfer:
        sl_f, sl_t = transfer[:]
        field_from = field[sl_f[0][0]:sl_f[0][1], sl_f[1][0]:sl_f[1][1], sl_f[2][0]:sl_f[2][1]]
        field[sl_t[0][0]:sl_t[0][1], sl_t[1][0]:sl_t[1][1], sl_t[2][0]:sl_t[2][1]] = field_from


def init_bnd_flux_corr(comm, area_eff_x, area_eff_y, area_eff_z, dhsurfdx, dhsurfdy, param_dict):
    """
    Initializes the  mass flux correction scheme 
    for the velocity boundary fields.
    The flux correction uses dimensional splitting, 
    i.e. each one-dimensional net flux is balanced to zero.
    The boundary objects with the perpendicular velocity 
    component and density are collected in new lists. 
    The parts of the integration surface are adressed with 
    fancy indexing, whose index lists are also created here. 
    For the integration of the mass flux density, communications are necessary, 
    and the processor ids sharing boundary parts in each separate dimension are stored in
    new comm objects.

    comm... ddcp.communicator
    area_eff_x, area_eff_y, area_eff_z... effective cell-face areas
    dhsurfdx, dhsurfdy... spatial derivatives of terrain function
    param_dict... parameter dictionary 
    """

    global bnds_global_vel_perpendicular_x
    global bnds_global_vel_perpendicular_y
    global bnds_global_vel_perpendicular_z
    global bnds_global_rho_x
    global bnds_global_rho_y
    global bnds_global_rho_z
    global bnds_global_vel_u_z
    global bnds_global_vel_v_z
 
    global bnd_area_x
    global bnd_area_y
    global bnd_area_z

    global dhsurfdx_lst
    global dhsurfdy_lst

    global bnds_global_lst
    global ng
 
    ng1 = ng - 1

    bnds_global_vel_perpendicular_x = []
    bnds_global_vel_perpendicular_y = []
    bnds_global_vel_perpendicular_z = []
    bnds_global_vel_u_z = []
    bnds_global_vel_v_z = []
    bnds_global_rho_x = []
    bnds_global_rho_y = []
    bnds_global_rho_z = []

    bnd_area_x = []
    bnd_area_y = []
    bnd_area_z = []

    dhsurfdx_lst = []
    dhsurfdy_lst = []

    pids = comm.pids
    npc = comm.npc
    npr = comm.npr
    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()   
    ind_p = pids.index(rank)
    ind_pr = ind_p / npc
    ind_pc = ind_p - ind_pr * npc  

    bnd_xl = param_dict['bnd_xl']  
    bnd_yl = param_dict['bnd_yl']  
    bnd_zl = param_dict['bnd_zl'] 

    nz_sub, nr_sub, nc_sub = area_eff_x.shape[:]
    nc_sub -= 1 + 2 * ng
    nz_sub -= 2 * ng
    nr_sub -= 2 * ng

# u perpendicular    

    if all((not bnd_xl == 'cyclic', any((ind_pc == 0, ind_pc == npc - 1)))):
        if bnd_zl == 'cyclic':
            k_st = ng
            k_end = ng + nz_sub
        else:
            k_st = ng1
            k_end = ng + nz_sub + 1            
        if bnd_yl == 'cyclic':
            i_st = ng
            i_end = ng + nr_sub
        elif ind_pr == 0 and ind_pr == npr - 1:
            i_st = ng1
            i_end = ng + nr_sub + 1
        elif ind_pr == 0:
            i_st = ng1
            i_end = ng + nr_sub
        elif ind_pr == npr - 1:
            i_st = ng
            i_end = ng + nr_sub + 1
        else:
            i_st = ng
            i_end = ng + nr_sub
        
        if ind_pc == 0:
            j_st_from = ng1
            j_end_from = ng1 + 1
            j_st_to = 0
            j_end_to = ng

            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st_from, j_end_from)]).T)
            inds_to = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st_to, j_end_to)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_vel_perpendicular_x.append(ddcp.transfer(fld, inds_from, inds_to, 'w'))
            bnds_global_rho_x.append(ddcp.transfer(fld, inds_from, inds_from, 'w'))
            bnd_area_x.append(area_eff_x[k_st:k_end, i_st:i_end, j_st_from:j_end_from].flatten())
            
        if ind_pc == npc - 1:
            j_st_from = ng + nc_sub + 1 
            j_end_from = ng + nc_sub + 2
            j_st_to = ng + nc_sub + 1
            j_end_to = 2 * ng + nc_sub + 1

            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st_from, j_end_from)]).T)
            inds_to = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st_to, j_end_to)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_vel_perpendicular_x.append(ddcp.transfer(fld, inds_from, inds_to, 'e'))
            j_st_from = ng + nc_sub
            j_end_from = ng + nc_sub + 1
            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st_from, j_end_from)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_rho_x.append(ddcp.transfer(fld, inds_from, inds_from, 'e'))
            bnd_area_x.append(-area_eff_x[k_st:k_end, i_st:i_end, j_st_from:j_end_from].flatten())

# v perpendicular    

    if all((not bnd_yl == 'cyclic', any((ind_pr == 0, ind_pr == npr - 1)))):
        if bnd_zl == 'cyclic':
            k_st = ng
            k_end = ng + nz_sub
        else:
            k_st = ng1
            k_end = ng + nz_sub + 1
        if bnd_xl == 'cyclic':
            j_st = ng
            j_end = ng + nc_sub
        elif ind_pc == 0 and ind_pc == npc - 1:
            j_st = ng1
            j_end = ng + nc_sub + 1
        elif ind_pc == 0:
            j_st = ng1
            j_end = ng + nc_sub
        elif ind_pc == npc - 1:
            j_st = ng
            j_end = ng + nc_sub + 1
        else:
            j_st = ng
            j_end = ng + nc_sub

        if ind_pr == 0:

            i_st_from = ng1
            i_end_from = ng1 + 1
            i_st_to = 0
            i_end_to = ng

            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st_from, i_end_from) for j in range(j_st, j_end)]).T)
            inds_to = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st_to, i_end_to) for j in range(j_st, j_end)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_vel_perpendicular_y.append(ddcp.transfer(fld, inds_from, inds_to, 's'))
            bnds_global_rho_y.append(ddcp.transfer(fld, inds_from, inds_from, 's'))
            bnd_area_y.append(area_eff_y[k_st:k_end, i_st_from:i_end_from, j_st:j_end].flatten())

        if ind_pr == npr - 1: 
            i_st_from = ng + nr_sub + 1
            i_end_from = ng + nr_sub + 2
            i_st_to = ng + nr_sub + 1
            i_end_to = 2 * ng + nr_sub + 1
 
            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st_from, i_end_from) for j in range(j_st, j_end)]).T)
            inds_to = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st_to, i_end_to) for j in range(j_st, j_end)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_vel_perpendicular_y.append(ddcp.transfer(fld, inds_from, inds_to, 'n'))
            i_st_from = ng + nr_sub
            i_end_from = ng + nr_sub + 1
            inds_from = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st_from, i_end_from) for j in range(j_st, j_end)]).T)
            fld = np.zeros([len(inds_from[0])])
            bnds_global_rho_y.append(ddcp.transfer(fld, inds_from, inds_from, 'n'))
            bnd_area_y.append(-area_eff_y[k_st:k_end, i_st_from:i_end_from, j_st:j_end].flatten())


    if not bnd_zl == 'cyclic':
        k_st_from = ng1
        k_end_from = ng1 + 1
        k_st_to = 0
        k_end_to = ng

        if bnd_yl == 'cyclic':
            i_st = ng
            i_end = ng + nr_sub
        elif ind_pr == 0 and ind_pr == npr - 1:    
            i_st = ng1
            i_end = ng + nr_sub  + 1
        elif ind_pr == 0:
            i_st = ng1
            i_end = ng + nr_sub
        elif ind_pr == npr - 1:
            i_st = ng
            i_end = ng + nr_sub  + 1
        else:
            i_st = ng
            i_end = ng + nr_sub


        if bnd_xl == 'cyclic':
            j_st = ng
            j_end = ng + nc_sub
        elif ind_pc == 0 and ind_pc == npc - 1:
            j_st = ng1
            j_end = ng + nc_sub  + 1
        elif ind_pc == 0:
            j_st = ng1
            j_end = ng + nc_sub     
        elif ind_pc == npc - 1:
            j_st = ng
            j_end = ng + nc_sub  + 1
        else:
            j_st = ng
            j_end = ng + nc_sub
          

        inds_from = list(np.array([(k, i, j) for k in range(k_st_from, k_end_from) for i in range(i_st, i_end) for j in range(j_st, j_end)]).T)
        inds_to = list(np.array([(k, i, j) for k in range(k_st_to, k_end_to) for i in range(i_st, i_end) for j in range(j_st, j_end)]).T)
        fld = np.zeros([len(inds_from[0])])
        bnds_global_vel_perpendicular_z.append(ddcp.transfer(fld, inds_from, inds_to, 'b'))
        bnds_global_rho_z.append(ddcp.transfer(fld, inds_from, inds_from, 'b'))
        bnd_area_z.append(area_eff_z[k_st_from:k_end_from, i_st:i_end, j_st:j_end].flatten())


        k_st_from = ng + nz_sub + 1
        k_end_from = ng + nz_sub + 2
        k_st_to = ng + nz_sub + 1
        k_end_to = 2 * ng + nz_sub + 1

        inds_from = list(np.array([(k, i, j) for k in range(k_st_from, k_end_from) for i in range(i_st, i_end) for j in range(j_st, j_end)]).T)
        inds_to = list(np.array([(k, i, j) for k in range(k_st_to, k_end_to) for i in range(i_st, i_end) for j in range(j_st, j_end)]).T)
        fld = np.zeros([len(inds_from[0])])
        bnds_global_vel_perpendicular_z.append(ddcp.transfer(fld, inds_from, inds_to, 't'))
        k_st_from = ng + nz_sub
        k_end_from = ng + nz_sub + 1
        inds_from = list(np.array([(k, i, j) for k in range(k_st_from, k_end_from) for i in range(i_st, i_end) for j in range(j_st, j_end)]).T)
        fld = np.zeros([len(inds_from[0])])
        bnds_global_rho_z.append(ddcp.transfer(fld, inds_from, inds_from, 't'))
        bnd_area_z.append(-area_eff_z[k_st_from:k_end_from, i_st:i_end, j_st:j_end].flatten())

        k_st = ng1
        k_end = ng1 + 1
 
        j_end_s = j_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st, j_end_s)]).T)
        fld = np.zeros([len(inds[0])]) 
        bnds_global_vel_u_z.append(ddcp.transfer(fld, inds, inds, 'b'))
        dhsurfdx_lst.append(dhsurfdx[0, i_st:i_end, j_st:j_end_s])

        i_end_s = i_end + 1      
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end_s) for j in range(j_st, j_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_global_vel_v_z.append(ddcp.transfer(fld, inds, inds, 'b'))
        dhsurfdy_lst.append(dhsurfdy[0, i_st:i_end_s, j_st:j_end])

        k_st = ng + nz_sub
        k_end = ng + nz_sub + 1 

        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end) for j in range(j_st, j_end_s)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_global_vel_u_z.append(ddcp.transfer(fld, inds, inds, 't'))
        dhsurfdx_lst.append(dhsurfdx[0, i_st:i_end, j_st:j_end_s])

        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(i_st, i_end_s) for j in range(j_st, j_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_global_vel_v_z.append(ddcp.transfer(fld, inds, inds, 't'))
        dhsurfdy_lst.append(dhsurfdy[0, i_st:i_end_s, j_st:j_end])



def bnd_mass_flux(comm, vel_flds, rho):
    """
    Integrates the dimensionally-split boundary mass-flux
    balance from the current data on the boundary objects.

    vel_flds... velocity fields
    rho... density field
    """


    global bnds_global_vel_perpendicular_x
    global bnds_global_vel_perpendicular_y
    global bnds_global_vel_perpendicular_z
    global bnds_global_vel_u_z
    global bnds_global_vel_v_z
    global dhsurfdx_lst
    global dhsurfdy_lst

    global bnds_global_rho_x
    global bnds_global_rho_y
    global bnds_global_rho_z

    global bnd_area_x
    global bnd_area_y
    global bnd_area_z

    u, v, w = vel_flds[:]

    sum_flux_local = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_x):    
         rho_fld = rho[bnds_global_rho_x[n].inds1]
         vel_perp_fld = u[bnds_global_vel_perpendicular_x[n].inds1]
         sum_flux_local += np.sum(rho_fld * vel_perp_fld * bnd_area_x[n])    

    sum_flux_glob_x = ddcp.sum_para(comm.mpicomm, sum_flux_local, comm.pids[1:], comm.pids[0])

    sum_flux_local = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_y):
         rho_fld = rho[bnds_global_rho_y[n].inds1]
         vel_perp_fld = v[bnds_global_vel_perpendicular_y[n].inds1]
         sum_flux_local += np.sum(rho_fld * vel_perp_fld * bnd_area_y[n])

    sum_flux_glob_y = ddcp.sum_para(comm.mpicomm, sum_flux_local, comm.pids[1:], comm.pids[0])

    sum_flux_local = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_z):    
         vel_perp_fld = w[bnds_global_vel_perpendicular_z[n].inds1]
         udhsurfdx = dhsurfdx_lst[n] * (u[bnds_global_vel_u_z[n].inds1]).reshape(dhsurfdx_lst[n].shape)
         vel_perp_fld -= 0.5 * (udhsurfdx[:, 1:] + udhsurfdx[:, :-1]).flatten()
         vdhsurfdy = dhsurfdy_lst[n] * (v[bnds_global_vel_v_z[n].inds1]).reshape(dhsurfdy_lst[n].shape)
         vel_perp_fld -= 0.5 * (vdhsurfdy[1:] + vdhsurfdy[:-1]).flatten()
         rho_fld = rho[bnds_global_rho_z[n].inds1]
         sum_flux_local += np.sum(rho_fld * vel_perp_fld * bnd_area_z[n])

    sum_flux_glob_z = ddcp.sum_para(comm.mpicomm, sum_flux_local, comm.pids[1:], comm.pids[0])    
 
    return sum_flux_glob_x, sum_flux_glob_y, sum_flux_glob_z


def bnd_flux_correct_dimsplit(comm, vel_flds, rho):
    """
    A correction scheme for the perpendicular
    velocity boundary fields to satisfy 
    global mass conservation. It is achieved
    by balancing the radiation flux with the
    prescribed Dirichlet flux. This routine
    balances each 1d-flux to zero.
   
    mass_dirich_x, mass_dirich_y, mass_dirich_z... Dirichlet mass fluxes
    mass_rad_x, mass_rad_y, mass_rad_z... radiation mass fluxes 
    """

    global bnds_global_vel_perpendicular_x
    global bnds_global_vel_perpendicular_y
    global bnds_global_vel_perpendicular_z
    global bnds_global_rho_x
    global bnds_global_rho_y
    global bnds_global_rho_z
    global bnd_area_x
    global bnd_area_y
    global bnd_area_z

    u, v, w = vel_flds[:]

    dmass_x, dmass_y, dmass_z = bnd_mass_flux(comm, vel_flds, rho)

    weights = []
    norm_wghts_loc = 0.0
    rho_areas = []
    for n, bnd in enumerate(bnds_global_vel_perpendicular_x):
         rho_fld = rho[bnds_global_rho_x[n].inds1]
         rho_areas.append(rho_fld * bnd_area_x[n])
         vel = u[bnd.inds1]
         weight = np.maximum(-vel * bnd_area_x[n], 0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)

    norm_wghts = ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    delta_u = [-dmass_x * weight / (norm_wghts * rho_areas[n] + 1e-20) for n, weight in enumerate(weights)]
    for n, bnd in enumerate(bnds_global_vel_perpendicular_x):
        k_s, k_e = bnd.sl2[0][:]
        i_s, i_e = bnd.sl2[1][:]
        j_s, j_e = bnd.sl2[2][:]
        u[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_u[n]).reshape(bnd.shape1)

    weights = []
    norm_wghts_loc = 0.0
    rho_areas = []
    for n, bnd in enumerate(bnds_global_vel_perpendicular_y):
         rho_fld = rho[bnds_global_rho_y[n].inds1]
         rho_areas.append(rho_fld * bnd_area_y[n])
         vel = v[bnd.inds1]
         weight = np.maximum(-vel * bnd_area_y[n], 0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)

    norm_wghts = ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    delta_v = [-dmass_y * weight / (norm_wghts * rho_areas[n] + 1e-20) for n, weight in enumerate(weights)]
    for n, bnd in enumerate(bnds_global_vel_perpendicular_y):
        k_s, k_e = bnd.sl2[0][:] 
        i_s, i_e = bnd.sl2[1][:] 
        j_s, j_e = bnd.sl2[2][:]
        v[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_v[n]).reshape(bnd.shape1)


    weights = []
    norm_wghts_loc = 0.0
    rho_areas = []
    for n, bnd in enumerate(bnds_global_vel_perpendicular_z):
         rho_fld = rho[bnds_global_rho_z[n].inds1]
         rho_areas.append(rho_fld * bnd_area_z[n])
         vel = w[bnd.inds1]
         weight = np.maximum(-vel * bnd_area_z[n], 0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)

    norm_wghts = ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    delta_w = [-dmass_z * weight / (norm_wghts * rho_areas[n] + 1e-20) for n, weight in enumerate(weights)]    
    for n, bnd in enumerate(bnds_global_vel_perpendicular_z):
        k_s, k_e = bnd.sl2[0][:] 
        i_s, i_e = bnd.sl2[1][:] 
        j_s, j_e = bnd.sl2[2][:]
        w[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_w[n]).reshape(bnd.shape1)


def bnd_flux_correct_3d(comm, vel_flds, rho):
    """
    A correction scheme for the perpendicular
    velocity boundary fields to satisfy 
    global mass conservation. It is achieved
    by balancing the radiation flux with the
    prescribed Dirichlet flux.
   
    mass_dirich_x, mass_dirich_y, mass_dirich_z... Dirichlet mass fluxes
    mass_rad_x, mass_rad_y, mass_rad_z... radiation mass fluxes 
    """

    global bnds_global_vel_perpendicular_x
    global bnds_global_vel_perpendicular_y
    global bnds_global_vel_perpendicular_z
    global bnds_global_rho_x
    global bnds_global_rho_y
    global bnds_global_rho_z
    global bnd_area_x
    global bnd_area_y
    global bnd_area_z

    u, v, w = vel_flds[:]

    dmass_x, dmass_y, dmass_z = bnd_mass_flux(comm, vel_flds, rho)

    dmass = dmass_x + dmass_y + dmass_z

    weights = []
    rho_areas = []
    
    norm_wghts_loc = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_x):
         rho_fld = rho[bnds_global_rho_x[n].inds1]
         rho_areas.append(rho_fld * bnd_area_x[n])
         vel = u[bnd.inds1]
         weight = np.maximum(-np.sign(vel) * bnd_area_x[n], 0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)
   
    norm_wghts = ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    norm_wghts_loc = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_y):
         rho_fld = rho[bnds_global_rho_y[n].inds1]         
         rho_areas.append(rho_fld * bnd_area_y[n])
         vel = v[bnd.inds1]
         weight = np.maximum(-np.sign(vel) * bnd_area_y[n], 0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)    
   
    norm_wghts += ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    norm_wghts_loc = 0.0
    for n, bnd in enumerate(bnds_global_vel_perpendicular_z):
         rho_fld = rho[bnds_global_rho_z[n].inds1]
         rho_areas.append(rho_fld * bnd_area_z[n])
         vel = w[bnd.inds1]
         weight = np.maximum(-np.sign(vel) * bnd_area_z[n], 0.0)
         weight.fill(0.0)
         weights.append(weight)
         norm_wghts_loc += np.sum(weight)
         
    norm_wghts += ddcp.sum_para(comm.mpicomm, norm_wghts_loc, comm.pids[1:], comm.pids[0])

    delta_u = [-dmass * weights[n] / (norm_wghts * rho_areas[n] + 1e-20) for n, bnd in enumerate(bnds_global_vel_perpendicular_x)]

    for n, bnd in enumerate(bnds_global_vel_perpendicular_x):
        k_s, k_e = bnd.sl2[0][:] 
        i_s, i_e = bnd.sl2[1][:] 
        j_s, j_e = bnd.sl2[2][:]
        u[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_u[n]).reshape(bnd.shape1)

    m = len(bnds_global_vel_perpendicular_x)
    delta_v = [-dmass * weights[n + m] / (norm_wghts * rho_areas[n + m] + 1e-20) for n, bnd in enumerate(bnds_global_vel_perpendicular_y)]

    for n, bnd in enumerate(bnds_global_vel_perpendicular_y):
        k_s, k_e = bnd.sl2[0][:] 
        i_s, i_e = bnd.sl2[1][:] 
        j_s, j_e = bnd.sl2[2][:]
        v[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_v[n]).reshape(bnd.shape1)

    m = len(bnds_global_vel_perpendicular_x) + len(bnds_global_vel_perpendicular_y)
    delta_w = [-dmass * weights[n + m] / (norm_wghts * rho_areas[n + m] + 1e-20) for n, bnd in enumerate(bnds_global_vel_perpendicular_z)]

    for n, bnd in enumerate(bnds_global_vel_perpendicular_z):
        k_s, k_e = bnd.sl2[0][:] 
        i_s, i_e = bnd.sl2[1][:] 
        j_s, j_e = bnd.sl2[2][:]
        w[k_s:k_e, i_s:i_e, j_s:j_e] += (delta_w[n]).reshape(bnd.shape1)



def init_turbrec_scheme(comm, param_dict):
    """
    Initializes the turbulence recycling scheme, i.e.
    the filter operators and boundary objects for parallel
    filtering and data transfer to the inflow boundaries.

    comm... ddcp.communicator
    param_dict... parameter disctionary
    """

    global ng

    global bnds_global_lst

    global filter_u_lst, filter_v_lst, filter_w_lst

    global mapping_u_lst
    global mapping_v_lst
    global mapping_w_lst

    # temporary velocity component with ghost cells to filter
    global u_tmp_lst, v_tmp_lst, w_tmp_lst
    global shp_u_lst, shp_v_lst, shp_w_lst

    global rand_fld_u_lst, rand_fld_v_lst, rand_fld_w_lst
    global u_rand_shp_lst, v_rand_shp_lst, w_rand_shp_lst
    global u_filtered, v_filtered, w_filtered

    global ind_tpls_same

    # bound objects for turbulence recycling (if the recycling plane is outside the bounding subdomain)
    global bnds_send_u, bnds_send_v, bnds_send_w
    global bnds_recv_u, bnds_recv_v, bnds_recv_w
    global bnds_3d_recv_u, bnds_3d_recv_v, bnds_3d_recv_w

    # bound object for communications associated with the filter operation
    global bnds_filter_send_u, bnds_filter_send_v, bnds_filter_send_w
    global bnds_filter_recv_u, bnds_filter_recv_v, bnds_filter_recv_w

    # process ids for parallel calculation of rms
    global pids_w_child, pids_e_child, pids_s_child, pids_n_child
    global pid_w_root, pid_e_root, pid_s_root, pid_n_root

    global bnd_obj_ind_recturb

    plane_bnd_lst = []
    infl_bnd_lst = []

    filter_u_lst = []
    filter_v_lst = []
    filter_w_lst = []

    mapping_u_lst = []
    mapping_v_lst = []
    mapping_w_lst = []

    bnds_send_u = []
    bnds_recv_u = []
    bnds_send_v = []
    bnds_recv_v = []
    bnds_send_w = []
    bnds_recv_w = []

    bnds_3d_recv_u = []
    bnds_3d_recv_v = []
    bnds_3d_recv_w = []

    bnds_filter_send_u = []
    bnds_filter_recv_u = []
    bnds_filter_send_v = []
    bnds_filter_recv_v = []
    bnds_filter_send_w = []
    bnds_filter_recv_w = []

    rand_fld_u_lst = []
    rand_fld_v_lst = []
    rand_fld_w_lst = []
    u_rand_shp_lst = []
    v_rand_shp_lst = []
    w_rand_shp_lst = []

    u_tmp_lst = []
    v_tmp_lst = []
    w_tmp_lst = []
    shp_u_lst = []
    shp_v_lst = []
    shp_w_lst = []

    ind_tpls_same = []
   
    bnd_obj_ind_recturb = []

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    pids = comm.pids
    npr = comm.npr
    npc = comm.npc
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz

    dh = param_dict['dh']
    xcoord = param_dict['xcoord']
    ycoord = param_dict['ycoord']
    zcoord = param_dict['zcoord']
    x2coord = param_dict['x2coord']
    y2coord = param_dict['y2coord']
    zcoord = param_dict['zcoord']
    dz_min = np.min(param_dict['dz'][param_dict['dz'] > 0])
    dz_max = np.max(param_dict['dz'][param_dict['dz'] > 0])
    dz = dz_min

    n_plane = 1

    bnd_x = param_dict['bnd_xl']
    bnd_y = param_dict['bnd_yl']

    ind_p = pids.index(rank)
    ind_pr = ind_p / (len(ncj) - 1)
    ind_pc = ind_p - ind_pr * (len(ncj) - 1)
    nc_sub = ncj[ind_pc + 1] - ncj[ind_pc] + 2 * ng
    nr_sub = nri[ind_pr + 1] - nri[ind_pr] + 2 * ng
    nz_sub = nz

    #operations on the subdomains of the recycling plane

    sides = []

    if not bnd_x == 'cyclic':
        if 'w' in param_dict["rec_turb_sides"]:
            sides.append('w')
        if 'e' in param_dict["rec_turb_sides"]:
            sides.append('e')
    if not bnd_y == 'cyclic':
        if 's' in param_dict["rec_turb_sides"]:
            sides.append('s')
        if 'n' in param_dict["rec_turb_sides"]:
            sides.append('n')

    for n, side in enumerate(sides):
        if side == 'w':
            ind_plane = np.argmin(np.absolute(xcoord - xcoord[0] - param_dict['rec_turb_plane_dists'][0]))            
            ind_p_rec = np.argmin(np.absolute(0.5 * (np.array(ncj)[1:] + np.array(ncj)[:-1]) - ind_plane))            
            ir_st = 0
            ir_end = nr_sub -  2 * ng
            jc_st = ind_plane - ncj[ind_p_rec] + ng
            jc_end = jc_st + n_plane
        elif side == 'e':
            ind_plane = np.argmin(np.absolute(xcoord - xcoord[-1] + param_dict['rec_turb_plane_dists'][1]))            
            ind_p_rec = np.argmin(np.absolute(0.5 * (np.array(ncj)[1:] + np.array(ncj)[:-1]) - ind_plane))
            ir_st = 0
            ir_end = nr_sub - 2 * ng
            jc_st = ind_plane - ncj[ind_p_rec] + ng
            jc_end = jc_st + n_plane
        elif side == 's':
            ind_plane = np.argmin(np.absolute(ycoord - ycoord[0] - param_dict['rec_turb_plane_dists'][2]))
            ind_p_rec = np.argmin(np.absolute(0.5 * (np.array(nri)[1:] + np.array(nri)[:-1]) - ind_plane))
            ir_st = ind_plane - nri[ind_p_rec] + ng
            ir_end = ir_st + n_plane
            jc_st = 0
            jc_end = nc_sub - 2 * ng
        elif side == 'n':
            ind_plane = np.argmin(np.absolute(ycoord - ycoord[-1] + param_dict['rec_turb_plane_dists'][3]))
            ind_p_rec = np.argmin(np.absolute(0.5 * (np.array(nri)[1:] + np.array(nri)[:-1]) - ind_plane))
            ir_st = ind_plane - nri[ind_p_rec] + ng
            ir_end = ir_st + n_plane
            jc_st = 0
            jc_end = nc_sub - 2 * ng

        k_st = 0
        k_end = nz

        for m, bnd in enumerate(bnds_global_lst[0]):
            if bnd.side == side:
                bnd_obj_ind_recturb.append(m)  


        if ind_pc == ind_p_rec and (side == 'w' or side == 'e'):

            
            filter_2d_u, bnds_send_u_sub_lst, bnds_recv_u_sub_lst, ind_tpl_same, u_tmp, shp_u_tmp = filt.setup_plane_filter(comm, 'we', ind_p_rec, param_dict, tag_ref=1000000, type='c')
            filter_2d_v, bnds_send_v_sub_lst, bnds_recv_v_sub_lst, ind_tpl_same, v_tmp, shp_v_tmp = filt.setup_plane_filter(comm, 'we', ind_p_rec, param_dict, tag_ref=3000000, type='sv')
            filter_2d_w, bnds_send_w_sub_lst, bnds_recv_w_sub_lst, ind_tpl_same, w_tmp, shp_w_tmp = filt.setup_plane_filter(comm, 'we', ind_p_rec, param_dict, tag_ref=5000000, type='sw')

            ind_tpls_same.append(ind_tpl_same)
 
            u_tmp_lst.append(u_tmp)
            v_tmp_lst.append(v_tmp)
            w_tmp_lst.append(w_tmp)

            shp_u_lst.append(shp_u_tmp)
            shp_v_lst.append(shp_v_tmp)
            shp_w_lst.append(shp_w_tmp)

            bnds_filter_send_u.append(bnds_send_u_sub_lst)
            bnds_filter_send_v.append(bnds_send_v_sub_lst)
            bnds_filter_send_w.append(bnds_send_w_sub_lst)

            bnds_filter_recv_u.append(bnds_recv_u_sub_lst)
            bnds_filter_recv_v.append(bnds_recv_v_sub_lst)
            bnds_filter_recv_w.append(bnds_recv_w_sub_lst)

            filter_u_lst.append(filter_2d_u)
            filter_v_lst.append(filter_2d_v)
            filter_w_lst.append(filter_2d_w)

            if side == 'w':
                pid = ind_pr * npc
            else:
                pid = (ind_pr + 1) * npc - 1
            tag = n * 500000 + ind_pr
            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_u.append(ddcp.bound(fld, inds, pid, side, tag))
            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end + 1) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_v.append(ddcp.bound(fld, inds, pid, side, tag + 300000))
            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_w.append(ddcp.bound(fld, inds, pid, side, tag + 600000))


        if ind_pr == ind_p_rec and (side == 's' or side == 'n'):

            if side == 's':
                pid = ind_pc
            else:
                pid = ind_pc + (npr - 1) * npc
            tag = n * 500000 + ind_pc

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_u.append(ddcp.bound(fld, inds, pid, side, tag))
            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_v.append(ddcp.bound(fld, inds, pid, side, tag + 300000))
            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_send_w.append(ddcp.bound(fld, inds, pid, side, tag + 600000))

            filter_2d_u, bnds_send_u_sub_lst, bnds_recv_u_sub_lst, ind_tpl_same, u_tmp, shp_u_tmp = filt.setup_plane_filter(comm, 'sn', ind_p_rec, param_dict, tag_ref=1000000, type='sv')
            filter_2d_v, bnds_send_v_sub_lst, bnds_recv_v_sub_lst, ind_tpl_same, v_tmp, shp_v_tmp = filt.setup_plane_filter(comm, 'sn', ind_p_rec, param_dict, tag_ref=3000000, type='c')
            filter_2d_w, bnds_send_w_sub_lst, bnds_recv_w_sub_lst, ind_tpl_same, w_tmp, shp_w_tmp = filt.setup_plane_filter(comm, 'sn', ind_p_rec, param_dict, tag_ref=5000000, type='sw')

            ind_tpls_same.append(ind_tpl_same)

            u_tmp_lst.append(u_tmp)
            v_tmp_lst.append(v_tmp)
            w_tmp_lst.append(w_tmp)

            shp_u_lst.append(shp_u_tmp)
            shp_v_lst.append(shp_v_tmp)
            shp_w_lst.append(shp_w_tmp)

            bnds_filter_send_u.append(bnds_send_u_sub_lst)
            bnds_filter_send_v.append(bnds_send_v_sub_lst)
            bnds_filter_send_w.append(bnds_send_w_sub_lst)

            bnds_filter_recv_u.append(bnds_recv_u_sub_lst)
            bnds_filter_recv_v.append(bnds_recv_v_sub_lst)
            bnds_filter_recv_w.append(bnds_recv_w_sub_lst)

            filter_u_lst.append(filter_2d_u)
            filter_v_lst.append(filter_2d_v)
            filter_w_lst.append(filter_2d_w)


    # at the inflow boundary

        if ind_pc == 0 and side == 'w':

            ir_st = 0
            ir_end = nr_sub - 2 * ng
            jc_st = 0
            jc_end = n_plane
            k_st = 0
            k_end = nz
            pid = ind_pr * npc + ind_p_rec
            tag = n * 500000 + ind_pr

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(0, 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_u.append(ddcp.bound(fld, inds, pid, 'w', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end + 1) for j in range(0, 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_v.append(ddcp.bound(fld, inds, pid, 'w', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(0, 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_w.append(ddcp.bound(fld, inds, pid, 'w', tag + 600000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(0, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_u.append(ddcp.bound(fld, inds, pid, 'w', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end + 1) for j in range(0, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_v.append(ddcp.bound(fld, inds, pid, 'w', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(0, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_w.append(ddcp.bound(fld, inds, pid, 'w', tag + 600000))

            pids_w_child = [npc * p for p in range(1, npr)]
            pid_w_root = 0

        if ind_pc == npc - 1 and side == 'e':

            ir_st = 0
            ir_end = nr_sub - 2 * ng
            jc_st = nc_sub - n_plane
            jc_end = nc_sub
            k_st = 0
            k_end = nz
            pid = ind_pr * npc + ind_p_rec
            tag = n * 500000 + ind_pr

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(ng - 1, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_u.append(ddcp.bound(fld, inds, pid, 'e', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end + 1) for j in range(ng - 1, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_v.append(ddcp.bound(fld, inds, pid, 'e', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(ng - 1, ng)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_w.append(ddcp.bound(fld, inds, pid, 'e', tag + 600000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(nc_sub + 1 - ng, nc_sub + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_u.append(ddcp.bound(fld, inds, pid, 'e', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end + 1) for j in range(nc_sub - ng, nc_sub)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_v.append(ddcp.bound(fld, inds, pid, 'e', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ir_st, ir_end) for j in range(nc_sub - ng, nc_sub)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_w.append(ddcp.bound(fld, inds, pid, 'e', tag + 600000))

            pids_e_child = [npc * p - 1 for p in range(1, npr)]
            pid_e_root = npr * npc - 1

        if ind_pr == 0 and side == 's':

            ir_st = 0
            ir_end = n_plane
            jc_st = 0
            jc_end = nc_sub - 2 * ng
            k_st = 0
            k_end = nz
            pid = ind_p_rec * npc + ind_pc
            tag = n * 500000 + ind_pc

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(0, 1) for j in range(jc_st, jc_end + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_u.append(ddcp.bound(fld, inds, pid, 's', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(0, 1) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_v.append(ddcp.bound(fld, inds, pid, 's', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(0, 1) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_w.append(ddcp.bound(fld, inds, pid, 's', tag + 600000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(0, ng) for j in range(jc_st, jc_end + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_u.append(ddcp.bound(fld, inds, pid, 's', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(0, ng) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_v.append(ddcp.bound(fld, inds, pid, 's', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(0, ng) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_w.append(ddcp.bound(fld, inds, pid, 's', tag + 600000))

            pids_s_child = [p for p in range(0, npc - 1)]
            pid_s_root = npc - 1

        if ind_pr == npr - 1 and side == 'n':
            ir_st = nr_sub - n_plane
            ir_end = nr_sub
            jc_st = 0
            jc_end = nc_sub - 2 * ng
            k_st = 0
            k_end = nz
            pid = ind_p_rec * npc + ind_pc
            tag = n * 500000 + ind_pc

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ng - 1, ng) for j in range(jc_st, jc_end + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_u.append(ddcp.bound(fld, inds, pid, 'n', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ng - 1, ng) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_v.append(ddcp.bound(fld, inds, pid, 'n', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(ng - 1, ng) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_recv_w.append(ddcp.bound(fld, inds, pid, 'n', tag + 600000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(nr_sub - ng, nr_sub) for j in range(jc_st, jc_end + 1)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_u.append(ddcp.bound(fld, inds, pid, 'n', tag))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(nr_sub + 1 - ng, nr_sub + 1) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_v.append(ddcp.bound(fld, inds, pid, 'n', tag + 300000))

            inds = list(np.array([(k, i, j) for k in range(k_st, k_end + 1) for i in range(nr_sub - ng, nr_sub) for j in range(jc_st, jc_end)]).T)
            fld = np.zeros([len(inds[0])])
            bnds_3d_recv_w.append(ddcp.bound(fld, inds, pid, 'n', tag + 600000))

            pids_n_child = [(npr - 1) * npc + p for p in range(1, npc)]
            pid_n_root = (npr - 1) * npc



def seed_random_fluct(thetav, vels, param_dict, ncells=5, nr_eck=0.010):
    """
    Adds random fluctuations to the thetav field
    using horizontal square blocks of ncells x ncells  
    with a mean specific amplitude calculated from 
    the perturbation Eckert number (Munoz-Esparzaet al., 2015.
         
    thetav... prognostic field of virtual potential temperature
    vels... velocity component
    param_dict... parameter dictionary
    ncells... block size
    nr_eck... prescribed perturbation Eckert number      
    """

    global ng

    u, v, w = vels[:]

    nz = thetav.shape[0] - 2 * ng

    speed_avg = np.sqrt(np.mean(u[ng:-ng, ng:-ng, ng:-ng]) ** 2 + np.mean(v[ng:-ng, ng:-ng, ng:-ng]) ** 2 + np.mean(w[ng + 1:-ng, ng:-ng, ng:-ng]) ** 2)

    amp = 1.0 / nr_eck * speed_avg ** 2 / 1000.0

    thetav_per = np.zeros_like(thetav[ng:-ng, ng:-ng, ng:-ng])

    shp = thetav_per.shape

    blocks_nz = [1 for k in range(shp[0])]
    blocks_ny = [ncells for i in range(int(shp[1]/ncells))]
    if shp[1] % ncells != 0:
            blocks_ny.append(shp[1] % ncells)
    blocks_nx = [ncells for j in range(int(shp[2]/ncells))]
    if shp[2] % ncells != 0:
            blocks_nx.append(shp[2] % ncells)

    nblocks_x = len(blocks_nx)
    nblocks_y = len(blocks_ny)
    nblocks_z = len(blocks_nz)

    rand_nrs = np.random.normal(np.zeros([nblocks_x * nblocks_y * nblocks_z]), amp)

    blocks = []
    for k in range(nblocks_z):
        k_block = []
        for i in range(nblocks_y):
            i_block = []
            for j in range(nblocks_x):
                i_block.append(np.full([blocks_nz[k], blocks_ny[i], blocks_nx[j]], rand_nrs[k * nblocks_y * nblocks_x +  i * nblocks_x + j]))
            k_block.append(i_block)
        blocks.append(k_block)

    thetav_per[:] = np.block(blocks)

    thetav[ng:-ng, ng:-ng, ng:-ng] += thetav_per[:, :, :]



def recycle_turb(comm, u, v, w, urms, vrms, wrms, param_dict):
    """
    Turbulence recycling scheme for faster spinup 
    of turbulence at the inflow boundaries.
    The small-scale spectral part is filtered out,
    rescaled and superimposed on the Dirichlet inflow boundary fields.
    The filter-width determines the scale separation. 
    
    comm... ddcp.communicator 
    u, v, w... prognostic velocity fields
    urms, vrms, wrms... target root mean square of fluctuations
    param_dict... parameter dictionary
    """

    global filter_u_lst, filter_v_lst, filter_w_lst

    # temporary velocity component with ghost cells to filter
    global u_tmp_lst, v_tmp_lst, w_tmp_lst
    global shp_u_lst, shp_v_lst, shp_w_lst

    # bound objects for turbulence recycling (if the recycling plane is outside the bounding subdomain)
    global bnds_send_u, bnds_send_v, bnds_send_w
    global bnds_recv_u, bnds_recv_v, bnds_recv_w

    # bound object for communications associated with the filter operation
    global bnds_filter_send_u, bnds_filter_send_v, bnds_filter_send_w
    global bnds_filter_recv_u, bnds_filter_recv_v, bnds_filter_recv_w

    global ng

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    pids = comm.pids
    npr = comm.npr
    npc = comm.npc
    nri = comm.nri
    ncj = comm.ncj
    nz = comm.nz
    ind_p = pids.index(rank)
    ind_pr = ind_p / (len(ncj) - 1)
    ind_pc = ind_p - ind_pr * (len(ncj) - 1)

    u_sl_fluct_lst = []
    v_sl_fluct_lst = []
    w_sl_fluct_lst = []

    for n, obj in enumerate(filter_u_lst):

        # copy data on temporary field
        if bnds_send_u[n].side == 'w' or bnds_send_u[n].side == 'e':
            sl = bnds_send_u[n].slice
            shp = bnds_send_u[n].shape
            ind_tpl_j = sl[2]
            il_same, ih_same = ind_tpls_same[n]            
            u_tmp_lst[n][:, il_same:ih_same, :] = u[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            v_tmp_lst[n][:, il_same:ih_same + 1, :] = v[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            w_tmp_lst[n][:, il_same:ih_same, :] = w[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            u_send = u[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            v_send = v[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            w_send = w[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
        else:
            sl = bnds_send_v[n].slice
            shp = bnds_send_v[n].shape
            ind_tpl_i = sl[1]
            jl_same, jh_same = ind_tpls_same[n]
            u_tmp_lst[n][:, :, jl_same:jh_same + 1] = u[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            v_tmp_lst[n][:, :, jl_same:jh_same] = v[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            w_tmp_lst[n][:, :, jl_same:jh_same] = w[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            u_send = u[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            v_send = v[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            w_send = w[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()


        shp_u = shp_u_lst[n][0]
        shp_v = shp_v_lst[n][0]
        shp_w = shp_w_lst[n][0]
        u_lb, u_hb = shp_u_lst[n][1][:]
        v_lb, v_hb = shp_v_lst[n][1][:]
        w_lb, w_hb = shp_w_lst[n][1][:]

        # first communications to collect the missing field parts before filtering

        ddcp.cptobounds(u_send, bnds_filter_send_u[n], mode = 'repl')
        ddcp.cptobounds(v_send, bnds_filter_send_v[n], mode = 'repl')
        ddcp.cptobounds(w_send, bnds_filter_send_w[n], mode = 'repl')

        ddcp.exchange_fields(mpicomm, bnds_filter_send_u[n], bnds_filter_recv_u[n])
        ddcp.exchange_fields(mpicomm, bnds_filter_send_v[n], bnds_filter_recv_v[n])
        ddcp.exchange_fields(mpicomm, bnds_filter_send_w[n], bnds_filter_recv_w[n])

        ddcp.cpfrombounds(u_tmp_lst[n], bnds_filter_recv_u[n], mode = 'repl')
        ddcp.cpfrombounds(v_tmp_lst[n], bnds_filter_recv_v[n], mode = 'repl')
        ddcp.cpfrombounds(w_tmp_lst[n], bnds_filter_recv_w[n], mode = 'repl')

        #do the filtering
        u_filt = (filter_u_lst[n] * u_tmp_lst[n].flatten()).reshape(shp_u)
        v_filt = (filter_v_lst[n] * v_tmp_lst[n].flatten()).reshape(shp_v)
        w_filt = (filter_w_lst[n] * w_tmp_lst[n].flatten()).reshape(shp_w)        

        #subtract from the base to get the fluctuations
        if bnds_send_u[n].side == 'w' or bnds_send_u[n].side == 'e':
            u_sl_filtered = u[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            v_sl_filtered = v[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()
            w_sl_filtered = w[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]].copy()

            u_sl_filtered[:], v_sl_filtered[:], w_sl_filtered[:] = u_filt, v_filt, w_filt
            u_sl_fluct = u[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]] - u_sl_filtered
            v_sl_fluct = v[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]] - v_sl_filtered
            w_sl_fluct = w[ng:-ng, ng:-ng, ind_tpl_j[0]:ind_tpl_j[1]] - w_sl_filtered

        else:
            u_sl_filtered = u[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            v_sl_filtered = v[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            w_sl_filtered = w[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng].copy()
            u_sl_filtered[:], v_sl_filtered[:], w_sl_filtered[:] = u_filt, v_filt, w_filt
            u_sl_fluct = u[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng] -  u_sl_filtered
            v_sl_fluct = v[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng] -  v_sl_filtered
            w_sl_fluct = w[ng:-ng, ind_tpl_i[0]:ind_tpl_i[1], ng:-ng] - w_sl_filtered

        u_sl_fluct = u_sl_fluct * param_dict['rec_turb_sign']
        v_sl_fluct = v_sl_fluct * param_dict['rec_turb_sign']        
        w_sl_fluct = w_sl_fluct * param_dict['rec_turb_sign']
        w_sl_fluct[0] = 0.5 * (w_sl_fluct[0] + w_sl_fluct[-1])
        w_sl_fluct[-1] = w_sl_fluct[0]

        #final communications to insert the recycled turbulence at the inflow boundaries
        bnds_send_u[n].update(u_sl_fluct.flatten())
        bnds_send_v[n].update(v_sl_fluct.flatten())
        bnds_send_w[n].update(w_sl_fluct.flatten())

    ddcp.exchange_fields(mpicomm, bnds_send_u, bnds_recv_u)
    ddcp.exchange_fields(mpicomm, bnds_send_v, bnds_recv_v)
    ddcp.exchange_fields(mpicomm, bnds_send_w, bnds_recv_w)


    #rescale turbulent intensity to target intensity

    # maximum turbulence amplification
    max_ampl = param_dict['rec_turb_maxfac']
    
    for n, bnd in enumerate(bnds_recv_u):
        side = bnd.side
        data = bnd.data
        shape = bnd.shape
        data_3d = data.reshape(shape)
        if side == 'w':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_w_child, pid_w_root, axis=(1,))
           resc_fac = np.minimum(urms[ng:-ng, ng:-ng, :shape[2]] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 'e':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_e_child, pid_e_root, axis=(1,))
           resc_fac = np.minimum(urms[ng:-ng, ng:-ng, -shape[2]:] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 's':
           rms = ddcp.fld_rms_para(mpicomm, data_3d[:, : ,:-1], pids_s_child, pid_s_root, axis=(2,))
           resc_fac = np.minimum(urms[ng:-ng, :shape[1], ng:-ng] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 'n':
           rms = ddcp.fld_rms_para(mpicomm, data_3d[:, : , :-1], pids_n_child, pid_n_root, axis=(2,))
           resc_fac = np.minimum(urms[ng:-ng, -shape[1]:, ng:-ng] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        bnd.update(data)
     
    for n, bnd in enumerate(bnds_recv_v):
        side = bnd.side
        data = bnd.data
        shape = bnd.shape
        data_3d = data.reshape(shape)
        if side == 'w':
           rms = ddcp.fld_rms_para(mpicomm, data_3d[:, :-1], pids_w_child, pid_w_root, axis=(1,))
           resc_fac = np.minimum(vrms[ng:-ng, ng:-ng, :shape[2]] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 'e':
           rms = ddcp.fld_rms_para(mpicomm, data_3d[:, :-1], pids_e_child, pid_e_root, axis=(1,))
           resc_fac = np.minimum(vrms[ng:-ng, ng:-ng, -shape[2]:] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 's':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_s_child, pid_s_root, axis=(2,))
           resc_fac = np.minimum(vrms[ng:-ng, :shape[1], ng:-ng] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        elif side == 'n':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_n_child, pid_n_root, axis=(2,))
           resc_fac = np.minimum(vrms[ng:-ng, -shape[1]:, ng:-ng] / (rms + 1e-20), max_ampl)
           data = data * resc_fac.flatten()
        bnd.update(data)

    for n, bnd in enumerate(bnds_recv_w):
        side = bnd.side
        data = bnd.data
        shape = bnd.shape       
        data_3d = data.reshape(shape)
        if side == 'w':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_w_child, pid_w_root, axis=(1,))
           resc_fac = np.minimum(wrms[ng:-ng, ng:-ng, :shape[2]] / (rms + 1e-20), max_ampl)
           data[:] = data * resc_fac.flatten()
        elif side == 'e':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_e_child, pid_e_root, axis=(1,))
           resc_fac = np.minimum(wrms[ng:-ng, ng:-ng, -shape[2]:] / (rms + 1e-20), max_ampl)
           data[:] = data * resc_fac.flatten()
        elif side == 's':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_s_child, pid_s_root, axis=(2,))
           resc_fac = np.minimum(wrms[ng:-ng, :shape[1], ng:-ng] / (rms + 1e-20), max_ampl)
           data[:] = data * resc_fac.flatten()
        elif side == 'n':
           rms = ddcp.fld_rms_para(mpicomm, data_3d, pids_n_child, pid_n_root, axis=(2,))
           resc_fac = np.minimum(wrms[ng:-ng, -shape[1]:, ng:-ng] / (rms + 1e-20), max_ampl)
           data[:] = data * resc_fac.flatten()
        bnd.update(data)
   
    ddcp.shiftdata(bnds_3d_recv_u)
    ddcp.shiftdata(bnds_3d_recv_v)
    ddcp.shiftdata(bnds_3d_recv_w)
    
    ddcp.cpbound2bound(bnds_3d_recv_u, bnds_recv_u)
    ddcp.cpbound2bound(bnds_3d_recv_v, bnds_recv_v)
    ddcp.cpbound2bound(bnds_3d_recv_w, bnds_recv_w)

    return [bnds_3d_recv_u, bnds_3d_recv_v, bnds_3d_recv_w]

