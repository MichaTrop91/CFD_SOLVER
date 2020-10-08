# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
#Last modified: 10.10.2020


# load external python packages
import numpy as np
from scipy.sparse import csr_matrix

#load model specific *py files
from mpi4py import MPI
import sps_operators as ops


class bound:
   """
   A field part that is shared with other processors.
   The indices correspond to the indices of the data-field.
   The processor id is that to/from which data is send/received.
   """

   def __init__(self, data, inds, pid, side, tag=None):
       """
       Initializes the bound object.

       data... data array to share
       inds... list of index arrays of subdomain field part to which data corresponds
       pid... process id with which data is shared
       side... side in case of a lateral boundary (can be 'w', 'e', 's', 'n', 'b', 't', or combinations 'sw', 'se', ...)
       tag... communication tag
       """
    
       self.data = data.copy()
       self.inds = inds
       self.pid = pid
       self.side = side
       self.tag = tag
       self.aux = 1.0
       self.aux2 = 1.0
       shp = [len(set(inds_comp.tolist())) for inds_comp in inds]
       self.shape = shp
       if len(inds[0]):
           sl = [(np.min(array), np.max(array) + 1) for array in inds]
       else:
           sl = []
       self.slice = sl

   def update(self, data):
       """
       Copies new data on bound

       data... new data
       """

       self.data[:] = data

   def add(self, fld):
       """
       Adds data to existing data on bound.
       
       fld... data to add
       """

       self.data[:] = self.data + fld

   def sub(self, fld):
       """
       Subtracts data from existing data on bound.
       
       fld... data to subtract
       """

       self.data[:] = self.data - fld

   def mul(self, fld):    
       """
       Multiplies data with existing data on bound.
       
       fld... data to multiply
       """
             
       self.data[:] = self.data * fld
        
   def div(self, fld):
       """
       Divides existing data on bound.
       
       fld... divisor
       """

       self.data[:] = self.data / fld

   def add_aux(self, aux):
       """
       Stores an auxiliary field on bound
       which is repeatedly used.
       
       aux... auxiliary field
       """

       self.aux = aux.copy()

   def add_aux2(self, aux2):
       """
       Stores a second auxiliary field on bound
       which is repeatedly used.
       
       aux2... auxiliary field
       """

       self.aux2 = aux2.copy()

   def add_op(self, OP):
       """
       Stores a linear operator on bound
       which is repeatedly applied on bound data.
       
       OP... linear operator
       """

       self.op = OP.copy()

   def mul_op(self):
       """
       Apply linear operator on bound data.
       """

       self.data[:] = self.op * self.data
      
       
class transfer:
   """
   Transfer object to transfer data from one
   field to another field on the same processor.
   """

   def __init__(self, data, inds1, inds2, side=None):
       """
       Initializes transfer  object.

       data... data to transfer
       inds1... list of index arrays to adress data on first field
       inds2... list of index arrays to adress data on second field
       """

       self.data = data
       self.inds1 = inds1
       self.inds2 = inds2
       self.aux = 1.0
       self.aux2 = 1.0
       self.side = side

       shp1 = [len(set(inds_comp.tolist())) for inds_comp in inds1]
       shp2 = [len(set(inds_comp.tolist())) for inds_comp in inds2]
       
       sl1 = [(np.min(inds_comp), np.max(inds_comp) + 1) for inds_comp in inds1]
       sl2 = [(np.min(inds_comp), np.max(inds_comp) + 1) for inds_comp in inds2] 

       self.shape1 = shp1
       self.shape2 = shp2

       self.sl1 = sl1
       self.sl2 = sl2

   def update(self, data):
       """
       Copies new data on transfer.

       data... new data
       """

       self.data[:]  = data

   def mul(self, fld):
       """
       Multiplies data with existing data on transfer.
       
       fld... data to multiply
       """
                 
       self.data[:] = self.data * fld

   def add(self, fld):
       """
       Adds data to existing data on transfer.
       
       fld... data to add
       """

       self.data[:] = self.data + fld

   def add_aux(self, aux):
       """
       Stores an auxiliary field on transfer
       which is repeatedly used.
       
       aux... auxiliary field
       """

       self.aux = aux.copy()

   def add_aux2(self, aux2):
       """
       Stores a second auxiliary field on transfer
       which is repeatedly used.
       
       aux2... auxiliary field
       """

       self.aux2 = aux2.copy()
  
   def add_op(self, OP):
       """
       Stores a linear operator on transfer
       which is repeatedly applied on transfer data.
       
       OP... linear operator
       """

       self.op = OP.copy()


class communicator:
    """
    MPI communicator with 
    additional information for
    the subdomain decomposition.
    """

    def __init__(self, mpicomm, npr, npc, pids): 
        """
        Initializes the communicator object

        mpicomm... MPI communicator
        npr... number of processes in each row
        npc... number of processes in each column
        pids... process ids
        """

        self.mpicomm = mpicomm
        self.npr = npr
        self.npc = npc
        self.pids = pids

    def set_dombnds(self, nz, nri, ncj):
        """
        Adds additional information after
        the domain decomposition is determined.

        nz... number of vertical layers
        nri... global row boundary indices of subdomains
        ncj... global column boundary indices of subdomains
        """

        self.nri = nri
        self.ncj = ncj
        self.nz = nz
        
        mpicomm = self.mpicomm
        rank = mpicomm.Get_rank()
        ind_p = self.pids.index(rank)
        ind_pr = int(ind_p / self.npc)
        ind_pc = ind_p - ind_pr * self.npc
        self.ind_p = ind_p
        self.ind_pr = ind_pr
        self.ind_pc = ind_pc


def partition_domain(nr, nc, npr, npc):
    """
    Defines a domain decomposition
    given the domain size and the number 
    of proccessors.
    
    nr, nc... number of rows/columns (y, x)
    npr, npc... number of processors in each row/column 
    """
 
    nri_av = float(nr) / float(npr)
    ncj_av = float(nc) / float(npc) 
    nri_l = int(nri_av)
    nri_h = nri_l + 1
    ncj_l = int(ncj_av)
    ncj_h = ncj_l + 1
    nl = npr * (nri_h) - nr
    nh = npr - nl 
    nri = [0]

    n_min = min(nh, nl)
    for n in range(n_min):
        nri.append(nri[-1] + nri_h)
        nri.append(nri[-1] + nri_l)

    for n in range(max(0, nh - n_min)):
        nri.append(nri[-1] + nri_h)
    for n in range(max(0, nl - n_min)):
        nri.append(nri[-1] + nri_l)

    nl = npc * (ncj_h) - nc
    nh = npc - nl
    ncj = [0]
    n_min = min(nh, nl)
    for n in range(n_min):
        ncj.append(ncj[-1] + ncj_h)
        ncj.append(ncj[-1] + ncj_l)
    for n in range(max(0, nh - n_min)):
        ncj.append(ncj[-1] + ncj_h)
    for n in range(max(0, nl - n_min)):
        ncj.append(ncj[-1] + ncj_l)           

    return nri, ncj

       
def make_halos(comm, bnd_x, bnd_y, bnd_z, type='c', ghst_inds=3):
    """
    This routine constructs the boundary fields
    for data exchange between subdomains to derive
    the explicit tendencies.

    comm... communicator
    bnd_x, bnd_y, bnd_z... type of boundary conditions
    type... field type ('c': volume centred; 'u', 'v', 'w': area centred
    ghst_inds... number of ghost-layers (determined by the spatial order of advection)
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

    bnds_expl_s = []
    bnds_expl_r = []
    
    bnds_expl_cycl_s = []
    bnds_expl_cycl_r = [] 

    if (ind_pc > 0):
    # west
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = ghst_inds
        ir_end = nr_sub - ghst_inds
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([ind_pr == 0, not bnd_y == 'cyclic', type == 'p']):
            ir_st = ir_st - 1
        if all([ind_pr == npr - 1, not bnd_y == 'cyclic', type == 'p']):
            ir_end = ir_end + 1
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - 1]
        tag = pid * rank         
        bnds_expl_s.append(bound(fld, inds, pid, 'w', tag + 1000000))
        jc_st = 0
        jc_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'w', tag + 1300000))
         
    if (ind_pc < npc - 1):
    # east
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = ghst_inds
        ir_end = nr_sub - ghst_inds
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([ind_pr == 0, not bnd_y == 'cyclic', type == 'p']):
            ir_st = ir_st - 1
        if all([ind_pr == npr - 1, not bnd_y == 'cyclic', type == 'p']):
            ir_end = ir_end + 1
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + 1]         
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'e', tag + 1300000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub 
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'e', tag + 1000000))       
    
    if (ind_pr > 0):
    # south
        jc_st = ghst_inds
        jc_end = nc_sub - ghst_inds
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([ind_pc == 0, not bnd_x == 'cyclic', type == 'p']):
            jc_st = jc_st - 1
        if all([ind_pc == npc - 1, not bnd_x == 'cyclic', type == 'p']):
            jc_end = jc_end + 1
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc]
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 's', tag + 1600000))
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 's', tag + 1900000))

    if (ind_pr < npr - 1):
    # north
        jc_st = ghst_inds
        jc_end = nc_sub - ghst_inds
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([ind_pc == 0, not bnd_x == 'cyclic', type == 'p']):
            jc_st = jc_st - 1
        if all([ind_pc == npc - 1, not bnd_x == 'cyclic', type == 'p']):
            jc_end = jc_end + 1
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc]
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'n', tag + 1900000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'n', tag + 1600000))    

    if ind_pc > 0 and ind_pr > 0:
    # south west
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc - 1]
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'sw', tag + 2200000))
        jc_st = 0
        jc_end = ghst_inds
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'sw', tag + 2500000))
    if ind_pc < npc - 1 and ind_pr < npr - 1:
    # north east
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc + 1]
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'ne', tag + 2500000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'ne', tag + 2200000))    
    if ind_pc < npc - 1 and ind_pr > 0:
    # south east
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds - c_x
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - npc + 1]
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'se', tag + 2800000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        ir_st = 0
        ir_end = ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'se', tag + 3100000))
    if ind_pc > 0 and ind_pr < npr - 1:
    # north west
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds - c_y
        k_st = ghst_inds
        k_end = nz_sub - ghst_inds
        if all([not bnd_z == 'cyclic', type == 'p']):
            k_st = k_st - 1
            k_end = k_end + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc - 1] 
        tag = pid * rank
        bnds_expl_s.append(bound(fld, inds, pid, 'nw', tag + 3100000))
        jc_st = 0
        jc_end = ghst_inds
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_r.append(bound(fld, inds, pid, 'nw', tag + 2800000))

    # for cyclic boundaries
    if ind_pc == 0 and bnd_x == 'cyclic':
        jc_st = ghst_inds + c_x
        jc_end = 2 * ghst_inds + c_x
        ir_st = 0
        ir_end = nr_sub
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + npc - 1]
        tag = pid * rank
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 'w', tag + 3200000))
        jc_st = 0
        jc_end = ghst_inds + c_x
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 'w', tag + 3600000))
               
    if ind_pc == npc - 1 and bnd_x == 'cyclic':
        jc_st = nc_sub - 2 * ghst_inds - c_x
        jc_end = nc_sub - ghst_inds
        ir_st = 0
        ir_end = nr_sub
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])]) 
        pid = pids[ind_p - npc + 1]
        tag = pid * rank
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 'e', tag + 3600000))
        jc_st = nc_sub - ghst_inds
        jc_end = nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 'e', tag + 3200000))

    if ind_pr == 0 and bnd_y == 'cyclic':
        jc_st = 0
        jc_end = nc_sub 
        ir_st = ghst_inds + c_y
        ir_end = 2 * ghst_inds + c_y
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p + (npr - 1) * npc]
        tag = pid * rank
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 's', tag + 3700000))
        ir_st = 0
        ir_end = ghst_inds + c_y
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 's', tag + 4000000))

    if ind_pr == npr - 1 and bnd_y == 'cyclic':
        jc_st = 0
        jc_end = nc_sub 
        ir_st = nr_sub - 2 * ghst_inds - c_y
        ir_end = nr_sub - ghst_inds
        k_st = 0
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = pids[ind_p - (npr - 1) * npc]
        tag = pid * rank
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 'n', tag + 4000000))
        ir_st = nr_sub - ghst_inds
        ir_end = nr_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 'n', tag + 3700000))

    if bnd_z == 'cyclic':
        jc_st = 0
        jc_end = nc_sub
        ir_st = 0 
        ir_end = nr_sub 
        k_st = ghst_inds + c_z
        k_end = 2 * ghst_inds + c_z
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = rank
        tag = rank ** 2
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 'b', tag + 4300000))
        k_st = 0
        k_end = ghst_inds + c_z
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 'b', tag + 4600000))   

        k_st = nz_sub - 2 * ghst_inds - c_z
        k_end = nz_sub - ghst_inds
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        pid = rank
        tag = rank ** 2
        bnds_expl_cycl_s.append(bound(fld, inds, pid, 't', tag + 4600000))
        k_st = nz_sub - ghst_inds
        k_end = nz_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        bnds_expl_cycl_r.append(bound(fld, inds, pid, 't', tag + 4300000))    

    return bnds_expl_s, bnds_expl_r, bnds_expl_cycl_s, bnds_expl_cycl_r



def make_bnds_adv_mom(comm, bnd_x, bnd_y, bnd_z, ng):
    """
    This routine constructs the boundary fields
    used by the momentum advection routine.
    Communications are needed to interpolate the cell-centred
    scalar fields to the cell-faces at the subdomain boundaries.
    Only the subdomain boundaries perpendicular to the respective
    velocity component require data exchange (e.g. yz-boundaries for u-component).

    comm... communicator
    bnd_x, bnd_y, bnd_z... type of boundary conditions
    ng... number of ghost-layers (determined by the spatial order of advection)
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

    c_st = ncj[ind_pc]
    c_end = ncj[ind_pc + 1]
    r_st = nri[ind_pr]
    r_end = nri[ind_pr + 1]

    nc_sub = c_end - c_st
    nr_sub = r_end - r_st

    bnds_c_x_s = []
    bnds_c_x_r = []
    bnds_c_x_cycl_s = []
    bnds_c_x_cycl_r = []

    bnds_c_y_s = []
    bnds_c_y_r = []
    bnds_c_y_cycl_s = []
    bnds_c_y_cycl_r = []

    bnds_c_z_s = []
    bnds_c_z_r = []
    bnds_c_z_cycl_s = []
    bnds_c_z_cycl_r = []

    if ind_pc > 0 or (ind_pc == 0 and bnd_x == 'cyclic'):
        k_st = ng
        k_end = nz + ng
        ir_st = ng
        ir_end = ng + nr_sub
        jc_st = ng
        jc_end = ng + 1 
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        if ind_pc == 0 and bnd_x == 'cyclic':
            pid = pids[(ind_pr + 1) * npc - 1]
            tag = pid * rank
            bnds_c_x_cycl_s.append(bound(fld, inds, pid, 'w', tag + 1000000))
            bnds_c_x_cycl_r.append(bound(fld, inds, pid, 'w', tag + 1300000))
        else:
            pid = pids[ind_p - 1]
            tag = pid * rank
            bnds_c_x_s.append(bound(fld, inds, pid, 'w', tag + 1000000))
            bnds_c_x_r.append(bound(fld, inds, pid, 'w', tag + 1300000))

    if ind_pc < npc - 1 or (ind_pc == npc - 1 and bnd_x == 'cyclic'):
        k_st = ng
        k_end = nz + ng
        ir_st = ng
        ir_end = ng + nr_sub
        jc_st = ng + nc_sub
        jc_end = ng + nc_sub + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])          
        if ind_pc == npc - 1 and bnd_x == 'cyclic':
            pid = pids[ind_pr * npc]
            tag = pid * rank
            bnds_c_x_cycl_s.append(bound(fld, inds, pid, 'e', tag + 1300000))
            bnds_c_x_cycl_r.append(bound(fld, inds, pid, 'e', tag + 1000000))
        else:
            pid = pids[ind_p + 1]
            tag = pid * rank
            bnds_c_x_s.append(bound(fld, inds, pid, 'e', tag + 1300000))
            bnds_c_x_r.append(bound(fld, inds, pid, 'e', tag + 1000000))

    if ind_pr > 0 or (ind_pr == 0 and bnd_y == 'cyclic'):
        k_st = ng
        k_end = nz + ng
        ir_st = ng
        ir_end = ng + 1
        jc_st = ng
        jc_end = ng + nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        if ind_pr == 0 and bnd_y == 'cyclic':
            pid = pids[(npr - 1) * npc + ind_pc]
            tag = pid * rank
            bnds_c_y_cycl_s.append(bound(fld, inds, pid, 's', tag + 1600000))
            bnds_c_y_cycl_r.append(bound(fld, inds, pid, 's', tag + 1900000))
        else:
            pid = pids[ind_p - npc]
            tag = pid * rank
            bnds_c_y_s.append(bound(fld, inds, pid, 's', tag + 1600000))
            bnds_c_y_r.append(bound(fld, inds, pid, 's', tag + 1900000))
    
    if ind_pr < npr - 1 or (ind_pr == npr - 1 and bnd_y == 'cyclic'):
        k_st = ng
        k_end = nz + ng
        ir_st = ng + nr_sub
        ir_end = ng + nr_sub + 1
        jc_st = ng
        jc_end = ng + nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        if ind_pr == npr - 1 and bnd_y == 'cyclic':
            pid = pids[ind_pc]
            tag = pid * rank
            bnds_c_y_cycl_s.append(bound(fld, inds, pid, 'n', tag + 1900000))
            bnds_c_y_cycl_r.append(bound(fld, inds, pid, 'n', tag + 1600000))
        else:
            pid = pids[ind_p + npc]
            tag = pid * rank
            bnds_c_y_s.append(bound(fld, inds, pid, 'n', tag + 1900000))
            bnds_c_y_r.append(bound(fld, inds, pid, 'n', tag + 1600000))


    if bnd_z == 'cyclic':
        k_st = ng
        k_end = ng + 1
        ir_st = ng 
        ir_end = ng + nr_sub 
        jc_st = ng
        jc_end = ng + nc_sub
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])]) 
        tag = ind_p ** 2
        bnds_c_z_cycl_s.append(bound(fld, inds, ind_p, 'b', tag + 2200000))
        bnds_c_z_cycl_r.append(bound(fld, inds, ind_p, 'b', tag + 2500000))
        k_st = ng + nz
        k_end = ng + nz + 1
        inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)]).T)
        fld = np.zeros([len(inds[0])])
        tag = ind_p ** 2
        bnds_c_z_cycl_s.append(bound(fld, inds, ind_p, 't', tag + 2500000))
        bnds_c_z_cycl_r.append(bound(fld, inds, ind_p, 't', tag + 2200000))

    bnd_lst_send = [bnds_c_x_s, bnds_c_y_s, bnds_c_z_s, bnds_c_x_cycl_s, bnds_c_y_cycl_s, bnds_c_z_cycl_s]
    bnd_lst_recv = [bnds_c_x_r, bnds_c_y_r, bnds_c_z_r, bnds_c_x_cycl_r, bnds_c_y_cycl_r, bnds_c_z_cycl_r]

    return bnd_lst_send, bnd_lst_recv


def coarse_decomp(y2_fine, x2_fine, y2_coarse, x2_coarse, nri_fine, ncj_fine, npr_fine, npc_fine, pids_f, ncells_min=8):
    """
    This function creates a 2d horizontal coarse-grid domain decomposition
    for the parallel multigrid algorithm as the pressure solver.
    The coarse-grid decomposition can involve a reduced number
    of processors to balance communication overload.

    x2_fine, y2_fine... horizontal edge coordinates of fine grid
    x2_coarse, y2_coarse... horizontal edge coordinates of coarse grid
    nri_fine, ncj_fine... global row/column indices of fine-grid subdomain boundaries
    npr_fine, npc_fine... number of processes in each row/column of fine-grid decomposition
    pids_f... list of process ids of fine-grid decomposition
    ncells_min... minimum average (over all subdomains) number of unknowns per dimension to apply coarsening of respective dimension.
    """ 
    
    nc_fine = x2_fine.size - 1
    nr_fine = y2_fine.size - 1
    nc_coarse = x2_coarse.size - 1
    nr_coarse = y2_coarse.size - 1

    pids_fine_2d = np.array(pids_f).reshape(npr_fine, npc_fine)

    ncells_avg_c = nc_coarse /  npc_fine
    if ncells_avg_c < ncells_min:
        fac_c = 2
    else:
        fac_c = 1
    ncells_avg_r = nr_coarse /  npr_fine
    if ncells_avg_r < ncells_min:
        fac_r = 2
    else:
        fac_r = 1

    pids_coarse_2d = pids_fine_2d[::fac_r, ::fac_c]
    pids_coarse = pids_coarse_2d.flatten().tolist()
    npr_coarse, npc_coarse = pids_coarse_2d.shape[:]
    
    nri_coarse_tmp = nri_fine[::fac_r][:npr_coarse]
    ncj_coarse_tmp = ncj_fine[::fac_c][:npc_coarse]
    nri_coarse_tmp.append(nri_fine[-1])
    ncj_coarse_tmp.append(ncj_fine[-1])

    nri_coarse = [np.argmin(np.absolute(y2_coarse - y2_fine[subdomain_bnd])) for subdomain_bnd in nri_coarse_tmp]
    ncj_coarse = [np.argmin(np.absolute(x2_coarse - x2_fine[subdomain_bnd])) for subdomain_bnd in ncj_coarse_tmp]

    return nri_coarse, ncj_coarse, npr_coarse, npc_coarse, pids_coarse



def subdomain_to_subtomain_galerkin(mpicomm, y2_f, x2_f, y2_c, x2_c, nz_f, nri_f, ncj_f, nz_c, nri_c, ncj_c, pids_f, pids_c):
    """ 
    For a given coarse-grid subdomain
    this function organizes the data transfer between the
    fine-grid and coarse-grid subdomains for the
    derivation of the Galerkin coarse-grid approximation (DCA)
    used in the parallel multigrid algorithm.
    The function returns the necessary objects (bound class and transfer class)
    for data communication/transfer.

    mpicomm... MPI communicator
    x2_f, y2_f... horizontal edge coordinates of fine grid
    x2_c, y2_c... horizontal edge coordinates of coarse grid
    nri_f, ncj_f... global row/column indices of fine-grid subdomain boundaries
    nri_c, ncj_c... global row/column indices of coarse-grid subdomain boundaries
    nz_f, nz_c... number of vertical layers of fine/coarse grid
    pids_f, pids_c... list of process ids of fine/coarse-grid decomposition 
    """

    rank = mpicomm.Get_rank()


    bounds_send = []
    bounds_recv = []
    fld_part_same = None
    field_tmp = None


    if not rank in pids_f:
        # the process is not active at the fine grid
        pass
    else:
        # subdomain of fine grid
        ind_pf = pids_f.index(rank)
        ind_pfr = ind_pf / (len(ncj_f) - 1)
        ind_pfc = ind_pf - ind_pfr * (len(ncj_f) - 1)
        ir_fst = nri_f[ind_pfr]
        ir_fend = nri_f[ind_pfr + 1] + 2
        jc_fst = ncj_f[ind_pfc]
        jc_fend = ncj_f[ind_pfc + 1] + 2
        nr_sub = ir_fend - ir_fst
        nc_sub = jc_fend - jc_fst

        for ind_pc, pid in enumerate(pids_c):
            ind_pcr = ind_pc / (len(ncj_c) - 1)
            ind_pcc = ind_pc - ind_pcr * (len(ncj_c) - 1)

            ir_cst = max(np.argwhere(y2_f == y2_c[nri_c[ind_pcr]])[0][0] - 1, 0)
            ir_cend = min(np.argwhere(y2_f == y2_c[nri_c[ind_pcr + 1] + 2])[0][0] + 1, y2_f.size - 1)
            jc_cst = max(np.argwhere(x2_f == x2_c[ncj_c[ind_pcc]])[0][0] - 1, 0) 
            jc_cend = min(np.argwhere(x2_f == x2_c[ncj_c[ind_pcc + 1] + 2])[0][0] + 1, x2_f.size - 1)

            outside = [ir_cst > ir_fend, ir_cend < ir_fst, jc_cst > jc_fend, jc_cend < jc_fst]

            if any(outside):
                continue

            ir_st = max(ir_cst, ir_fst) - ir_fst
            ir_end = min(ir_cend, ir_fend) - ir_fst
            jc_st = max(jc_cst, jc_fst) - jc_fst
            jc_end = min(jc_cend, jc_fend) - jc_fst
            k_st = 0
            k_end = nz_f

            fld1 = np.zeros([nz_f, ir_cend - ir_cst, jc_cend - jc_cst])

            fld1[:, max(ir_cst, ir_fst) - ir_cst:min(ir_cend, ir_fend) - ir_cst, max(jc_cst, jc_fst) - jc_cst:min(jc_cend, jc_fend) - jc_cst] = 1.0
            EXP_OP = ops.make_expansion_op(fld1.flatten())

            fld2 = np.zeros([nz_f, ir_fend - ir_fst, jc_fend - jc_fst])
            fld2[:, ir_st:ir_end, jc_st:jc_end] = 1.0
            EXP_OP = EXP_OP * (ops.make_expansion_op(fld2.flatten())).transpose()

            # subdomain indices of shared field part
            ir_st = max(ir_cst, ir_fst) - ir_fst
            ir_end = min(ir_cend, ir_fend) - ir_fst
            jc_st = max(jc_cst, jc_fst) - jc_fst
            jc_end = min(jc_cend, jc_fend) - jc_fst
            k_st = 0
            k_end = nz_f

            inds1 = [np.array([k * nr_sub * nc_sub +  i * nc_sub +  j for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)])]
            fld = np.array([None  for n in inds1[0]])

            if not rank == pid:
                tag = pid * rank + 100000
                bounds_send.append(bound(fld, inds1, pid, None, tag))
                bounds_send[-1].add_op(EXP_OP)
            else:
                inds1_tmp = inds1
                EXP_OP_same = EXP_OP
    if not rank in pids_c:
        # processor is not active on the coarse grid
        pass
    else:
        ind_pc = pids_c.index(rank)
        ind_pcr = ind_pc / (len(ncj_c) - 1)
        ind_pcc = ind_pc - ind_pcr * (len(ncj_c) - 1)

        ir_cst = max(np.argwhere(y2_f == y2_c[nri_c[ind_pcr]])[0][0], - 1, 0)
        ir_cend = min(np.argwhere(y2_f == y2_c[nri_c[ind_pcr + 1] + 2])[0][0] + 1, y2_f.size - 1)
        jc_cst = max(np.argwhere(x2_f == x2_c[ncj_c[ind_pcc]])[0][0] - 1, 0)
        jc_cend = min(np.argwhere(x2_f == x2_c[ncj_c[ind_pcc + 1] + 2])[0][0] + 1, x2_f.size - 1) 

        nr_sub = ir_cend - ir_cst
        nc_sub = jc_cend - jc_cst

  
        for ind_pf, pid in enumerate(pids_f):

            ind_pfr = ind_pf / (len(ncj_f) - 1)
            ind_pfc = ind_pf - ind_pfr * (len(ncj_f) - 1)
            ir_fst = nri_f[ind_pfr]
            ir_fend = nri_f[ind_pfr + 1] + 2
            jc_fst = ncj_f[ind_pfc] 
            jc_fend = ncj_f[ind_pfc + 1] + 2

            outside = [ir_cst > ir_fend, ir_cend < ir_fst, jc_cst > jc_fend, jc_cend < jc_fst]
            if any(outside):
                continue

            # boundary indices of field parts
            ir_st = max(ir_fst, ir_cst) - ir_cst
            ir_end = min(ir_fend, ir_cend) - ir_cst
            jc_st = max(jc_fst, jc_cst) - jc_cst
            jc_end = min(jc_fend, jc_cend) - jc_cst
            k_st = 0
            k_end = nz_f

            inds2 = [np.array([k * nr_sub * nc_sub +  i * nc_sub +  j for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)])]
            fld = np.array([None for n in inds2[0]])
            if rank == pid:
                fld_part_same = transfer(fld, inds1_tmp, inds2)
                fld_part_same.add_op(EXP_OP_same)
            else:
                tag = pid * rank + 100000
                bounds_recv.append(bound(fld, inds2, pid, None, tag))

    return bounds_send, bounds_recv, fld_part_same



def org_subdomains(mpicomm, y2_f, x2_f, y2_c, x2_c, nri_f, ncj_f, nz_f, nz_c, nri_c, ncj_c, pids_f, pids_c):
    """ 
    For a given coarse-grid subdomain
    this function organizes the data transfer between the
    fine-grid and coarse-grid subdomains for the
    restriction/prolongation operation in the parallel multigrid
    algorithm.
    The function returns the necessary objects (bound class and transfer class)
    for data communication/transfer.

    mpicomm... MPI communicator
    x2_f, y2_f... horizontal edge coordinates of fine grid
    x2_c, y2_c... horizontal edge coordinates of coarse grid
    nri_f, ncj_f... global row/column indices of fine-grid subdomain boundaries
    nri_c, ncj_c... global row/column indices of coarse-grid subdomain boundaries
    nz_f, nz_c... number of vertical layers of fine/coarse grid
    pids_f, pids_c... list of process ids of fine/coarse-grid decomposition 
    """

    rank = mpicomm.Get_rank()

    # organize data transfer necessary before prolongation
    bounds_prol_send = []
    bounds_prol_recv = []
    fld_part_prol_same = None
    field_prol_tmp = None
    dom_bnds_prol = None

    x_c = 0.5 * (x2_c[1:] + x2_c[:-1])
    y_c = 0.5 * (y2_c[1:] + y2_c[:-1])
    x_f = 0.5 * (x2_f[1:] + x2_f[:-1])
    y_f = 0.5 * (y2_f[1:] + y2_f[:-1])

    if not rank in pids_c:
        # the process is not active on the coarse grid
        pass
    else:
        # subdomain of coarse grid
        ind_pc = pids_c.index(rank)
        ind_pcr = ind_pc / (len(ncj_c) - 1)
        ind_pcc = ind_pc - ind_pcr * (len(ncj_c) - 1)
        ir_cst = nri_c[ind_pcr]
        ir_cend = nri_c[ind_pcr + 1]
        jc_cst = ncj_c[ind_pcc]
        jc_cend = ncj_c[ind_pcc + 1]
        nr_sub = ir_cend - ir_cst
        nc_sub = jc_cend - jc_cst
        
        for ind_pf, pid in enumerate(pids_f):
            ind_pfr = ind_pf / (len(ncj_f) - 1)
            ind_pfc = ind_pf - ind_pfr * (len(ncj_f) - 1)
            
	    ir_fst = np.argwhere(y_c < y_f[nri_f[ind_pfr]])
            if not len(ir_fst):
                 ir_fst = 0
            else:
                 ir_fst = ir_fst[-1][0]                 
            ir_fend = np.argwhere(y_c > y_f[nri_f[ind_pfr + 1] - 1]) 
            if not len(ir_fend):
                 ir_fend = y_f.size
            else:
                 ir_fend = ir_fend[0][0] + 1


            jc_fst = np.argwhere(x_c < x_f[ncj_f[ind_pfc]])
            if not len(jc_fst):
                 jc_fst = 0
            else:
                 jc_fst = jc_fst[-1][0]

            jc_fend = np.argwhere(x_c > x_f[ncj_f[ind_pfc + 1] - 1])
            if not len(jc_fend):
                 jc_fend = x_f.size
            else:
                 jc_fend = jc_fend[0][0] + 1
            outside = [ir_cst >= ir_fend, ir_cend <= ir_fst, jc_cst >= jc_fend, jc_cend <= jc_fst]
            if any(outside) and not rank == pid:
                continue

        # subdomain indices of shared field part
            ir_st = max(ir_fst, ir_cst) - ir_cst
            ir_end = min(ir_fend, ir_cend) - ir_cst
            jc_st = max(jc_fst, jc_cst) - jc_cst
            jc_end = min(jc_fend, jc_cend) - jc_cst
            k_st = 0
            k_end = nz_c

            inds1 = [np.array([k * nr_sub * nc_sub +  i * nc_sub +  j for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)])]
            fld = np.zeros([len(inds1[0])])

            if not rank == pid:
                tag = pid * rank + 1000000
                bounds_prol_send.append(bound(fld, inds1, pid, None, tag))
            else:
                inds1_tmp = inds1

    if not rank in pids_f:
        # processor is not active on the coarse grid
        pass
    else:
        ind_pf = pids_f.index(rank)
        ind_pfr = ind_pf / (len(ncj_f) - 1)
        ind_pfc = ind_pf - ind_pfr * (len(ncj_f) - 1)

        ir_fst = np.argwhere(y_c <  y_f[nri_f[ind_pfr]])
        if not len(ir_fst):
            ir_fst = 0
        else:            
            ir_fst = ir_fst[-1][0]
        ir_fend = np.argwhere(y_c > y_f[nri_f[ind_pfr + 1] - 1])
        if not len(ir_fend):
            ir_fend = y_c.size
        else:            
            ir_fend = ir_fend[0][0] + 1 
        jc_fst = np.argwhere(x_c <  x_f[ncj_f[ind_pfc]])
        if not len(jc_fst):
            jc_fst = 0
        else:
            jc_fst = jc_fst[-1][0]
        jc_fend = np.argwhere(x_c > x_f[ncj_f[ind_pfc + 1] - 1])
        if not len(jc_fend):
            jc_fend = x_c.size
        else:
            jc_fend = jc_fend[0][0] + 1

        dom_bnds_prol = [[(ir_fst, ir_fend), (jc_fst, jc_fend)], [(nri_f[ind_pfr], nri_f[ind_pfr + 1]), (ncj_f[ind_pfc], ncj_f[ind_pfc + 1])]]
        nr_sub = ir_fend - ir_fst
        nc_sub = jc_fend - jc_fst

        field_prol_tmp = np.empty([nz_c, nr_sub, nc_sub])

#    fld_part_same = None
        for ind_pc, pid in enumerate(pids_c):

            ind_pcr = ind_pc / (len(ncj_c) - 1)
            ind_pcc = ind_pc - ind_pcr * (len(ncj_c) - 1)
            ir_cst = nri_c[ind_pcr]
            ir_cend = nri_c[ind_pcr + 1]
            jc_cst = ncj_c[ind_pcc]
            jc_cend = ncj_c[ind_pcc + 1]

            outside = [ir_fst >= ir_cend, ir_fend <= ir_cst, jc_fst >= jc_cend, jc_fend <= jc_cst]

            if any(outside) and not rank == pid:
                continue
            
            # boundary indices of field parts
            ir_st = max(ir_cst, ir_fst) - ir_fst            
            ir_end = min(ir_cend, ir_fend) - ir_fst
            jc_st = max(jc_cst, jc_fst) - jc_fst
            jc_end = min(jc_cend, jc_fend) - jc_fst
            
            k_st = 0
            k_end = nz_c

            inds2 = [np.array([k * nr_sub * nc_sub +  i * nc_sub +  j for k in range(k_st, k_end) for i in range(ir_st, ir_end) for j in range(jc_st, jc_end)])]
            fld = np.zeros([len(inds2[0])])

            if rank == pid:
                fld_part_prol_same = transfer(fld, inds1_tmp, inds2)
            else:
                tag = pid * rank + 1000000
                bounds_prol_recv.append(bound(fld, inds2, pid, None, tag))


    return_lists = [
                        
                        bounds_prol_send, bounds_prol_recv, fld_part_prol_same, field_prol_tmp,
                        dom_bnds_prol
                   ]        
                        
    return return_lists
           

def distribute_data(mpicomm, field, field_sub, nri_st, nri_end, ncj_st, ncj_end, pids):
    """
    Distribution of a full field
    from root processor (node=0)
    to the subdomains.

    mpicomm... MPI communicator
    field... full field on root node
    field_sub... local subdomain field (still empty)
    nri_st... global row indices of lower boundary of subdomains
    nri_end... global row indices of upper boudnary of subdomains
    ncj_st... global column indices of lower boundary of subdomains
    ncj_end... global column indices of upper boudnary of subdomains
    pids... list of process ids
    """
    
    rank = mpicomm.Get_rank()

    shp_sub = field_sub.shape[:]
    nr_sub, nc_sub = shp_sub[1], shp_sub[2]

    gr = nr_sub - (nri_end[0] - nri_st[0])
    gc = nc_sub - (ncj_end[0] - ncj_st[0])
        
    if rank == 0:
        count = 0
        for i in range(len(nri_st)):
            for j in range(len(ncj_st)):   
                if i == 0 and j == 0:
                    field_sub[:] =  field[:, :nr_sub , :nc_sub]
                else:
                    bf_send = field[:, nri_st[i]:nri_end[i] + gr, ncj_st[j]:ncj_end[j] + gc]
                    mpicomm.send(bf_send, dest=pids[count])
                count += 1
    else:
        field_sub[:] = mpicomm.recv(source=0)           


def gather_data(mpicomm, field, field_sub, nri, ncj, k, pids, type='c'):
    """
    Reverse of distribute_data: 
    Gathering of all subdomain fields on root 
    node to form the full field.

    mpicomm... MPI communicator
    field... full field (still empty) on root node
    field_sub... local subdomain field 
    nri... row indices of subdomain boundaries
    ncj... column indices of subdomain boundaries
    k... vertical level index to gather
    pids... list of process ids
    type... field type ('c': volume centred; 'u', 'v', 'w': area centred
    """

    rank = mpicomm.Get_rank()

    c_x = 0
    c_y = 0
    c_z = 0

    if type == 'u':
        c_x = 1
    if type == 'v':
        c_y = 1

    if rank == 0:
          field[k, :nri[1] + c_y, :ncj[1] + c_x] = field_sub

    for n, pid in enumerate(pids[1:]):
        i = pid / (len(ncj) - 1)
        j = pid - i * (len(ncj) - 1)
        if rank == pid:
            mpicomm.send(field_sub, dest=0)
        if rank == 0:

            recv = mpicomm.recv(source=pid)
            n2 = nri[i + 1] + c_y - nri[i]
            n3 = ncj[j + 1] + c_x - ncj[j]
            size = n2 * n3
            field[k, nri[i]:nri[i + 1] + c_y, ncj[j]:ncj[j + 1] + c_x] = recv[:size].reshape(1, n2, n3)


            
def scatter_point(mpicomm, objs, pids, wait=False):
    """
    Scatters a list of buffer objects to
    all working nodes.
    Objs is a list of buffer objects with the
    same size as nnodes
    The sending is non-blocking.
    """

    n = 0
    reqs = []
    for pid in pids:
        if pid != 0:            
            reqs.append(mpicomm.Isend(objs[n], dest=pid))            
            n += 1
    if wait:
        for req in reqs:
            req.Wait()

def gather_point(mpicomm, objs, pids):
    """
    Collects a list of buffer objects from
    all working nodes.

    mpicomm... MPI communicator
    objs... list of buffer objects
    pids... list of process ids
    """

    reqs = []
    n = 0
    for id in pids:
        if id != 0:
           reqs.append(mpicomm.Irecv(objs[n], source = id))
           n += 1
    for n, req in enumerate(reqs):
        req.wait()
         
              

def exchange_fields(mpicomm, bnd_send, bnd_recv):
    """
    This function exchanges the data located on objects 
    of bound class.

    mpicomm... MPI communicator
    bnd_send... list of bounds containing data to send
    bnd_recv... list of bounds to store received data
    """

    req_send = []
    req_recv = []

    rank = mpicomm.Get_rank()

    for bnd in bnd_send:
        req = mpicomm.Isend([bnd.data, MPI.FLOAT], dest=bnd.pid, tag=bnd.tag) 
        req_send.append(req)

    for bnd in bnd_recv:     
        req = mpicomm.Irecv([bnd.data, MPI.FLOAT], source=bnd.pid, tag=bnd.tag) 
        req_recv.append(req)    

    for req in req_recv:       
        req.wait() 
 
    for req in req_send:
        req.wait()


def gather_galerkin_coarse_op(mpicomm, OP_current, obj_send, obj_same, obj_recv):
    """
    This function is needed to construct the Galerkin coarse-grid operator.
    For each coarse-grid subdomain, the operator is derived by applying
    interpolation matrices on a congruent fine-grid operator. 
    The fine-grid operator is collected from operator parts located
    on various processing nodes of the next finer level.
    
    OP_current ... the fine-grid oeprator on the current fine-grid subdomain
    obj_send ... communication object to send parts of OP_current to other processing nodes
    obj_same ... object to transfer the operator part that does not need to be communicated, because it is used on the same processing node again
    obj_recv ... communication object to receive parts of OP_current from other processing nodes
    OP_fine ... The complete fine-grid operator that is used in the Galerkin coarsening:  OP_coarse = REST * OP_fine * PROL
    """

    OP_lst_send_data = []
    OP_lst_send_indices = []
    OP_lst_send_indptr = []

    OP_lst_recv_data = []
    OP_lst_recv_indices = []
    OP_lst_recv_indptr = []

    OP_to_send_lst = []   

    if isinstance(obj_same, transfer):
        # obj_same.op is an operator to "cut out" the required part of the whole operator and puts it in the m x m shape of the final operator
        OP_fine = obj_same.op * OP_current * obj_same.op.transpose()
        size = OP_fine.shape[0]

    else:
        OP_fine = None

    OP_lst_recv_data = [np.empty([size * 25], dtype=np.float32) for obj in obj_recv]
    OP_lst_recv_indices = [np.full([size * 25], -1, dtype=np.int32) for obj in obj_recv]
    OP_lst_recv_indptr = [np.full([size + 1], -1, dtype=np.int32) for obj in obj_recv]    

    for n, obj in enumerate(obj_send):
        OP_to_send = obj.op * OP_current * obj.op.transpose()
        
        OP_to_send_lst.append(OP_to_send)

    reqs_send = []
    reqs_recv = []

    for n, obj in enumerate(obj_send):
        data_to_send = np.array(OP_to_send_lst[n].data, dtype=np.float32)
        req = mpicomm.Isend([data_to_send, data_to_send.size, MPI.FLOAT], dest=obj.pid, tag=obj.tag)
        reqs_send.append(req)

    for n, obj in enumerate(obj_recv):        
        req = mpicomm.Irecv([OP_lst_recv_data[n], OP_lst_recv_data[n].size, MPI.FLOAT], source=obj.pid, tag=obj.tag)
        reqs_recv.append(req)

    for req in reqs_recv:
        req.wait()

    for req in reqs_send:
        req.wait()

    reqs_send = []
    reqs_recv = []

    for n, obj in enumerate(obj_send):
        data_to_send = np.array(OP_to_send_lst[n].indices, dtype=np.int32)
        req = mpicomm.Isend([data_to_send, data_to_send.size, MPI.INT], dest=obj.pid, tag=obj.tag)
        reqs_send.append(req)

    for n, obj in enumerate(obj_recv):
        OP_lst_recv_indices[n].fill(-1)
        req = mpicomm.Irecv([OP_lst_recv_indices[n], OP_lst_recv_indices[n].size, MPI.INT], source=obj.pid, tag=obj.tag)
        reqs_recv.append(req)

    for req in reqs_recv:
        req.wait()

    for req in reqs_send:
        req.wait()

    reqs_send = []
    reqs_recv = []

    for n, obj in enumerate(obj_send):
        data_to_send = np.array(OP_to_send_lst[n].indptr, dtype=np.int32)
        
        req = mpicomm.Isend([data_to_send, data_to_send.size, MPI.INT], dest=obj.pid, tag=obj.tag)
        reqs_send.append(req)

    for n, obj in enumerate(obj_recv):
        OP_lst_recv_indptr[n].fill(-1)
        req = mpicomm.Irecv([OP_lst_recv_indptr[n], OP_lst_recv_indptr[n].size, MPI.INT], source=obj.pid, tag=obj.tag)
        reqs_recv.append(req)

    for req in reqs_recv:
        req.wait()

    for req in reqs_send:
        req.wait()


    for n, indices in enumerate(OP_lst_recv_indices):
        indices_true = indices[indices >= 0]                
        data = OP_lst_recv_data[n][indices >= 0]
        indptr = OP_lst_recv_indptr[n]
        indptr_true = indptr[indptr >= 0]
        OP_part_recv = csr_matrix((data, indices_true, indptr_true), OP_fine.shape)        

        # merge operator parts
        OP_fine = ops.merge_operators(OP_fine, OP_part_recv)

    return OP_fine


 

def cptobounds(field, bounds, mode='repl'):
    """
    Copies data to a boundary field.

    field... field from which data is copied
    bounds... list of bound objects
    mode... determines whether data on bound is overwritten, added, or subtracted
    """

    if mode == 'repl':
        for bound in bounds:
            bound.update(field[bound.inds]) 
    elif mode == 'add':
        for bound in bounds:
            bound.data[:] += field[bound.inds] 
    elif mode == 'sub':
        for bound in bounds:
            bound.data[:] -= field[bound.inds]         


def cpfrombounds(field, bounds, mode='add'):
    """
    Inserts data into a field from a bound object.

    field... field into which data is inserted
    bounds... list of bound objects
    mode... determines whether field data is overwritten, added, or subtracted
    """
  
    if mode == 'add': 
        for bound in bounds:
            field[bound.inds] += bound.data
    elif mode == 'sub':
        for bound in bounds:
            field[bound.inds] -= bound.data
    elif mode == 'mul':
        for bound in bounds:
            field[bound.inds] *= bound.data
    elif mode == 'repl':
        for bound in bounds:
            field[bound.inds] = bound.data           
       


def cpbound2bound(bounds1, bounds2):
    """
    Special function to copy the data from one list
    of bound objects to another list of bound objects 
    with different shape.

    bounds1... list of bound objects to copy data from
    bounds2... list of bound objects to copy data to
    """

    for n, bound in enumerate(bounds2):
        inds = bound.inds
        shape2 = bound.shape
        shape1 = bounds1[n].shape 
        bounds1[n].data.reshape(shape1)[inds] = bound.data


def shiftdata(bounds):
    """
    Special function used in the turbulence recycling 
    scheme to shift (advance) the data on bound one plane foreward
    each time step.

    bounds... list of bound objects
    """

    for bound in bounds:        
        fld = bound.data.reshape(bound.shape)
        if bound.side == 'w':
            n_planes = fld.shape[2]
            for n in range(n_planes - 1):
                fld[:, :, -1 - n] = fld[:, :, -2 - n]            
            bound.update(fld.flatten())
        elif bound.side == 'e':
            n_planes = fld.shape[2]
            for n in range(n_planes - 1):
                fld[:, :, n] = fld[:, :, n + 1]
            bound.update(fld.flatten())
        elif bound.side == 's':
            n_planes = fld.shape[1]
            for n in range(n_planes - 1):
                fld[:, -1 - n] = fld[:, -2 - n]
            bound.update(fld.flatten())
        elif bound.side == 'n':
            n_planes = fld.shape[1]
            for n in range(n_planes - 1):
                fld[:, n] = fld[:, n + 1]
            bound.update(fld.flatten())

def dot_para(comm, a, b):
    """
    Computes the dot product of distributed vectors
    using root collect (instead of all to all)

    comm... communicator
    a... vector a
    b... vector b
    """
   
    
    mpicomm = comm.mpicomm
    pids = comm.pids
    rank = mpicomm.Get_rank()
    size = mpicomm.Get_size()
   
    dot_loc = np.dot(a, b)    
    dot_loc_vec = [np.empty([1]) for n in range(size)]
    if rank == 0:
        buffs = [np.zeros([1]) for p in range(size - 1)]
        dot_loc_vec[0] = dot_loc
        gather_point(mpicomm, dot_loc_vec[1:], pids)        
        dotp = sum(np.array(dot_loc_vec))
        reqs = []
        buffs = [dotp for buf in list(buffs)]
        scatter_point(mpicomm, buffs, pids[1:])
    else:

        req = mpicomm.Isend(dot_loc, dest=0)
        dotp = np.array(dot_loc)
        req.wait()
        req = mpicomm.Irecv(dotp, source=0)
        req.wait()
    mpicomm.barrier()
    return dotp


def min_para(mpicomm, numb, child_ids, root_id=0):
    """
    Parallel minimum over an ensemble of processes.
    Communication is via root child mode.    

    mpicomm... MPI communicator
    numb... numerical value
    child_ids... list of child process ids
    root_id... root process id
    """

    rank = mpicomm.Get_rank()
    numb = np.full([1], numb)
    glob_min = numb

    if rank == root_id:
        buffs = [np.full([1], numb) for id in child_ids]
        reqs = []
        for k, id in enumerate(child_ids):
            reqs.append(mpicomm.Irecv(buffs[k], source=id))
        received = [0 for id in child_ids]
        while not all(received):
            for k, req in enumerate(reqs):
                if not received[k]:
                    received[k] = req.test()[0]
                else:
                    continue
                if received[k]:                    
                    glob_min = min([buffs[k][0], glob_min])

        reqs = []
        for id in child_ids:
            reqs.append(mpicomm.Isend(np.full([1], glob_min), dest = id))
        for req in reqs:
            req.wait()
    elif rank in child_ids:
         reqs = mpicomm.Isend(numb, dest=root_id)
         glob_min = numb.copy()
         reqr = mpicomm.Irecv(glob_min, source=root_id)
         reqr.wait()
         glob_min = glob_min[0]
    return glob_min


def max_para(mpicomm, numb, child_ids, root_id=0):
    """
    Parallel maximum over an ensemble of processes.
    Communication is via root child mode.    

    mpicomm... MPI communicator
    numb... numerical value
    child_ids... list of child process ids
    root_id... root process id
    """

    rank = mpicomm.Get_rank()
    numb = np.full([1], numb)
    glob_max = numb

    if rank == root_id:
        buffs = [np.full([1], numb) for id in child_ids]
        reqs = []
        for k, id in enumerate(child_ids):
            reqs.append(mpicomm.Irecv(buffs[k], source=id))
        received = [0 for id in child_ids]
        while not all(received):
            for k, req in enumerate(reqs):
                if not received[k]:
                    received[k] = req.test()[0]
                else:
                    continue
                if received[k]:
                    glob_max = max([buffs[k][0], glob_max])

        reqs = []
        for id in child_ids:
            reqs.append(mpicomm.Isend(np.full([1], glob_max), dest = id))
        for req in reqs:
            req.wait()
    elif rank in child_ids:
         reqs = mpicomm.Isend(numb, dest=root_id)
         glob_max = numb.copy()
         reqr = mpicomm.Irecv(glob_max, source=root_id)
         reqr.wait()
         glob_max = glob_max[0]
    return glob_max


def sum_para(mpicomm, numb, child_ids, root_id=0):
    """
    Parallel sumation over an ensemble of processes.
    Communication is via root child mode.    

    mpicomm... MPI communicator
    numb... numerical values
    child_ids... list of child process ids
    root_id... root process id
    """

    
    rank = mpicomm.Get_rank()    
    numb = np.array(numb)
    glob_sum = numb

    if rank == root_id:
        buffs = [np.array(numb) for id in child_ids]
        reqs = []
        for k, id in enumerate(child_ids):
            reqs.append(mpicomm.Irecv(buffs[k], source=id))
        received = [0 for id in child_ids]
        while not all(received):
            for k, req in enumerate(reqs):
                if not received[k]:
                    received[k] = req.test()[0]
                else:
                    continue
                if received[k]:
                    glob_sum += buffs[k]

        reqs = []
        for id in child_ids:
            reqs.append(mpicomm.Isend(glob_sum, dest = id))
        for req in reqs:
            req.wait()
    elif rank in child_ids:
         reqs = mpicomm.Isend(numb, dest=root_id)
         glob_sum = numb.copy()
         reqr = mpicomm.Irecv(glob_sum, source=root_id)
         reqr.wait()
    
    return glob_sum


def fld_avg_para(mpicomm, field, pids_child, pid_root, axis):
    """
    Parallel field average  over an ensemble of processes.
    Communication is via root child mode.    

    comm... MPI communicator
    field... multidimensional numpy array
    axis... tuple of axes to average
    root_id... root process id
    """

    shape = list(field.shape)
    for ax in axis:
        shape[ax] = 1

    fld_sum = (np.sum(field, axis=axis)).reshape(shape)

    fld_sum_glob = sum_para(mpicomm, fld_sum, pids_child, pid_root)

    axis_size = 1
    for ax in axis:
        axis_size = axis_size * field.shape[ax]
    
    axis_size_glob = sum_para(mpicomm, axis_size, pids_child, pid_root)
    
    fld_mean_glob = fld_sum_glob / axis_size_glob 
    
    return fld_mean_glob 


def fld_rms_para(mpicomm, field, pids_child, pid_root, axis):
    """
    Parallel field root mean square over an ensemble of processes.
    Communication is via root child mode.    

    comm... MPI communicator
    field... multidimensional numpy array
    axis... tuple of axes to average
    root_id... root process id
    """

    avg_para = fld_avg_para(mpicomm, field, pids_child, pid_root, axis)
    rms = np.sqrt(fld_avg_para(mpicomm, (field - avg_para) ** 2, pids_child, pid_root, axis))

    return rms
