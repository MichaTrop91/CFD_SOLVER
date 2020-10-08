# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020


# load external python packages
import numpy as np
from netCDF4 import Dataset
from scipy.sparse import csr_matrix

# load model specific *py files
import domain_decomp as ddcp

# This module contains a plane 2d-filter operator 
# using domain decompostion
# It is used to filter velocity slices for the turbulence recycling scheme and
# for the outflow boundary conditions

def setup_plane_filter(comm, orient, ind_p_filt, param_dict, tag_ref, type='c'):
    """
    This routine creates the boundary objects for communication, the temporary
    fields, and the filter itself in csr_matrix format.

    comm... communicator
    orient... orientation of filter plane ('we', 'sn')
    ind_p_filt... process row/column id containing the filter plane
    param_dict... parameter dictionary
    tag_ref... tag base
    type... field type ('c':centred, 'sv':staggered v, 'sw':staggered w)
    """

    global ng
    ng = int(param_dict['n_ghost'])

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
    nc_sub = ncj[ind_pc + 1] - ncj[ind_pc] + 2 * ng
    nr_sub = nri[ind_pr + 1] - nri[ind_pr] + 2 * ng
    nz_sub = nz

    if type == 'sw':
        nz_sub += 1

    dh = param_dict['dh']

    xcoord = param_dict['xcoord']
    ycoord = param_dict['ycoord']
    x2coord = param_dict['x2coord']
    y2coord = param_dict['y2coord']
   
    lscal_h = param_dict['turb_lscale_h']
    nh_cutoff = int(lscal_h / dh) + 4

    if nh_cutoff % 2 ==0:
        nh_cutoff += 1 

    wh_plane = int((nh_cutoff - 1) / 2)
    n_plane = 1

    bnds_filter_send = []
    bnds_filter_recv = []

    k_st = 0
    k_end = nz_sub

    if ind_pc == ind_p_filt and orient == 'we':

        bnd = param_dict['bnd_yl']
        if not bnd == 'cyclic':

            ir_glob_st_l = max(0, nri[ind_pr] - wh_plane - ng)
            ir_glob_st_r = min(nri[ind_pr + 1], nri[-1])
            ir_glob_end_l = nri[ind_pr]
            ir_glob_end_r = min(nri[-1], nri[ind_pr + 1] + wh_plane + ng)
            loc_st_r = ir_glob_st_r - ir_glob_st_l
            len_fld_h = ir_glob_end_r - ir_glob_st_l
            len_bnd_l = ir_glob_end_l - ir_glob_st_l
            len_bnd_r = ir_glob_end_r - ir_glob_st_r

            flt_out_l = max(len_bnd_l, 0)
            flt_out_h = min(len_bnd_l - 2 * ng + nr_sub, len_fld_h)
 
            if type == 'sv':
                ind_tpl_same = (nri[ind_pr] - ir_glob_st_l, nri[ind_pr + 1] - ir_glob_st_l + 1)
            else:
                ind_tpl_same = (ir_glob_end_l - ir_glob_st_l, ir_glob_st_r - ir_glob_st_l)

            if type == 'sv':
                fld_tmp = np.zeros([nz_sub, len_fld_h + 1, n_plane])
            else:
                fld_tmp = np.zeros([nz_sub, len_fld_h, n_plane])

            if type == 'sv':
                shp_fld_tmp = ((nz_sub, nr_sub - 2 * ng + 1, n_plane), (0, nr_sub - 2 * ng + 1)) 
            else:
                shp_fld_tmp = ((nz_sub, nr_sub - 2 * ng, n_plane), (ng, nr_sub - 2 * ng))

            if type == 'sv':
                y2 = y2coord[ir_glob_st_l:ir_glob_end_r + 1]
            else:
                y = ycoord[ir_glob_st_l:ir_glob_end_r]

            if type == 'c':  
                filter_2d = make_filter_plane((0, len_fld_h), (flt_out_l, flt_out_h), (0, nz_sub), lscal_h, y)
            elif type == 'sv':
                filter_2d = make_filter_plane((0, len_fld_h + 1), (flt_out_l, flt_out_h + 1), (0, nz_sub), lscal_h, y2)

            elif type == 'sw':
                filter_2d = make_filter_plane((0, len_fld_h), (flt_out_l, flt_out_h), (0, nz_sub), lscal_h, y)
        else:

            ir_glob_st_l = nri[ind_pr] - wh_plane - ng
            ir_glob_st_r = nri[ind_pr + 1]
            loc_st_r = ir_glob_st_r - ir_glob_st_l
            len_fld_h = 2 * wh_plane + nr_sub
            len_bnd_l = wh_plane + ng
            len_bnd_r = wh_plane + ng

            if type == 'sv':
                 ind_tpl_same = (nri[ind_pr] - ir_glob_st_l, nri[ind_pr + 1] - ir_glob_st_l + 1)
            else:
                 ind_tpl_same = (nri[ind_pr] - ir_glob_st_l, nri[ind_pr + 1] - ir_glob_st_l)

            if type == 'sv':
                fld_tmp = np.zeros([nz_sub, nr_sub + 1 + 2 * wh_plane, n_plane])
                shp_fld_tmp = ((nz_sub, nr_sub + 1, n_plane), (0, nr_sub + 1))
            else:
                fld_tmp = np.zeros([nz_sub, nr_sub + 2 * wh_plane, n_plane])
                shp_fld_tmp = ((nz_sub, nr_sub, n_plane), (0, nr_sub))

            size_y = ycoord.size

            if type == 'sv':
                y2coord_tmp = y2coord.tolist()
                y2coord_tmp = y2coord.tolist() + (y2coord[1:] + y2coord[-1] - y2coord[0]).tolist() + (y2coord[1:] + 2 * (y2coord[-1] - y2coord[0])).tolist()
                y2 = y2coord_tmp[ir_glob_st_l + size_y:ir_glob_st_l + len_fld_h + 1 + size_y]
                y2  = np.array(y2)
            else:  
                ycoord_tmp = ycoord.tolist()
                ycoord_tmp = ycoord.tolist() + (ycoord[1:] + ycoord[-1] - ycoord[0]).tolist() + (ycoord[1:] + 2 * (ycoord[-1] - ycoord[0])).tolist()
                y = ycoord_tmp[ir_glob_st_l + size_y:ir_glob_st_l + len_fld_h + size_y]
                y = np.array(y)

            if type == 'c':
                filter_2d = make_filter_plane((0, len_fld_h), (wh_plane + ng, wh_plane - ng + nr_sub), (0, nz_sub), lscal_h, y)
            elif type == 'sv':
                filter_2d = make_filter_plane((0, len_fld_h + 1), (wh_plane + ng, wh_plane - ng + nr_sub + 1), (0, nz_sub), lscal_h, y2)
            elif type == 'sw':
                filter_2d = make_filter_plane((0, len_fld_h), (wh_plane + ng, wh_plane - ng + nr_sub), (0, nz_sub), lscal_h, y)

        jc_st_pl = 0
        jc_end_pl = n_plane

        if len_bnd_l > 0:
            
            list_c = org_parts_comm(nri, ir_glob_st_l, 0, len_bnd_l, wh_plane, type=type, bnd=bnd)

            for obj in list_c:
                pid_r, tpl1, tpl2 = obj[:]
                ir_st_recv, ir_end_recv = tpl1[:]
                ir_st_send, ir_end_send = tpl2[:]
                pid = pid_r * npc + ind_p_filt
                tag = (ind_p + 20) * (pid + 20)

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_recv, ir_end_recv) for j in range(jc_st_pl, jc_end_pl)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_recv.append(ddcp.bound(fld_part, inds, pid, 's', tag_ref + tag))

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_send, ir_end_send) for j in range(jc_st_pl, jc_end_pl)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_send.append(ddcp.bound(fld_part, inds, pid, 'n', tag_ref + 10000 + tag))

        if len_bnd_r > 0:
            if type == 'sv':
                list_c = org_parts_comm(nri, ir_glob_st_r, loc_st_r, len_bnd_r + 1, wh_plane, type=type, bnd=bnd)
            else:
                list_c = org_parts_comm(nri, ir_glob_st_r, loc_st_r, len_bnd_r, wh_plane, type=type, bnd=bnd)

            for obj in list_c:
                pid_r, tpl1, tpl2 = obj[:]
                ir_st_recv, ir_end_recv = tpl1[:]
                ir_st_send, ir_end_send = tpl2[:]
                pid = pid_r * npc + ind_p_filt
                tag = (ind_p + 20) * (pid + 20)

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_recv, ir_end_recv) for j in range(jc_st_pl, jc_end_pl)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_recv.append(ddcp.bound(fld_part, inds, pid, 'n', tag_ref + 10000 + tag))

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_send, ir_end_send) for j in range(jc_st_pl, jc_end_pl)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_send.append(ddcp.bound(fld_part, inds, pid, 's', tag_ref + tag))



    if ind_pr == ind_p_filt and orient == 'sn':
        bnd = param_dict['bnd_xl']
        if not bnd == 'cyclic':

            jc_glob_st_l = max(0, ncj[ind_pc] - wh_plane - ng)
            jc_glob_st_r = min(ncj[ind_pc + 1], ncj[-1])
            jc_glob_end_l = ncj[ind_pc]
            jc_glob_end_r = min(ncj[-1], ncj[ind_pc + 1] + wh_plane + ng)
            loc_st_r = jc_glob_st_r - jc_glob_st_l
            len_fld_h = jc_glob_end_r - jc_glob_st_l
            len_bnd_l = jc_glob_end_l - jc_glob_st_l
            len_bnd_r = jc_glob_end_r - jc_glob_st_r

            flt_out_l = max(len_bnd_l, 0)
            flt_out_h = min(len_bnd_l - 2 * ng + nc_sub, len_fld_h)

            if type == 'sv':
                ind_tpl_same = (jc_glob_end_l - jc_glob_st_l, jc_glob_st_r - jc_glob_st_l + 1)
            else:
                ind_tpl_same = (jc_glob_end_l - jc_glob_st_l, jc_glob_st_r - jc_glob_st_l)

            if type == 'sv':
                fld_tmp = np.zeros([nz_sub, n_plane, len_fld_h + 1])
            else:
                fld_tmp = np.zeros([nz_sub, n_plane, len_fld_h])

            if type == 'sv':
                shp_fld_tmp = ((nz_sub, n_plane, nc_sub - 2 * ng + 1), (ng, nc_sub - 2 * ng + 1))
            else:
                shp_fld_tmp = ((nz_sub, n_plane, nc_sub - 2 * ng), (ng, nc_sub - 2 * ng))

            if type == 'sv':
                x2 = x2coord[jc_glob_st_l:jc_glob_end_r + 1]
            else:
                x = xcoord[jc_glob_st_l:jc_glob_end_r]

            if type == 'c':
                filter_2d = make_filter_plane((0, len_fld_h), (flt_out_l, flt_out_h), (0, nz_sub), lscal_h, x)
            elif type == 'sv':
                filter_2d = make_filter_plane((0, len_fld_h + 1), (flt_out_l, flt_out_h + 1), (0, nz_sub), lscal_h, x2)
            elif type == 'sw':
                filter_2d = make_filter_plane((0, len_fld_h), (flt_out_l, flt_out_h), (0, nz_sub), lscal_h, x)
        else:

            jc_glob_st_l = ncj[ind_pc] - wh_plane - ng
            jc_glob_st_r = ncj[ind_pc + 1]
            loc_st_r = jc_glob_st_r - jc_glob_st_l
            len_fld_h = 2 * wh_plane + nr_sub
            len_bnd_l = wh_plane + ng
            len_bnd_r = wh_plane + ng

            if type == 'sv':
                ind_tpl_same = (ncj[ind_pc] - jc_glob_st_l, ncj[ind_pc + 1] - jc_glob_st_l + 1)
            else:
                ind_tpl_same = (ncj[ind_pc] - jc_glob_st_l, ncj[ind_pc + 1] - jc_glob_st_l)

            if type == 'sv':
                fld_tmp = np.zeros([nz_sub, n_plane, nc_sub + 1 + 2 * wh_plane])
                shp_fld_tmp = ((nz_sub, n_plane, nc_sub + 1), (0, nc_sub + 1))
            else:
                fld_tmp = np.zeros([nz_sub, n_plane, nc_sub + 2 * wh_plane])
                shp_fld_tmp = ((nz_sub, n_plane, nc_sub), (0, nc_sub))

            size_x = xcoord.size

            if type == 'sv':
                x2coord_tmp = x2coord.tolist()
                x2coord_tmp = x2coord.tolist() + (x2coord[1:] + x2coord[-1] - x2coord[0]).tolist() + (x2coord[1:] + 2 * (x2coord[-1] - x2coord[0])).tolist()
                x2 = x2coord_tmp[jc_glob_st_l + size_x:jc_glob_st_l + len_fld_h + 1 + size_x]
                x2  = np.array(x2)
            else:
                xcoord_tmp = xcoord.tolist()
                xcoord_tmp = xcoord.tolist() + (xcoord[1:] + xcoord[-1] - xcoord[0]).tolist() + (xcoord[1:] + 2 * (xcoord[-1] - xcoord[0])).tolist()
                x = xcoord_tmp[jc_glob_st_l + size_x:jc_glob_st_l + len_fld_h + size_x]
                x = np.array(x)

            if type == 'c':
                filter_2d = make_filter_plane((0, len_fld_h), (wh_plane, wh_plane + nr_sub), (0, nz_sub), lscal_h, x)
            elif type == 'sv':
                filter_2d = make_filter_plane((0, len_fld_h + 1), (wh_plane, wh_plane + nr_sub + 1), (0, nz_sub), lscal_h, x2)
            if type == 'sw':
                filter_2d = make_filter_plane((0, len_fld_h), (wh_plane, wh_plane + nr_sub), (0, nz_sub), lscal_h, x)

        ir_st_pl = 0
        ir_end_pl = n_plane

        if len_bnd_l > 0:
            list_c = org_parts_comm(ncj, jc_glob_st_l, 0, len_bnd_l, wh_plane, type=type, bnd=bnd)

            for obj in list_c:
                pid_c, tpl1, tpl2 = obj[:]
                jc_st_recv, jc_end_recv = tpl1[:]
                jc_st_send, jc_end_send = tpl2[:]
                pid = ind_p_filt * npc + pid_c
                tag = (ind_p + 20) * (pid + 20)

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_pl, ir_end_pl) for j in range(jc_st_recv, jc_end_recv)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_recv.append(ddcp.bound(fld_part, inds, pid, 'w', tag_ref + tag))

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_pl, ir_end_pl) for j in range(jc_st_send, jc_end_send)]).T)
                fld_part = np.zeros([len(inds[0])])
                bnds_filter_send.append(ddcp.bound(fld_part, inds, pid, 'e', tag_ref + 10000 + tag))

        if len_bnd_r > 0:
            if type == 'sv':
                list_c = org_parts_comm(ncj, jc_glob_st_r, loc_st_r, len_bnd_r + 1, wh_plane, type=type, bnd=bnd)
            else:
                list_c = org_parts_comm(ncj, jc_glob_st_r, loc_st_r, len_bnd_r, wh_plane, type=type, bnd=bnd)

            for obj in list_c:
                pid_c, tpl1, tpl2 = obj[:]
                jc_st_recv, jc_end_recv = tpl1[:]
                jc_st_send, jc_end_send = tpl2[:]
                pid = ind_p_filt * npc + pid_c
                tag = (ind_p + 20) * (pid + 20)

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_pl, ir_end_pl) for j in range(jc_st_recv, jc_end_recv)]).T)
                fld_part = np.zeros([len(inds[0])])

                bnds_filter_recv.append(ddcp.bound(fld_part, inds, pid, 'e', tag_ref + 10000 + tag))

                inds = list(np.array([(k, i, j) for k in range(k_st, k_end) for i in range(ir_st_pl, ir_end_pl) for j in range(jc_st_send, jc_end_send)]).T)
                fld_part = np.zeros([len(inds[0])])

                bnds_filter_send.append(ddcp.bound(fld_part, inds, pid, 'w', tag_ref + tag))

    return filter_2d, bnds_filter_send, bnds_filter_recv, ind_tpl_same, fld_tmp, shp_fld_tmp 



def make_filter_plane(h_inds_in, h_inds_out, v_inds, lscal_h, h_coord):
    """
    Returns a 1d-box filter operator acting on a 
    2d-velocity slice in the plane perpendicular to a boundary.

    h_inds_in... index tuple for horizontal dimension of input field
    h_inds_out... index tuple for horizontal dimension of output field
    v_inds... index tuple for vertical dimension
    lscal_h... horizontal filter length scale in coordinate units
    h_coord... horizontal coordinate of cell centers
    z_coord... vertical coordinate of cell centers
    """

    row_ids = []
    col_ids = []
    data = []

    hfirst_in, hlast_in = h_inds_in[:]
    hfirst_out, hlast_out = h_inds_out[:]

    vfirst, vlast = v_inds[:]

    nh_in = hlast_in - hfirst_in
    nh_out = hlast_out - hfirst_out    

    nv = vlast - vfirst

    dims_size_in = nh_in * nv
    dims_size_out = nh_out * nv

    op_shape = (dims_size_out, dims_size_in)

    nh_h = int(0.5 * lscal_h / (h_coord[1] - h_coord[0])) 

    for m in range(nh_out * nv):
        k = int(m / nh_out)
        h_out = m - k * nh_out + hfirst_out
        hl = max(h_out - nh_h, 0)
        hh = min(h_out + nh_h, nh_in - 1)

        col_ids_sub = [k * nh_in + ii for ii in range(hl, hh + 1)]
        row_ids_sub = [m for n in col_ids_sub]

        dist_h = [2.0 * abs(h_coord[ii] - h_coord[h_out]) / lscal_h for ii in range(hl, hh + 1)]
        data_sub = [1.0 if val < 1.0 else 0.0 for val in dist_h]
        norm = sum(data_sub)
        data_sub = [val / norm for val in data_sub]
        col_ids.extend(col_ids_sub)
        row_ids.extend(row_ids_sub)
        data.extend(data_sub)

    filt = csr_matrix((data, (row_ids, col_ids)), op_shape)
    return filt


def org_parts_comm(nricj, glob_st, loc_st_recv, len_bnd, wh_plane, type, bnd):
    '''
    Organizes the communication of missing field parts
    for the parallel decentralized filtering operation.

    nricj... global row/column boundary indices of subdomains
    glob_st... global start index of data stripe to communicate
    loc_st_recv... local start index of data stripe to receive/insert
    len_bnd... length of data stripe to communicate
    wh_plane... half length of full data stripe to filter
    type... field type ('c':centred, 'sv':staggered v, 'sw':staggered w)
    bnd... type of lateral boundary condition
    '''


    if bnd == 'cyclic':
        glob_st += nricj[-1]
        nricj_tmp = nricj[:]
        nricj_tmp.extend(val + nricj[-1] for val in nricj[1:])
        nricj_tmp.extend(val + 2 * nricj[-1] for val in nricj[1:])
        glob_end  = glob_st + len_bnd
    else:
        nricj_tmp = nricj[:]
        glob_end  = glob_st + len_bnd

    parts_comm = []
    if loc_st_recv == 0:
        ind_prc = np.argwhere(np.array(nricj_tmp[:]) > (glob_st + len_bnd))[0][0] - 1
        side = 'w'
    else:
        ind_prc = np.argwhere(np.array(nricj_tmp[:]) > (glob_st - ng))[0][0] - 1
        side = 'e'

    if type == 'c' or type =='sw':
        a = 0
        b = 0
    else:
        if side == 'w':
            a = 1
            b = 0
        else:
            a = 0
            b = 1
    n_inds_tot = 0


    while True:
        lst = np.argwhere(np.array(nricj_tmp[:]) > (glob_st))
        if len(lst):
            ind_prc_recv = lst[0][0] - 1
        elif glob_st == 0:
            ind_prc_recv = 0
        if len_bnd - n_inds_tot <= nricj_tmp[ind_prc_recv + 1] + b - glob_st:
            n_inds = min(nricj_tmp[ind_prc_recv + 1] + b, glob_end) - glob_st
        else:
            n_inds = min(nricj_tmp[ind_prc_recv + 1], glob_end) - glob_st
        if side == 'w':
            loc_st_send = 0
            if ind_prc == len(nricj_tmp) - 2:
                loc_end_send = min(nricj_tmp[ind_prc_recv + 1] + a + wh_plane + ng - nricj_tmp[ind_prc], nricj_tmp[ind_prc + 1] + a - nricj_tmp[ind_prc])
            else:
                loc_end_send = min(nricj_tmp[ind_prc_recv + 1] + a + wh_plane + ng - nricj_tmp[ind_prc], nricj_tmp[ind_prc + 1] - nricj_tmp[ind_prc])
        elif side == 'e':
            loc_st_send = max(nricj_tmp[ind_prc_recv] - (wh_plane + ng) - nricj_tmp[ind_prc], 0)
            loc_end_send = nricj_tmp[ind_prc + 1]  - nricj_tmp[ind_prc]
        parts_comm.append([ind_prc_recv, (loc_st_recv, loc_st_recv + n_inds), (loc_st_send, loc_end_send)])
        glob_st += n_inds
        loc_st_recv += n_inds
        n_inds_tot += n_inds
        if n_inds_tot  == len_bnd:
            break

    if bnd == 'cyclic':
        new_lst = []
        for obj in parts_comm:
            ind_prc_recv, tup1, tup2 = obj[:]
            ind_prc_recv -= len(nricj) - 1
            if ind_prc_recv < 0:
                ind_prc_recv += len(nricj) - 1
            elif ind_prc_recv >= len(nricj) - 1:
                ind_prc_recv -= len(nricj) - 1
            new_lst.append([ind_prc_recv, tup1, tup2])
        parts_comm = new_lst

    return parts_comm

