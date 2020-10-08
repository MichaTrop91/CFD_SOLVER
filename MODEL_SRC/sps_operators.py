# Code design: 
# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020

# load external python packages
import numpy as np
from scipy.sparse import csc_matrix, csr_matrix, diags, eye
import scipy
from scipy.sparse.linalg import norm
import scipy.sparse as sparse
from scipy.interpolate import interp2d
from scipy.sparse.linalg import spsolve, spilu
from scipy.linalg import lstsq
from scipy.optimize import lsq_linear
from copy import copy, deepcopy
from time import time

# load model specific *py files
import domain_decomp as ddcp
from sparse_approximate_inverse import make_SPAI, sparsificate, construct_sparsity_stencil



def make_DIV(nz, ny, nx, comp='full'):
    """ 
    Constructs the divergence stencil operator in sparse
    matrix form.
    
    nz, ny, nx... 3d-shape of divergence field 
    comp... specify whether the full gradient or only one-dimensional components 
            of the full stencil are constructed ('full', 'u', 'v', 'w')

    D_left_part, D_right_part... returned divergence stencil
    """

    col_ids_left = []
    col_ids_right = []
    row_ids = []
    data_left = []
    data_right = []

    shape = (nx * ny * nz, (nx + 1) * ny * nz + (ny + 1) * nx * nz + nx * ny * (nz + 1))

    for m in xrange(0, nx * ny * nz, 1):

        k = int(m / (ny * nx))
        i = int((m - k * ny * nx) / nx)
        j = m - k * ny * nx - i * nx        

        if comp == 'full' or comp == 'u':
            col_ids_left.append(find_nx(k, i, j, nz, ny, nx))
            col_ids_right.append(find_nx(k, i, j + 1, nz, ny, nx))
            data_left.append(-1.0)
            data_right.append(1.0)
            for n in range(1):
                row_ids.append(m)

        if comp == 'full' or comp == 'v':
            col_ids_left.append(find_ny(k, i, j, nz, ny, nx))
            col_ids_right.append(find_ny(k, i + 1, j, nz, ny, nx))
            data_left.append(-1.0)
            data_right.append(1.0)
            for n in range(1):
                row_ids.append(m)

        if comp == 'full' or comp == 'w':
            col_ids_left.append(find_nz(k, i, j, nz, ny, nx))
            col_ids_right.append(find_nz(k + 1, i, j, nz, ny, nx))
            data_left.append(-1.0)
            data_right.append(1.0)
            for n in range(1):
                row_ids.append(m)

    D_left_part = csr_matrix((data_left, (row_ids, col_ids_left)), shape)
    D_right_part = csr_matrix((data_right, (row_ids, col_ids_right)), shape)

    return D_left_part, D_right_part



def make_div_grad(grid_fields, z2_list, bnd_cond_neumann_lst, nhalo=1, level=0):
    """
    Constructs the divergence and gradient operators
    in terrain-following coordinates.
    grid_fields... list of grid-geometry fields
    z2_list... grid-plane coordinates of xy-planes
    neumann_bnd_cond_lst... list of bools for boundary value specification
    nhalo... number of lateral halo layers
    level... multigrid level

    OP_list... returned list of operators containing the divergence and gradient
               as wells as their metrically corrected ones.    
    """

    vols_eff, area_eff_x, area_eff_y, area_eff_z, dginv_comp_lev, dsurfdx, dsurfdy  = grid_fields[:]
    dginvx_l, dginvx_r, dginvy_l, dginvy_r, dginvz_l, dginvz_r = dginv_comp_lev[:]

    nz, nr, nc = vols_eff.shape

    vols_eff_1d = vols_eff.flatten()
    areas_eff_1d = put_1d([area_eff_x, area_eff_y, area_eff_z])
    dginv_l_1d = put_1d([dginvx_l, dginvy_l, dginvz_l])
    dginv_r_1d = put_1d([dginvx_r, dginvy_r, dginvz_r])
    dsurf_1d = put_1d([dsurfdx, dsurfdy, np.zeros_like(dginvz_r)])    

    # raw operators    
    FD = diags(areas_eff_1d, format='csr')
    VD = diags(vols_eff_1d, format='csr')
    
    vols_eff_1d_inv = 1.0 / (vols_eff_1d + 1e-100) * np.array(np.array(vols_eff_1d, dtype=bool), dtype=float)
    VDinv = diags(vols_eff_1d_inv, format='csr')
    DGinvl_eff, DGinvr_eff = diags(dginv_l_1d, format='csr'), diags(dginv_r_1d, format='csr')

    DIV_xl_stencil, DIV_xr_stencil = make_DIV(nz, nr, nc, comp='u')
    DIV_yl_stencil, DIV_yr_stencil = make_DIV(nz, nr, nc, comp='v')
    DIV_zl_stencil, DIV_zr_stencil = make_DIV(nz, nr, nc, comp='w')
   
    DIV_x_stencil = DIV_xl_stencil + DIV_xr_stencil
    DIV_y_stencil = DIV_yl_stencil + DIV_yr_stencil
    DIV_z_stencil = DIV_zl_stencil + DIV_zr_stencil

    DIV_x = VDinv * DIV_x_stencil * FD
    DIV_y = VDinv * DIV_y_stencil * FD
    DIV_z = VDinv * DIV_z_stencil * FD


    GRAD_x = -DGinvr_eff * DIV_xl_stencil.transpose() - DGinvl_eff * DIV_xr_stencil.transpose() 
    GRAD_y = -DGinvr_eff * DIV_yl_stencil.transpose() - DGinvl_eff * DIV_yr_stencil.transpose()
    GRAD_z = -DGinvr_eff * DIV_zl_stencil.transpose() - DGinvl_eff * DIV_zr_stencil.transpose() 

    DIV = DIV_x + DIV_y + DIV_z
    GRAD = GRAD_x + GRAD_y + GRAD_z

    # surface derivative operator
    DSURF = diags(dsurf_1d, format='csr')

    # interpolation operators    
    INT_c_u = make_interpol_V_F(nz, nr, nc, dims=(1, 0, 0))
    INT_c_v = make_interpol_V_F(nz, nr, nc, dims=(0, 1, 0))
    INT_w_c = make_interpol_V_F(nz, nr, nc, dims=(0, 0, 1), transpose=True)
    INT_c_w = make_interpol_V_F(nz, nr, nc, dims=(0, 0, 1))
    INT_u_c = make_interpol_V_F(nz, nr, nc, dims=(1, 0, 0), transpose=True)
    INT_v_c = make_interpol_V_F(nz, nr, nc, dims=(0, 1, 0), transpose=True)

    # selection operators
    ones_x = np.ones([nz, nr, nc + 1])
    zeros_x = np.zeros([nz, nr, nc + 1])
    ones_y = np.ones([nz, nr + 1, nc])
    zeros_y = np.zeros([nz, nr + 1, nc])
    ones_z = np.ones([nz + 1, nr, nc])
    zeros_z = np.zeros([nz + 1, nr, nc])

    X_comp = diags(put_1d([ones_x, zeros_y, zeros_z]), format='csr')
    Y_comp = diags(put_1d([zeros_x, ones_y, zeros_z]), format='csr')
    Z_comp = diags(put_1d([zeros_x, zeros_y, ones_z]), format='csr')

    # metric correction operators for gradient and divergence matrix

    TRANS_gradx = eye(GRAD.shape[0]) - DSURF * INT_c_u * INT_w_c * Z_comp
    TRANS_grady = eye(GRAD.shape[0]) - DSURF * INT_c_v * INT_w_c * Z_comp
    TRANS_gradz = eye(GRAD.shape[0]) - DSURF * INT_c_w * (INT_u_c * X_comp + INT_v_c * Y_comp)
    TRANSx_contra = eye(GRAD.shape[0]) - INT_c_w * INT_u_c * DSURF * X_comp
    TRANSy_contra = eye(GRAD.shape[0]) - INT_c_w * INT_v_c * DSURF * Y_comp

    # divergence and gradient operators in terrain-following coordinates

    DIV_contra =  DIV * (TRANSx_contra + TRANSy_contra - eye(GRAD.shape[0]))
    GRAD_metric =  (TRANS_gradx + TRANS_grady - eye(GRAD.shape[0])) * GRAD
    GRAD_metric = implement_neumann_bc(GRAD_metric, grid_fields, z2_list, bnd_cond_neumann_lst, nhalo=nhalo, level=level)

    # return the list of operators

    OP_list = [DIV, GRAD, DIV_contra, GRAD_metric]

    return OP_list


def implement_neumann_bc(GRAD_halo, grid_fields, z2_list, neumann_bnd_cond_lst, nhalo=1, level=0):
    """
    Setting lateral Neumann boundary condition in the gradient operator.
    
    GRAD_halo... Gradient operator with nhalo cells
    grid_fields... list of matching-level grid fields
    z2_list... xy-plane coordinates of matching-level grid
    neumann_bnd_cond_lst... list of bools for boundary value specification
    nhalo... number of lateral halo layers
    level... multigrid level

    GRAD_halo_Neumann... returned gradient operator with Neumann boundary conditions 
    """

    vols_eff, area_eff_x, area_eff_y, area_eff_z, dginv_comp, dsurfdx, dsurfdy  = grid_fields[:]
    nz, nr, nc = vols_eff.shape

    data = GRAD_halo.data
    indices = GRAD_halo.indices.tolist()
    indptr = GRAD_halo.indptr.tolist()
   
    dsurfdx_lst = (dsurfdx.flatten()).tolist()
    dsurfdy_lst = (dsurfdy.flatten()).tolist() 

    # west side
    if neumann_bnd_cond_lst[0] == True: 
        j = nhalo
        for k in range(nz):
            for i in range(nr):
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # east side
    if neumann_bnd_cond_lst[1] == True:
        j = nc - nhalo
        for k in range(nz):
            for i in range(nr):
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # south side
    if neumann_bnd_cond_lst[2] == True:
        i = nhalo
        for k in range(nz):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # north side
    if neumann_bnd_cond_lst[3] == True:
        i = nr - nhalo
        for k in range(nz):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    if neumann_bnd_cond_lst[4] == True:
        k = nhalo
        for i in range(nr):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                bnd_fac = 1.0 - abs(z2_list[0][2] - z2_list[0][1]) / abs(z2_list[level][2] - z2_list[level][1])

                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                if dsurfdx_lst[nrow] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k, nr, nc, bnd_fac)
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j + 1
                if dsurfdx_lst[nrow] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k, nr, nc, bnd_fac)
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                if dsurfdy_lst[nrow - nz * nr * (nc + 1)] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k, nr, nc, bnd_fac)
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + (i + 1) * nc + j
                if dsurfdy_lst[nrow - nz * nr * (nc + 1)] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k, nr, nc, bnd_fac)

    if neumann_bnd_cond_lst[5] == True:
        k = nz - nhalo
        for i in range(nr):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                bnd_fac = 1.0 - abs(z2_list[0][-3] - z2_list[0][-4]) / abs(z2_list[level][-3] - z2_list[level][-4])

                nrow = (k - 1) * nr * (nc + 1) + i * (nc + 1) + j
                if dsurfdx_lst[nrow] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k - 1, nr, nc, bnd_fac)
                nrow = (k - 1) * nr * (nc + 1) + i * (nc + 1) + j + 1
                if dsurfdx_lst[nrow] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k - 1, nr, nc, bnd_fac)
                nrow = nz * nr * (nc + 1) + (k - 1) * (nr + 1) * nc + i * nc + j
                if dsurfdy_lst[nrow - nz * nr * (nc + 1)] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k - 1, nr, nc, bnd_fac)
                nrow = nz * nr * (nc + 1) + (k - 1) * (nr + 1) * nc + (i + 1) * nc + j
                if dsurfdy_lst[nrow - nz * nr * (nc + 1)] != 0.0:
                    data[indptr[nrow]:indptr[nrow + 1]] = set_neumann_bc_terrain_in_DCA(data[indptr[nrow]:indptr[nrow + 1]], indices[indptr[nrow]:indptr[nrow + 1]], k - 1, nr, nc, bnd_fac)


    GRAD_halo_Neumann = csr_matrix((data, indices, indptr), GRAD_halo.shape)
    return GRAD_halo_Neumann


def implement_neumann_bc_pcorr(GRAD_pcorr, shape_3d, neumann_bnd_cond_lst, nhalo=1):
    """
    Setting lateral Neumann boundary condition in the gradient operator used to apply the pressure correction.
    
    GRAD_pcorr... Gradient operator with nhalo cells
    shape_3d... 3d-shape of subgrid + halo layer
    neumann_bnd_cond_lst... list of bools for boundary value specification
    nhalo... number of lateral halo layers

    GRAD_pcorr_Neumann... returned gradient operator for pressure correction with Neumann boundary conditions 
    """

    data = GRAD_pcorr.data
    indices = GRAD_pcorr.indices.tolist()
    indptr = GRAD_pcorr.indptr.tolist()

    nz, nr, nc = shape_3d[:]

    # west side
    if neumann_bnd_cond_lst[0] == True:
        j = nhalo
        for k in range(nz):
            for i in range(nr):
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + (i + 1) * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + (k + 1) * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # east side
    if neumann_bnd_cond_lst[1] == True:
        j = nc - nhalo
        for k in range(nz):
            for i in range(nr):
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j - 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + (i + 1) * nc + j - 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j - 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + (k + 1) * nr * nc + i * nc + j - 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # south side
    if neumann_bnd_cond_lst[2] == True:
        i = nhalo
        for k in range(nz):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * (nc + 1) * nr + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * (nc + 1) * nr + i * (nc + 1) + j + 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * nc * (nr + 1) + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * nc * (nr + 1) + (k + 1) * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    # north side
    if neumann_bnd_cond_lst[3] == True:
        i = nr - nhalo
        for k in range(nz):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * (nc + 1) * nr + (i - 1) * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * (nc + 1) * nr + (i - 1) * (nc + 1) + j + 1 
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * nc * (nr + 1) + k * nr * nc + (i - 1) * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + nz * nc * (nr + 1) + (k + 1) * nr * nc + (i - 1) * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0


    if neumann_bnd_cond_lst[4] == True:
        k = nhalo
        for i in range(nr):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = k * nr * (nc + 1) + i * (nc + 1) + j + 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + k * (nr + 1) * nc + (i + 1) * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    if neumann_bnd_cond_lst[5] == True:
        k = nz - nhalo
        for i in range(nr):
            for j in range(nc):
                nrow = nz * nr * (nc + 1) + nz * (nr + 1) * nc + k * nr * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = (k - 1) * nr * (nc + 1) + i * (nc + 1) + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = (k - 1) * nr * (nc + 1) + i * (nc + 1) + j + 1
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + (k - 1) * (nr + 1) * nc + i * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0
                nrow = nz * nr * (nc + 1) + (k - 1) * (nr + 1) * nc + (i + 1) * nc + j
                data[indptr[nrow]:indptr[nrow + 1]] = 0.0

    GRAD_pcorr_Neumann = csr_matrix((data, indices, indptr), GRAD_pcorr.shape)
    return GRAD_pcorr_Neumann


def set_neumann_bc_terrain_in_DCA(row_data, row_indices, k_bnd, nr, nc, bnd_fac):
    """
    Approximates Neumann boundary condition of the 
    metric derivatives near the domain top and bottom
    boundaries when using terrain-following coordinates.
    On the coarse grid, depending on the coarsening factor, the
    boundary condition is only imposed partly.

    row_data... matrix coefficients of the difference expression associated with a computation node 
    row_indices... matrix coefficient indices of the difference expression associated with a computation node 
    k_bnd... k index where the horizontal derivatives are to scale
    nr, nc... number of rows and number of colums of matrix
    bnd_fac... bound scaling factor depending on the coarsening factor
    """

    for n, ind in enumerate(row_indices):
        k_prime = ind / (nr * nc)
        if k_prime == k_bnd:
            row_data[n] *= bnd_fac
        else:
            row_data[n] = 0.0

    return row_data



def crop_operators(shape_3d, ncrop=1):
    """
    This function returns crop operators
    to remove ncrop halo-layers from
    (i) a cell-centred field
    (ii) a vertex-centred  field
    (i) applied from the left and its tranpose to the right is used to
    crop the Laplace operator.

    shape_3d... dimension sizes of the uncropped field.
    ncrop... number of boundary cells to crop

    CROP_OP_CELL... returned operator for cell-centred cropping
    CROP_OP_FACE... returned operator for face-centred cropping
    """

    nz, ny, nx = shape_3d[:]
    nz_c = nz - 2 * ncrop
    ny_c = ny - 2 * ncrop
    nx_c = nx - 2 * ncrop

    shape_crop_op = [nx_c * ny_c * nz_c, nx * ny * nz]

    data_crop_op = []
    rowids_crop_op = []
    colids_crop_op = []

    for n in range(nx_c * ny_c * nz_c):
        k_c = int(n / (nx_c * ny_c))
        i_c = int((n - k_c * nx_c * ny_c) / nx_c)
        j_c = n - k_c * nx_c * ny_c -  i_c * nx_c
        k = k_c + ncrop
        i = i_c + ncrop
        j = j_c + ncrop
        m = k * nx * ny + i * nx + j
        data_crop_op.append(1)
        rowids_crop_op.append(n)
        colids_crop_op.append(m)

    CROP_OP_CELL = csr_matrix((data_crop_op, (rowids_crop_op, colids_crop_op)), shape_crop_op)

    shape_crop_op = [
                        (nx_c + 1) * ny_c * nz_c + nx_c * (ny_c + 1) * nz_c + 
                        nx_c * ny_c * (nz_c + 1), (nx + 1) * ny * nz + 
                        nx * (ny + 1) * nz + nx * ny * (nz + 1)
                    ]
    data_crop_op = []
    rowids_crop_op = []
    colids_crop_op = []

    for n in range((nx_c + 1) * ny_c * nz_c):
        k_c = int(n / ((nx_c + 1) * ny_c))
        i_c = int((n - k_c * (nx_c + 1) * ny_c) / (nx_c + 1))
        j_c = n - k_c * (nx_c + 1) * ny_c - i_c * (nx_c + 1) 
        k = k_c + ncrop
        i = i_c + ncrop
        j = j_c + ncrop

        m = k * (nx + 1) * ny + i * (nx + 1) + j
        data_crop_op.append(1)
        rowids_crop_op.append(n)
        colids_crop_op.append(m)

    for n in range((ny_c + 1) * nx_c * nz_c):
        k_c = int(n / ((ny_c + 1) * nx_c))
        i_c = int((n - k_c * (ny_c + 1) * nx_c) / nx_c)
        j_c = n - k_c * (ny_c + 1) * nx_c - i_c * nx_c
        k = k_c + ncrop
        i = i_c + ncrop
        j = j_c + ncrop

        m = k * (ny + 1) * nx + i * nx + j
        data_crop_op.append(1)
        rowids_crop_op.append(n + nz_c * ny_c * (nx_c + 1))
        colids_crop_op.append(m + nz * ny * (nx + 1))

    for n in range(ny_c * nx_c * (nz_c + 1)):
        k_c = int(n / (ny_c * nx_c))
        i_c = int((n - k_c * ny_c * nx_c) / nx_c)
        j_c = n - k_c * ny_c * nx_c - i_c * nx_c
        k = k_c + ncrop
        i = i_c + ncrop
        j = j_c + ncrop

        m = k * nx * ny + i * nx + j 
        data_crop_op.append(1)
        rowids_crop_op.append(n + nz_c * ny_c * (nx_c + 1) + nx_c * (ny_c + 1) * nz_c)
        colids_crop_op.append(m + nz * ny * (nx + 1) + nx * (ny + 1) * nz)


    CROP_OP_FACE = csr_matrix((data_crop_op, (rowids_crop_op, colids_crop_op)), shape_crop_op)

    return CROP_OP_CELL, CROP_OP_FACE
    
 

def make_SPAI_smoother(MAT, shape_3d, param_dict, lev=0):
    """
    For an input matrix MAT an approximate inverse is derived,
    which is the smoothing matrix for the according multigrid level.


    MAT... matrix to which approximate the sparse inverse
    shape_3d... 3d field shape
    param_dict... parameter dictionary
    lev... multigrid level

    SMO... returned smoother
    """

    nz, ny, nx = shape_3d[:]

#    PATT = construct_sparsity_stencil(MAT, nz, ny, nx, l=param_dict['SPAI_q'])
#    SMO = make_SPAI(MAT, PATT)
    pot = 4
    PATT = sum([MAT ** n for n in range(pot)])
    SMO = make_SPAI(MAT, PATT) 
    SMO = sparsificate(SMO, nfill=int(param_dict['SPAI_nfillmax']), drop_tol=param_dict['SPAI_drop_tol'])
#    SMO = make_SPAI(MAT, SMO)
    
    return SMO



def make_checkerboard_line_pattern(nri, ncj, nz, pid_r, pid_c, bnd_x, bnd_y, bnd_z, direction):
    """
    Returns operators that filters
    the solution-vector in checker-board
    fashion using lines of constant color.

    nri, ncj... subdomain boundary indices (rows, columns)
    nz... number of vertical layers
    pid_r, pid_c... process row and column indices of the processor grid
    bnd_x, bnd_y, bnd_z... boundary conditions (Neumann or cyclic)
    direction... direction of lines     

    RD, BL... returned red and black operators
    """

    nx = ncj[pid_c + 1] - ncj[pid_c]
    ny = nri[pid_r + 1] - nri[pid_r]

    shape = [nx * ny * nz, nx * ny * nz]

    data_rd = []
    data_bl = []
    row_ids_rd = []
    row_ids_bl = []
    col_ids_rd = []
    col_ids_bl = []

    x_st = ncj[pid_c]
    y_st = nri[pid_r]


    if nz % 2 == 1 and bnd_z == 'cyclic':
        d_z = 1
    else:
        d_z = 0

    if direction == 'x':
        if y_st % 2 == 0:
            c = 0
        else:
            c = 1

        if all((pid_r == len(nri) - 2, nri[-1] % 2 == 1, bnd_y == 'cyclic')):
            d_y = 1
        else:
            d_y = 0

        d_x = 0

        def condition(k, i, j):
            return any((all((k % 2 == 0, i % 2 == 0)), all((k % 2 == 1, i % 2 == 1))))

    elif direction == 'y':
        if x_st % 2 == 0:
            c = 0
        else:
            c = 1

        if all((pid_c == len(ncj) - 2, ncj[-1] % 2 == 1, bnd_x == 'cyclic')):
            d_x = 1
        else:
            d_x = 0

        d_y = 0

        def condition(k, i, j):
            return any((all((k % 2 == 0, j % 2 == 0)), all((k % 2 == 1, j % 2 == 1))))

    elif direction == 'z':
        if y_st % 2 == 0:
            c = 0
        else:
            c = 1
        if x_st % 2 == 0:
            c = 0
        else:
            c = 1
        if all((pid_r == len(nri) - 2, nri[-1] % 2 == 1, bnd_y == 'cyclic')):
            d_y = 1
        else:
            d_y = 0
        if all((pid_c == len(ncj) - 2, ncj[-1] % 2 == 1, bnd_x == 'cyclic')):
            d_x = 1
        else:
            d_x = 0

        def condition(k, i, j):
            return any((all((i % 2 == 0, j % 2 == 0)), all((i % 2 == 1, j % 2 == 1))))

    p = 0
    q = 0

    for m in range(nz * ny * nx):
       k = int(m / (ny * nx))
       i = int((m - k * ny * nx) / nx)
       j = m - k * ny * nx - i * nx

       if k == nz - 1:
           k -= d_z
       if i == ny - 1:
           i -= d_y
       if j == nx - 1:
           j -= d_x

       if (condition(k, i, j) and c == 0) or (not condition(k, i, j) and c == 1):

           data_bl.append(1.0)
           row_ids_bl.append(m)
           col_ids_bl.append(m)
           p += 1
       else:
           data_rd.append(1.0)
           row_ids_rd.append(m)
           col_ids_rd.append(m)
           q += 1


    RD = csr_matrix((data_rd, (row_ids_rd, col_ids_rd)), shape)
    BL = csr_matrix((data_bl, (row_ids_bl, col_ids_bl)), shape)

    return RD, BL
    

def make_checkerboard_point_pattern(nri, ncj, nz, pid_r, pid_c, bnd_x, bnd_y, bnd_z):
    """
    Returns operators that filters
    the solution-vector in checker-board
    fashion.

    nri, ncj... subdomain boundary indices (rows, columns)
    nz... number of vertical layers
    pid_r, pid_c... process row and column indices of the processor grid
    bnd_x, bnd_y, bnd_z... boundary conditions (Neumann or cyclic)

    RD, BL... returned red and black operators
    """

    nx = ncj[pid_c + 1] - ncj[pid_c]
    ny = nri[pid_r + 1] - nri[pid_r] 
     
    shape = [nx * ny * nz, nx * ny * nz]   
    
    data_rd = []
    data_bl = []
    row_ids_rd = []
    row_ids_bl = []
    col_ids_rd = []
    col_ids_bl = []

    x_st = ncj[pid_c]
    y_st = nri[pid_r]
    
    if ((x_st % 2 == 0 and y_st % 2 == 0) or (x_st % 2 == 1 and y_st % 2 == 1)):
        c = 0
    else:
        c = 1

    if all((pid_c == len(ncj) - 2, ncj[-1] % 2 == 1, bnd_x == 'cyclic')):        
        d_x = 1
    else:
        d_x = 0
    if all((pid_r == len(nri) - 2, nri[-1] % 2 == 1, bnd_y == 'cyclic')):
        d_y = 1    
    else:
        d_y = 0

    if nz % 2 == 1 and bnd_z == 'cyclic':
        d_z = 1
    else:
        d_z = 0

    p = 0
    q = 0
    for m in range(nz * ny * nx):
       k = int(m / (ny * nx))
       i = int((m - k * ny * nx) / nx)
       j = m - k * ny * nx - i * nx

       if k == nz - 1:
           k -= d_z
       if i == ny - 1:
           i -= d_y
       if j == nx - 1:
           j -= d_x

       cond = any((
                      all((k % 2 == 0, i % 2 == 0, j % 2 == 0)), 
                      all((k % 2 == 0, i % 2 == 1, j % 2 == 1)),
                      all((k % 2 == 1, i % 2 == 1, j % 2 == 0)),
                      all((k % 2 == 1, i % 2 == 0, j % 2 == 1))
                 ))

       if (cond and c == 0) or (not cond and c == 1): 

           data_bl.append(1.0)
           row_ids_bl.append(m)
           col_ids_bl.append(m)
           p += 1
       else:
           data_rd.append(1.0)
           row_ids_rd.append(m)
           col_ids_rd.append(m)
           q += 1

    
    RD = csr_matrix((data_rd, (row_ids_rd, col_ids_rd)), shape)
    BL = csr_matrix((data_bl, (row_ids_bl, col_ids_bl)), shape)

    return RD, BL



def make_restr(edges_fine, edges_coarse, dom_bnds, order='constant', type='standard'):
    """ 
    Constructs the interpolation operator used for restriction of the residual,
    or alternatively in the Galerkin process to derive the coarse-grid
    approximation of the fine-grid operator.
    
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid
    dom_bnds... lateral boundary indices of the interpolation (defined on the global grid)
    order... order of  interpolation (constant or linear)
    type... Galerkin or standard

    REST... returned interpolation matrix
    """

    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]

    x_fine = 0.5 * (x2_fine[1:] + x2_fine[:-1])
    y_fine = 0.5 * (y2_fine[1:] + y2_fine[:-1])
    z_fine = 0.5 * (z2_fine[1:] + z2_fine[:-1])

    x_coarse = 0.5 * (x2_coarse[1:] + x2_coarse[:-1])
    y_coarse = 0.5 * (y2_coarse[1:] + y2_coarse[:-1])
    z_coarse = 0.5 * (z2_coarse[1:] + z2_coarse[:-1])

    nz_f = z2_fine.size - 1
    nz_c = z2_coarse.size - 1

    dom_bnds_out, dom_bnds_in = dom_bnds[:]

    ir_tpl_in, jc_tpl_in = dom_bnds_in[:]
    ir_tpl_out, jc_tpl_out = dom_bnds_out[:]    

    ir_cst, ir_cend = ir_tpl_out[:]
    jc_cst, jc_cend = jc_tpl_out[:]

    ir_fst, ir_fend = ir_tpl_in[:]
    jc_fst, jc_fend = jc_tpl_in[:]

    ny_c = ir_cend - ir_cst
    nx_c = jc_cend - jc_cst

    ny_f = ir_fend - ir_fst
    nx_f = jc_fend - jc_fst


    nrows = nx_c * ny_f * nz_f
    ncols = ny_f * nx_f * nz_f
    size = [nrows, ncols]

    data = []
    row_ids = []
    col_ids = []



    for n in xrange(0, nx_f * ny_f * nz_f):
        k, i, j = retrieve_kij(nz_f, ny_f, nx_f, n)

        x_mid = x_fine[jc_fst + j]
        x_low = x_fine[max(jc_fst + j - 1, 0)]
        x_high = x_fine[min(jc_fst + j + 1, x_fine.size - 1)]

        ind_c1 = np.argwhere(np.logical_and(x2_coarse[1:] > x_mid, x2_coarse[:-1] < x_mid))[0][0]
        jc1 = ind_c1 - jc_cst
        if x_mid < x_coarse[ind_c1]:
            ind_c2 = max(ind_c1 - 1, 0)
            jc2 = ind_c2 - jc_cst
            ind_c3 = ind_c1
            jc3 = ind_c3 - jc_cst
        elif x_mid > x_coarse[ind_c1]:
            ind_c2 = ind_c1
            jc2 = ind_c2 - jc_cst
            ind_c3 = min(ind_c1 + 1, x_coarse.size - 1)
            jc3 = ind_c3 - jc_cst
        else:
            ind_c2 = ind_c1
            jc2 = ind_c2 - jc_cst
            ind_c3 = ind_c1
            jc3 = ind_c3 - jc_cst

        row_ids_sub = []
        col_ids_sub = []
        data_sub = []

        if order == 'constant':
            f1 = 1.0
            f2 = 0.0
            f3 = 0.0

        elif order == 'linear':
            if ind_c1 == ind_c3 and ind_c1 == ind_c2:
                f1 = 1.0
                f2 = 0.0
                f3 = 0.0
            elif ind_c1 == ind_c2:
                f1 = 0.5 * (1.0 - abs(x_mid - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c3] + 1e-100))
                f2 = 0.5 * (1.0 - abs(x_mid - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c3] + 1e-100))
                f3 = (1.0 - abs(x_mid - x_coarse[ind_c3]) / abs(x_coarse[ind_c1] - x_coarse[ind_c3] + 1e-100))
            elif ind_c1 == ind_c3:
                f1 = 0.5 * (1.0 - abs(x_mid - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c2] + 1e-100))
                f3 = 0.5 * (1.0 - abs(x_mid - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c2] + 1e-100))
                f2 = (1.0 - abs(x_mid - x_coarse[ind_c2]) / abs(x_coarse[ind_c1] - x_coarse[ind_c2] + 1e-100))

        if x_fine[max(jc_fst + j - 1, 0)] == x_coarse[ind_c2]:
            f1 += f2 / 2.0
            f3 += f2 / 2.0
            f2 = 0.0
        if x_fine[min(jc_fst + j + 1, x_fine.size - 1)] == x_coarse[ind_c3]:
            f1 += f3 / 2.0
            f2 += f3 / 2.0
            f3 = 0.0                   

        if jc1 < 0 or jc1 > nx_c - 1:
            f1 = 0.0
        if jc2 < 0 or jc2 > nx_c - 1:
            f2 = 0.0
        if jc3 < 0 or jc3 > nx_c - 1:
            f3 = 0.0

        if jc1 == jc2 and jc1 == jc3:
            data_sub.append(f1 + f2 + f3)
            row_ids_sub.append(jc1 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
        elif jc1 == jc2: 
            data_sub.append(f1 + f2)
            row_ids_sub.append(jc1 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(jc3 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
        elif jc1 == jc3:
            data_sub.append(f1 + f3)
            row_ids_sub.append(jc1 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(jc2 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
        else:
            data_sub.append(f1)
            row_ids_sub.append(jc1 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(jc2 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(jc3 + i * nx_c + k * ny_f * nx_c)
            col_ids_sub.append(n)
    
        for ind in list(row_ids_sub):
            if ind < 0 or ind > nrows - 1:
                pos = row_ids_sub.index(ind) 
                row_ids_sub.remove(ind)    
                data_sub.remove(data_sub[pos])
                col_ids_sub.remove(col_ids_sub[pos])

        data.extend(data_sub)
        row_ids.extend(row_ids_sub)
        col_ids.extend(col_ids_sub)
        
    RESTX = csr_matrix((data, (row_ids, col_ids)), size)   

    nrows = nx_c * ny_c * nz_f
    ncols = ny_f * nx_c * nz_f
    size = [nrows, ncols]


    data = []
    row_ids = []
    col_ids = []

    for n in xrange(0, nx_c * ny_f * nz_f):
        k, i, j = retrieve_kij(nz_f, ny_f, nx_c, n)

        y_mid = y_fine[ir_fst + i]
        y_low = y_fine[max(ir_fst + i - 1, 0)]
        y_high = y_fine[min(ir_fst + i + 1, y_fine.size - 1)]


        ind_c1 = np.argwhere(np.logical_and(y2_coarse[1:] > y_mid, y2_coarse[:-1] < y_mid))[0][0]
        ir1 = ind_c1 - ir_cst
        if y_mid < y_coarse[ind_c1]:
            ind_c2 = max(ind_c1 - 1, 0)
            ir2 = ind_c2 - ir_cst
            ind_c3 = ind_c1
            ir3 = ind_c3 - ir_cst
        elif y_mid > y_coarse[ind_c1]:
            ind_c2 = ind_c1
            ir2 = ind_c2 - ir_cst
            ind_c3 = min(ind_c1 + 1, y_coarse.size - 1)
            ir3 = ind_c3 - ir_cst
        else:
            ind_c2 = ind_c1
            ir2 = ind_c2 - ir_cst
            ind_c3 = ind_c1
            ir3 = ind_c3 - ir_cst        

        row_ids_sub = []
        col_ids_sub = []
        data_sub = []

        if order == 'constant':
            f1 = 1.0
            f2 = 0.0
            f3 = 0.0

        elif order == 'linear':

            if ind_c1 == ind_c3 and ind_c1 == ind_c2 or ind_c1 != ind_c3 and ind_c1 != ind_c2:
                f1 = 1.0
                f2 = 0.0
                f3 = 0.0
            elif ind_c1 == ind_c2:
                f1 = 0.5 * (1.0 - abs(y_mid - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c3] + 1e-100))
                f2 = 0.5 * (1.0 - abs(y_mid - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c3] + 1e-100))
                f3 = (1.0 - abs(y_mid - y_coarse[ind_c3]) / abs(y_coarse[ind_c1] - y_coarse[ind_c3] + 1e-100))
            elif ind_c1 == ind_c3:
                f1 = 0.5 * (1.0 - abs(y_mid - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c2] + 1e-100))
                f3 = 0.5 * (1.0 - abs(y_mid - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c2] + 1e-100))
                f2 = (1.0 - abs(y_mid - y_coarse[ind_c2]) / abs(y_coarse[ind_c1] - y_coarse[ind_c2] + 1e-100))

        if y_fine[max(ir_fst + i - 1, 0)] == y_coarse[ind_c2]:
            f1 += f2 / 2.0
            f3 += f2 / 2.0
            f2 = 0.0
        if y_fine[min(ir_fst + i + 1, y_fine.size - 1)] == y_coarse[ind_c3]:
            f1 += f3 / 2.0
            f2 += f3 / 2.0
            f3 = 0.0

        if ir1 < 0 or ir1 > ny_c - 1:
            f1 = 0.0
        if ir2 < 0 or ir2 > ny_c - 1:            
            f2 = 0.0
        if ir3 < 0 or ir3 > ny_c - 1:            
            f3 = 0.0

        if ir1 == ir2 and ir1 == ir3:
            data_sub.append(f1 + f2 + f3)
            row_ids_sub.append(j + ir1 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
        elif ir1 == ir2:
            data_sub.append(f1 + f2)
            row_ids_sub.append(j + ir1 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(j + ir3 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
        elif ir1 == ir3:
            data_sub.append(f1 + f3)
            row_ids_sub.append(j + ir1 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(j + ir2 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
        else:
            data_sub.append(f1)
            row_ids_sub.append(j + ir1 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(j + ir2 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(j + ir3 * nx_c + k * ny_c * nx_c)
            col_ids_sub.append(n)

        for ind in list(row_ids_sub):
           if ind < 0 or ind > nrows - 1:
                pos = row_ids_sub.index(ind)
                row_ids_sub.remove(ind)
                data_sub.remove(data_sub[pos])
                col_ids_sub.remove(col_ids_sub[pos])
                 
        data.extend(data_sub)
        row_ids.extend(row_ids_sub)
        col_ids.extend(col_ids_sub)

    RESTY = csr_matrix((data, (row_ids, col_ids)), size)

    nrows = nx_c * ny_c * nz_c
    ncols = ny_c * nx_c * nz_f
    size = [nrows, ncols]


    data = []
    row_ids = []
    col_ids = []

    for n in xrange(0, nx_c * ny_c * nz_f):
        k, i, j = retrieve_kij(nz_f, ny_c, nx_c, n)

        z_mid = z_fine[k]
        z_low = z_fine[max(k - 1, 0)]
        z_high = z_fine[min(k + 1, z_fine.size - 1)]

        kk1 = np.argwhere(np.logical_and(z2_coarse[1:] > z_mid, z2_coarse[:-1] < z_mid))[0][0]
        if z_mid < z_coarse[kk1]:
            kk2 = max(kk1 - 1, 0)
            kk3 = kk1
        elif z_mid > z_coarse[kk1]:
            kk2 = kk1
            kk3 = min(kk1 + 1, z_coarse.size - 1)
        else:
            kk2 = kk1
            kk3 = kk1

        row_ids_sub = []
        col_ids_sub = []
        data_sub = []

        if order == 'constant':
            f1 = 1.0
            f2 = 0.0
            f3 = 0.0

        elif order == 'linear':
            if kk1 == kk3 and kk1 == kk2 or kk1 != kk3 and kk1 != kk2:
                f1 = 1.0
                f2 = 0.0
                f3 = 0.0
            elif kk1 == kk2:
                f1 = 0.5 * (1.0 - abs(z_mid - z_coarse[kk1]) / abs(z_coarse[kk1] - z_coarse[kk3] + 1e-100))
                f2 = 0.5 * (1.0 - abs(z_mid - z_coarse[kk1]) / abs(z_coarse[kk1] - z_coarse[kk3] + 1e-100))
                f3 = (1.0 - abs(z_mid - z_coarse[kk3]) / abs(z_coarse[kk1] - z_coarse[kk3] + 1e-100))
            elif kk1 == kk3:
                f1 = 0.5 * (1.0 - abs(z_mid - z_coarse[kk1]) / abs(z_coarse[kk1] - z_coarse[kk2] + 1e-100))
                f3 = 0.5 * (1.0 - abs(z_mid - z_coarse[kk1]) / abs(z_coarse[kk1] - z_coarse[kk2] + 1e-100))
                f2 = (1.0 - abs(z_mid - z_coarse[kk2]) / abs(z_coarse[kk1] - z_coarse[kk2] + 1e-100))

        if z_fine[max(k - 1, 0)] == z_coarse[kk2]:
            f1 += f2 / 2.0 
            f3 += f2 / 2.0
            f2 = 0.0
        if z_fine[min(k + 1, z_fine.size - 1)] == z_coarse[kk3]:
            f1 += f3 / 2.0
            f2 += f3 / 2.0
            f3 = 0.0

        if kk1 < 0 or kk1 > nz_c - 1:
            f1 = 0.0
        if kk2 < 0 or kk2 > nz_c - 1:
            f2 = 0.0
        if kk3 < 0 or kk3 > nz_c - 1:
            f3 = 0.0

        if kk1 == kk2 and kk1 == kk3:
            data_sub.append(f1 + f2 + f3)
            row_ids_sub.append(j + i * nx_c + kk1 * ny_c * nx_c)
            col_ids_sub.append(n)
        elif kk1 == kk2:
            data_sub.append(f1 + f2)
            row_ids_sub.append(j + i * nx_c + kk1 * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(j + i * nx_c + kk3 * ny_c * nx_c)
            col_ids_sub.append(n)
        elif kk1 == kk3:
            data_sub.append(f1 + f3)
            row_ids_sub.append(j + i * nx_c + kk1 * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(j + i * nx_c + kk2 * ny_c * nx_c)
            col_ids_sub.append(n)
        else:
            data_sub.append(f1)
            row_ids_sub.append(j + i * nx_c + kk1 * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f2)
            row_ids_sub.append(j + i * nx_c + kk2 * ny_c * nx_c)
            col_ids_sub.append(n)
            data_sub.append(f3)
            row_ids_sub.append(j + i * nx_c + kk3 * ny_c * nx_c)
            col_ids_sub.append(n)

        for ind in list(row_ids_sub):
            if ind < 0 or ind > nrows - 1:
                pos = row_ids_sub.index(ind)
                row_ids_sub.remove(ind)
                data_sub.remove(data_sub[pos])
                col_ids_sub.remove(col_ids_sub[pos])

        data.extend(data_sub)        
        row_ids.extend(row_ids_sub)
        col_ids.extend(col_ids_sub)

    RESTZ = csr_matrix((data, (row_ids, col_ids)), size)
    REST = RESTZ * RESTY * RESTX
    return REST



def REST_T_weight_scaling(edges_fine, edges_coarse, dom_bnds):
    """ 
    Constructs the cell-scaling matrix needed, when one of the interpolation
    operators is based on the transpose of the other.
    or alternatively in the Galerkin process to derive the coarse-grid.
    
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid
    dom_bnds... lateral boundary indices of the interpolation (defined on the global grid)
     
    SCAL... returned scaling matrix
    """


    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]

    x_fine = 0.5 * (x2_fine[1:] + x2_fine[:-1])
    y_fine = 0.5 * (y2_fine[1:] + y2_fine[:-1])
    z_fine = 0.5 * (z2_fine[1:] + z2_fine[:-1])

    x_coarse = 0.5 * (x2_coarse[1:] + x2_coarse[:-1])
    y_coarse = 0.5 * (y2_coarse[1:] + y2_coarse[:-1])
    z_coarse = 0.5 * (z2_coarse[1:] + z2_coarse[:-1])

    nz_f = z2_fine.size - 1
    nz_c = z2_coarse.size - 1

    dom_bnds_in, dom_bnds_out = dom_bnds[:]

    ir_tpl_in, jc_tpl_in = dom_bnds_in[:]
    ir_tpl_out, jc_tpl_out = dom_bnds_out[:]

    ir_cst, ir_cend = ir_tpl_in[:]
    jc_cst, jc_cend = jc_tpl_in[:]

    ir_fst, ir_fend = ir_tpl_out[:]
    jc_fst, jc_fend = jc_tpl_out[:]

    ny_c = ir_cend - ir_cst
    nx_c = jc_cend - jc_cst

    ny_f = ir_fend - ir_fst
    nx_f = jc_fend - jc_fst

    nrows = nx_f * ny_f * nz_f
    ncols = ny_c * nx_c * nz_c
    size = [ncols, ncols]

    data = []
    row_ids = []
    col_ids = []

    for m in range(nx_c * ny_c * nz_c):

         k = int(m / (nx_c * ny_c))
         i = int((m - k * nx_c * ny_c) / nx_c) 
         j = m - k * nx_c * ny_c - i * nx_c
         
         
         cells_x = np.argwhere(x2_coarse[jc_cst + j  + 1] == x2_fine)[0][0] - np.argwhere(x2_coarse[jc_cst + j] == x2_fine)[0][0]
         cells_y = np.argwhere(y2_coarse[ir_cst + i  + 1] == y2_fine)[0][0] - np.argwhere(y2_coarse[ir_cst + i] == y2_fine)[0][0] 
         cells_z = np.argwhere(z2_coarse[k  + 1] == z2_fine)[0][0] - np.argwhere(z2_coarse[k] == z2_fine)[0][0]
         data.append(1.0 / float(cells_x * cells_y * cells_z))
         row_ids.append(m)
         col_ids.append(m)

    SCAL = csr_matrix((data, (row_ids, col_ids)), size)
    return SCAL



def make_prol_volume(edges_fine, edges_coarse, dom_bnds, fvol, type='standard'):
    """ 
    Constructs the interpolation operator used for restriction of the residual,
    or alternatively in the Galerkin process to derive the coarse-grid
    approximation of the fine-grid operator. This special version uses the volume-law
    to derive a linear approximation. The volume-scaling field fvol can be used 
    as weights to enforce constant interpolation near obstacles.
    
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid
    dom_bnds... lateral boundary indices of the interpolation (defined on the global grid)
    fvol... scaling field of fine-grid volumes     

    PROL... returned interpolation matrix
    """


    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]
    
    x_fine = 0.5 * (x2_fine[1:] + x2_fine[:-1])
    y_fine = 0.5 * (y2_fine[1:] + y2_fine[:-1])
    z_fine = 0.5 * (z2_fine[1:] + z2_fine[:-1])

    x_coarse = 0.5 * (x2_coarse[1:] + x2_coarse[:-1])
    y_coarse = 0.5 * (y2_coarse[1:] + y2_coarse[:-1]) 
    z_coarse = 0.5 * (z2_coarse[1:] + z2_coarse[:-1])
    
    nz_f = z2_fine.size - 1
    nz_c = z2_coarse.size - 1

    dom_bnds_in, dom_bnds_out = dom_bnds[:]

    ir_tpl_in, jc_tpl_in = dom_bnds_in[:]
    ir_tpl_out, jc_tpl_out = dom_bnds_out[:]

    ir_cst, ir_cend = ir_tpl_in[:]
    jc_cst, jc_cend = jc_tpl_in[:]

    ir_fst, ir_fend = ir_tpl_out[:]
    jc_fst, jc_fend = jc_tpl_out[:]

    ny_c = ir_cend - ir_cst
    nx_c = jc_cend - jc_cst

    ny_f = ir_fend - ir_fst
    nx_f = jc_fend - jc_fst

    nrows = nx_f * ny_f * nz_f
    ncols = ny_c * nx_c * nz_c
    size = [nrows, ncols]

    data = []
    row_ids = []
    col_ids = []

    time_do_smth = 0.0
    st_time_total = time()
    
    if type == 'Galerkin':
       bnd_ind = 1
    else:
       bnd_ind = 0

    for m in xrange(0, nx_f * ny_f * nz_f):

        k, i, j = retrieve_kij(nz_f, ny_f, nx_f, m)
        x_center = x_fine[jc_fst + j]
        y_center = y_fine[ir_fst + i]
        z_center = z_fine[k]

        st_time = time()
#        ind_j1 = np.argmin(np.absolute(x_center - x_coarse))
#        ind_i1 = np.argmin(np.absolute(y_center - y_coarse))
#        ind_k1 = np.argmin(np.absolute(z_center - z_coarse))
        ind_j1 = np.argwhere(np.logical_and(x_center > x2_coarse[:-1], x_center < x2_coarse[1:]))[0][0] 
        ind_i1 = np.argwhere(np.logical_and(y_center > y2_coarse[:-1], y_center < y2_coarse[1:]))[0][0]
        ind_k1 = np.argwhere(np.logical_and(z_center > z2_coarse[:-1], z_center < z2_coarse[1:]))[0][0]

             
        if x_center < x_coarse[ind_j1]:
            ind_j2 = ind_j1
            ind_j1 = max(ind_j2 - 1, bnd_ind)
        elif x_center > x_coarse[ind_j1]:
            ind_j2 = min(ind_j1 + 1, x_coarse.size - 1 - bnd_ind)
        else:
            ind_j2 = ind_j1
        if y_center < y_coarse[ind_i1]:
            ind_i2 = ind_i1
            ind_i1 = max(ind_i2 - 1, bnd_ind)
        elif y_center > y_coarse[ind_i1]:
            ind_i2 = min(ind_i1 + 1, y_coarse.size - 1 - bnd_ind)
        else:
            ind_i2 = ind_i1
        if z_center < z_coarse[ind_k1]:
            ind_k2 = ind_k1
            ind_k1 = max(ind_k2 - 1, bnd_ind)
        elif z_center > z_coarse[ind_k1]:
            ind_k2 = min(ind_k1 + 1, z_coarse.size - 1 - bnd_ind)
        else:
            ind_k2 = ind_k1 


        if ind_k1 == ind_k2:
            z_low, z_high = z2_fine[k], z2_fine[k + 1]
        else:
            z_low, z_high = z_coarse[ind_k1], z_coarse[ind_k2]
        if ind_i1 == ind_i2:
            y_low, y_high = y2_fine[i + ir_fst], y2_fine[i + ir_fst + 1]
        else:
            y_low, y_high = y_coarse[ind_i1], y_coarse[ind_i2]
        if ind_j1 == ind_j2: 
            x_low, x_high = x2_fine[j + jc_fst], x2_fine[j + jc_fst + 1]
        else:
            x_low, x_high = x_coarse[ind_j1], x_coarse[ind_j2]


        ind_kl = ind_k1
        ind_kh = ind_k2
        ind_il = ind_i1
        ind_ih = ind_i2
        ind_jl = ind_j1
        ind_jh = ind_j2
           
        cell_coarse = [z_low, z_high, y_low, y_high, x_low, x_high]
        
        k_fine_s = np.argmin(np.absolute(z2_fine - z2_coarse[ind_k1]))
        k_fine_e = np.argmin(np.absolute(z2_fine - z2_coarse[ind_k2 + 1]))
        i_fine_s = np.argmin(np.absolute(y2_fine - y2_coarse[ind_i1]))
        i_fine_e = np.argmin(np.absolute(y2_fine - y2_coarse[ind_i2 + 1]))
        j_fine_s = np.argmin(np.absolute(x2_fine - x2_coarse[ind_j1]))
        j_fine_e = np.argmin(np.absolute(x2_fine - x2_coarse[ind_j2 + 1]))

        cells_fine_fvol = np.ones([k_fine_e - k_fine_s, i_fine_e - i_fine_s, j_fine_e - j_fine_s], dtype=float)
#        cells_fine_fvol = fvol[k_fine_s:k_fine_e, i_fine_s:i_fine_e, j_fine_s:j_fine_e]               

        cells_fine_kl = (z2_fine[k_fine_s:k_fine_e]).reshape(k_fine_s - k_fine_e, 1, 1)
        cells_fine_il = (y2_fine[i_fine_s:i_fine_e]).reshape(1, i_fine_s - i_fine_e, 1)
        cells_fine_jl = (x2_fine[j_fine_s:j_fine_e]).reshape(1, 1, j_fine_s - j_fine_e)
        cells_fine_kh = (z2_fine[k_fine_s + 1:k_fine_e + 1]).reshape(k_fine_s - k_fine_e, 1, 1)
        cells_fine_ih = (y2_fine[i_fine_s + 1:i_fine_e + 1]).reshape(1, i_fine_s - i_fine_e, 1)
        cells_fine_jh = (x2_fine[j_fine_s + 1:j_fine_e + 1]).reshape(1, 1, j_fine_s - j_fine_e)    
                      
        cells_fine = [cells_fine_kl, cells_fine_kh, cells_fine_il, cells_fine_ih, cells_fine_jl, cells_fine_jh, cells_fine_fvol]
#        volume_coarse = calc_volume_intersection(cell_coarse, cells_fine)                        

        fvol_min, fvol_max = np.min(cells_fine_fvol), np.max(cells_fine_fvol)
       
        cell_tne = [z_fine[k], z_high, y_fine[i + ir_fst], y_high, x_fine[j + jc_fst], x_high]
        vol_tne, vol_eff_tne = calc_volume_intersection(cell_tne, cells_fine)

        cell_tnw = [z_fine[k], z_high, y_fine[i + ir_fst], y_high, x_low, x_fine[j + jc_fst]]
        vol_tnw, vol_eff_tnw = calc_volume_intersection(cell_tnw, cells_fine)

        cell_tse = [z_fine[k], z_high, y_low, y_fine[i + ir_fst], x_fine[j + jc_fst], x_high]
        vol_tse, vol_eff_tse = calc_volume_intersection(cell_tse, cells_fine)

        cell_tsw = [z_fine[k], z_high, y_low, y_fine[i + ir_fst], x_low, x_fine[j + jc_fst]]
        vol_tsw, vol_eff_tsw = calc_volume_intersection(cell_tsw, cells_fine)

        cell_bne = [z_low, z_fine[k], y_fine[i + ir_fst], y_high, x_fine[j + jc_fst], x_high]
        vol_bne, vol_eff_bne = calc_volume_intersection(cell_bne, cells_fine)

        cell_bnw = [z_low, z_fine[k], y_fine[i + ir_fst], y_high, x_low, x_fine[j + jc_fst]]
        vol_bnw, vol_eff_bnw = calc_volume_intersection(cell_bnw, cells_fine)

        cell_bse = [z_low, z_fine[k], y_low, y_fine[i + ir_fst], x_fine[j + jc_fst], x_high]
        vol_bse, vol_eff_bse = calc_volume_intersection(cell_bse, cells_fine)

        cell_bsw = [z_low, z_fine[k], y_low, y_fine[i + ir_fst], x_low, x_fine[j + jc_fst]]
        vol_bsw, vol_eff_bsw = calc_volume_intersection(cell_bsw, cells_fine) 

        vols = [vol_tne, vol_tnw, vol_tse, vol_tsw, vol_bne, vol_bnw, vol_bse, vol_bsw]        
        vols_eff = [vol_eff_tne, vol_eff_tnw, vol_eff_tse, vol_eff_tsw, vol_eff_bne, vol_eff_bnw, vol_eff_bse, vol_eff_bsw]

        name_spc = ['tne', 'tnw', 'tse', 'tsw', 'bne', 'bnw', 'bse', 'bsw']
       
        if ind_kl == ind_kh:
            name_spc_tmp = [name.replace('t', '') for name in name_spc if 't' in name]
            vols_tmp = []
            vols_eff_tmp = []
            for name in name_spc_tmp:
                vols_tmp.append(sum([w for y, w in enumerate(vols) if all((letter in name_spc[y] for letter in name))]))
                vols_eff_tmp.append(sum([w for y, w in enumerate(vols_eff) if all((letter in name_spc[y] for letter in name))]))
            name_spc = name_spc_tmp
            vols = vols_tmp
            vols_eff = vols_eff_tmp

        if ind_il == ind_ih:
            name_spc_tmp = [name.replace('n', '') for name in name_spc if 'n' in name]
            vols_tmp = []
            vols_eff_tmp = []
            for name in name_spc_tmp:
                vols_tmp.append(sum([w for y, w in enumerate(vols) if all((letter in name_spc[y] for letter in name))]))
                vols_eff_tmp.append(sum([w for y, w in enumerate(vols_eff) if all((letter in name_spc[y] for letter in name))]))
            name_spc = name_spc_tmp
            vols = vols_tmp
            vols_eff = vols_eff_tmp

        if ind_jl == ind_jh:
            vols_tmp = []
            vols_eff_tmp = []
            name_spc_tmp = [name.replace('e', '') for name in name_spc if 'e' in name]
            for name in name_spc_tmp:
                vols_tmp.append(sum([w for y, w in enumerate(vols) if all((letter in name_spc[y] for letter in name))]))
                vols_eff_tmp.append(sum([w for y, w in enumerate(vols_eff) if all((letter in name_spc[y] for letter in name))]))
            name_spc = name_spc_tmp
            vols = vols_tmp
            vols_eff = vols_eff_tmp

        arg_vol_max = np.argmax(vols)
        arg_vol_min = np.argmin(vols)
        fvols = np.array(vols_eff) / np.array(vols)        
        fvol_std = np.std(fvols)
        fvol_ref = fvols[arg_vol_max]

        name_spc_rec = [invert_string(name) for name in name_spc]

        weights = [0.0 for n in range(len(vols))]
        if fvol_max -  fvol_min > 0.30:
            fac = 0.10
        else:
            fac = 1.0
        for n, volume in enumerate(vols):
            name = invert_string(name_spc[n])
            index = name_spc_rec.index(name)
#            weight = volume * np.exp(-abs(fvols[index] - fvol_ref) * 20)
            
            if n == arg_vol_max:
                 if fac == 0.1:
                    weight = volume * 0.5
                 else:
                    weight = volume
            else:
                
                weight = volume * fac
            weights[index] = weight 

        sum_weights = np.sum(weights)
        weights = (weights / sum_weights).tolist()        
       
        nz_sub = ind_kh + 1 - ind_kl
        ny_sub = ind_ih + 1 - ind_il
        nx_sub = ind_jh + 1 - ind_jl

        n = 0
        for r in range(ind_kl, ind_kh + 1):
            for s in range(ind_il, ind_ih + 1):
                for t in range(ind_jl, ind_jh + 1):           
                    s -= ir_fst
                    t -= jc_fst
                    cond = [
                               weights[n] != 0, r >= 0, r <= nz_c - 1,
                               s - ir_cst >= 0, s <= ny_c - 1,
                               t - jc_cst >= 0, t <= nx_c - 1
                           ]
                    if all(cond):
                        data.append(weights[n])
                        row_ids.append(m)
                        col_ids.append(r * ny_c * nx_c + s * nx_c + t)
                    n += 1

    PROL = csr_matrix((data, (row_ids, col_ids)), size)
    return PROL


def make_prol(edges_fine, edges_coarse, dom_bnds, order='constant', type='standard'):
    """ 
    Constructs the interpolation operator used for prolongation of the coarse-grid
    correction, or alternatively in the Galerkin process to derive the coarse-grid
    approximation of the fine-grid operator.
    
    edges_fine... grid-plane coordinates of fine grid
    edges_coarse... grid-plane coordinates of coarse grid
    dom_bnds... lateral boundary indices of the interpolation (defined on the global grid)
    order... order of  interpolation (constant or linear)
    type... Galerkin or standard

    PROL... returned interpolation matrix
    """

    x2_fine, y2_fine, z2_fine = edges_fine[:]
    x2_coarse, y2_coarse, z2_coarse = edges_coarse[:]
    
    x_fine = 0.5 * (x2_fine[1:] + x2_fine[:-1])
    y_fine = 0.5 * (y2_fine[1:] + y2_fine[:-1])
    z_fine = 0.5 * (z2_fine[1:] + z2_fine[:-1])

    x_coarse = 0.5 * (x2_coarse[1:] + x2_coarse[:-1])
    y_coarse = 0.5 * (y2_coarse[1:] + y2_coarse[:-1])
    z_coarse = 0.5 * (z2_coarse[1:] + z2_coarse[:-1])

    nz_f = z2_fine.size - 1
    nz_c = z2_coarse.size - 1

    dom_bnds_in, dom_bnds_out = dom_bnds[:]

    ir_tpl_in, jc_tpl_in = dom_bnds_in[:]
    ir_tpl_out, jc_tpl_out = dom_bnds_out[:]

    ir_cst, ir_cend = ir_tpl_in[:]
    jc_cst, jc_cend = jc_tpl_in[:]    

    ir_fst, ir_fend = ir_tpl_out[:]
    jc_fst, jc_fend = jc_tpl_out[:]

    ny_c = ir_cend - ir_cst
    nx_c = jc_cend - jc_cst

    ny_f = ir_fend - ir_fst
    nx_f = jc_fend - jc_fst

    nrows = nx_f * ny_c * nz_c
    ncols = ny_c * nx_c * nz_c
    size = [nrows, ncols]

    data = []
    row_ids = []
    col_ids = []

    if type == 'Galerkin':
       bnd_ind = 1
    else:
       bnd_ind = 0

    for m in xrange(0, nx_f * ny_c * nz_c):
        k, i, j = retrieve_kij(nz_c, ny_c, nx_f, m)
        x_center = x_fine[jc_fst + j]

        ind_c1 = np.argwhere(np.logical_and(x_center > x2_coarse[:-1], x_center < x2_coarse[1:]))[0][0]

        if order == 'constant':
           f1 = 1.0
           f2 = 0.0
           ind_c2 = ind_c1

        elif order == 'linear':
            if x_center < x_coarse[ind_c1]:
                ind_c2 = max(ind_c1 - 1, bnd_ind)
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(x_center - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c2])                
                    f2 = 1.0 - f1
            elif x_center > x_coarse[ind_c1]:
                ind_c2 = min(ind_c1 + 1, x_coarse.size - 1 - bnd_ind)
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(x_center - x_coarse[ind_c1]) / abs(x_coarse[ind_c1] - x_coarse[ind_c2])        
                    f2 = 1.0 - f1
            else:
                f1 = 1.0
                ind_c2 = ind_c1
                f2 = 0.0

        cells_int = float(np.argwhere(x2_coarse[ind_c1 + 1] == x2_fine)[0][0] - np.argwhere(x2_coarse[ind_c1] == x2_fine)[0][0])

        weight1 = 1.0 / cells_int * f1
        weight2 = 1.0 / cells_int * f2

        if type == 'Galerkin':
            if x_center == x_fine[0]:
                weight1 = (x2_fine[-2] - x2_fine[-3]) / (x2_coarse[-2] - x2_coarse[-3])
            if x_center == x_fine[-1]:
                weight1 = (x2_fine[2] - x2_fine[1]) / (x2_coarse[2] - x2_coarse[1])

        if all((weight1 != 0, ind_c1 - jc_cst >=  0, ind_c1 - jc_cst <=  nx_c - 1)):
            data.append(weight1)
            row_ids.append(m)
            col_ids.append(k * ny_c * nx_c + i * nx_c + ind_c1 - jc_cst)
        if all((weight2 != 0, ind_c2 - jc_cst >=  0, ind_c2 - jc_cst <=  nx_c - 1)):
            data.append(weight2)
            row_ids.append(m)
            col_ids.append(k * ny_c * nx_c + i * nx_c + ind_c2 - jc_cst)

    PROLX = csr_matrix((data, (row_ids, col_ids)), size)

    nrows = nx_f * ny_f * nz_c
    ncols = ny_c * nx_f * nz_c
    size = [nrows, ncols]


    data = []
    row_ids = []
    col_ids = []


    for m in xrange(0, nx_f * ny_f * nz_c):
        k, i, j = retrieve_kij(nz_c, ny_f, nx_f, m)
        y_center = y_fine[ir_fst + i]

        ind_c1 = np.argwhere(np.logical_and(y_center > y2_coarse[:-1], y_center < y2_coarse[1:]))[0][0]        

        if order == 'constant':
           f1 = 1.0
           f2 = 0.0
        elif order == 'linear':
            if y_center < y_coarse[ind_c1]:
                ind_c2 = max(ind_c1 - 1, bnd_ind)
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(y_center - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c2])                
                    f2 = 1.0 - f1
            elif y_center > y_coarse[ind_c1]:
                ind_c2 = min(ind_c1 + 1, y_coarse.size - 1 - bnd_ind)
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(y_center - y_coarse[ind_c1]) / abs(y_coarse[ind_c1] - y_coarse[ind_c2])        
                    f2 = 1.0 - f1
            else:
                f1 = 1.0
                ind_c2 = ind_c1
                f2 = 0.0

        cells_int = float(np.argwhere(y2_coarse[ind_c1 + 1] == y2_fine)[0][0] - np.argwhere(y2_coarse[ind_c1] == y2_fine)[0][0])

        weight1 = 1.0 / cells_int * f1
        weight2 = 1.0 / cells_int * f2

        if type == 'Galerkin':
            if y_center == y_fine[0]:        
                weight1 = (y2_fine[-2] - y2_fine[-3]) / (y2_coarse[-2] - y2_coarse[-3])
            if y_center == y_fine[-1]:
                weight1 = (y2_fine[2] - y2_fine[1]) / (y2_coarse[2] - y2_coarse[1])


        if all((weight1 != 0, ind_c1 - ir_cst >=  0, ind_c1 - ir_cst <=  ny_c - 1)):
            data.append(weight1)
            row_ids.append(m)
            col_ids.append(k * ny_c * nx_f + (ind_c1 - ir_cst) * nx_f + j)
        if all((weight2 != 0, ind_c2 - ir_cst >=  0, ind_c2 - ir_cst <=  ny_c - 1)):
            data.append(weight2)
            row_ids.append(m)
            col_ids.append(k * ny_c * nx_f + (ind_c2 - ir_cst) * nx_f + j)
       

    PROLY = csr_matrix((data, (row_ids, col_ids)), size)

    nrows = nx_f * ny_f * nz_f
    ncols = ny_f * nx_f * nz_c
    size = [nrows, ncols]

    data = []
    row_ids = []
    col_ids = []                
    for m in xrange(0, nx_f * ny_f * nz_f):
        k, i, j = retrieve_kij(nz_f, ny_f, nx_f, m)
        z_center = z_fine[k]

        ind_c1 = np.argwhere(np.logical_and(z_center > z2_coarse[:-1], z_center < z2_coarse[1:]))[0][0]

        if order == 'constant':
           f1 = 1.0
           f2 = 0.0
        elif order == 'linear':
            if z_center < z_coarse[ind_c1]:
                ind_c2 = max(ind_c1 - 1, bnd_ind)    
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(z_center - z_coarse[ind_c1]) / abs(z_coarse[ind_c1] - z_coarse[ind_c2])
                    f2 = 1.0 - f1
            elif z_center > z_coarse[ind_c1]:
                ind_c2 = min(ind_c1 + 1, z_coarse.size - 1 - bnd_ind)
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
                else:
                    f1 = 1.0 - abs(z_center - z_coarse[ind_c1]) / abs(z_coarse[ind_c1] - z_coarse[ind_c2])        
                    f2 = 1.0 - f1
                if ind_c1  == ind_c2:
                    f1 = 1.0
                    f2 = 0.0
            else:
                f1 = 1.0
                ind_c2 = ind_c1
                f2 = 0.0

        cells_int = float(np.argwhere(z2_coarse[ind_c1 + 1] == z2_fine)[0][0] - np.argwhere(z2_coarse[ind_c1] == z2_fine)[0][0])

        weight1 = 1.0 / cells_int * f1
        weight2 = 1.0 / cells_int * f2
    
        if type == 'Galerkin':
            if z_center == z_fine[0]:
                if (z2_fine[-2] - z2_fine[-3]) < (z2_coarse[-2] - z2_coarse[-3]):
                    weight1 = 0.5
                else:
                    weight1 = 1.0
            if z_center == z_fine[-1]:
                if (z2_fine[2] - z2_fine[1]) < (z2_coarse[2] - z2_coarse[1]):
                    weight1 = 0.5
                else:
                    weight1 = 1.0

        if all((weight1 != 0, ind_c1 >=  0, ind_c1 <=  nz_c - 1)):
            data.append(weight1)
            row_ids.append(m)
            col_ids.append(ind_c1 * ny_f * nx_f + i * nx_f + j)
        if all((weight2 != 0, ind_c2 >=  0, ind_c2 <=  nz_c - 1)):
            data.append(weight2)
            row_ids.append(m)
            col_ids.append(ind_c2 * ny_f * nx_f + i * nx_f + j)

    PROLZ = csr_matrix((data, (row_ids, col_ids)), size)


    PROL = PROLZ * PROLY * PROLX

    return PROL



def vol_stag(vol1d, comp, nz, ny, nx):
    """
    interpolates the cell volumes on the stagered grid
    for the gradient operator using arithmetic averaging
    
    vol1d... input volumes in flattened shape
    comp... gradient component (can be either 'u', 'v' or 'w') 
    nz, ny, nx... 3d shape of the volumes field

    vol1d_stag... returned averaged volumes in flattened shape
    """  

    vol3d = vol1d.reshape(nz, ny, nx)

    if comp == 'u':
        vol3d_stag = np.empty([nz, ny, nx + 1])
        vol3d_stag[:, :, 1:-1] = 0.5 * (vol3d[:, :, 1:] + vol3d[:, :, :-1])
        vol3d_stag[:, :, 0] = vol3d[:, :, 0]
        vol3d_stag[:, :, -1] = vol3d[:, :, -1]
        vol1d_stag = np.zeros([(nx + 1) * ny * nz + (ny + 1) * nx * nz + (nz + 1) * ny * nx])
        vol1d_stag[:(nx + 1) * ny * nz] = vol3d_stag.flatten()
    if comp == 'v':
        vol3d_stag = np.empty([nz, ny + 1, nx])
        vol3d_stag[:, 1:-1] = 0.5 * (vol3d[:, 1:] + vol3d[:, :-1])
        vol3d_stag[:, 0] = vol3d[:, 0]
        vol3d_stag[:, -1] = vol3d[:, -1]
        vol1d_stag = np.zeros([(nx + 1) * ny * nz + (ny + 1) * nx * nz + (nz + 1) * ny * nx])
        vol1d_stag[(nx + 1) * ny * nz:(nx + 1) * ny * nz + (ny + 1) * nx * nz] = vol3d_stag.flatten()
    if comp == 'w':
        vol3d_stag = np.empty([nz + 1, ny, nx])
        vol3d_stag[1:-1] = 0.5 * (vol3d[1:] + vol3d[:-1])
        vol3d_stag[0] = vol3d[0]
        vol3d_stag[-1] = vol3d[-1]
        vol1d_stag = np.zeros([(nx + 1) * ny * nz + (ny + 1) * nx * nz + (nz + 1) * ny * nx])
        vol1d_stag[(nx + 1) * ny * nz + (ny + 1) * nx * nz:] = vol3d_stag.flatten()
    return vol1d_stag


def rescale_smoother(MAT, SPAI):
    """
    Applies a Frobenius-norm rescaling to 
    retrieve a new smoother for a slightly modified MAT.

    MAT... matrix whose inverse is approximated
    SPAI... smoother to be rescaled

    SPAI_new... returned rescaled smoother
    """

    mat_norm = norm(MAT, axis=1)
    mat_norm_inv = 1.0 / (mat_norm + 1e-100) * np.array(np.array(mat_norm, dtype=bool), dtype=float)

    spai_norm = norm(SPAI, axis=1)
    spai_norm_inv = 1.0 / (spai_norm + 1e-100) * np.array(np.array(spai_norm, dtype=bool), dtype=float)

    NORMinv_MAT = diags(mat_norm_inv, format='csr')
    NORMinv_SMO = diags(spai_norm_inv, format='csr')

    SPAI_new = NORM_MATinv * NORMinv_SMO * SPAI

    return SPAI_new



def make_SPAIE(SPAI, A, n_update = 1, n_add=2):
    """
    Improves a sparse approximate inverse
    using the SPAI algorithm (Huckle et al.,1995).
   
    SPAI... initial sparse approximate inverse
    A... matrix whose inverse is approximated
    n_update... number of update steps 
    n_add... number of indices per row to add at each update

    SPAI_new... returned new sparse approximate inverse
    """

    shape = A.shape
    data = A.data.tolist()
    indices = A.indices.tolist()
    indptr = A.indptr.tolist()
    A_k = norm(A, axis=1)
    A_T = A.transpose(copy=True)
    data_T = A_T.data.tolist()
    indices_T = A_T.indices.tolist()
    indptr_T = A_T.indptr.tolist()
    start_time = time()
    data_sp = SPAI.data.tolist()
    indices_sp = SPAI.indices.tolist()
    intdptr_sp = SPAI.indptr.tolist()
    data_sp_new = []
    row_sp_new = []
    col_sp_new = []
    cont = False
    for i in xrange(0, A.shape[0], 1):
        st_time = time()
        row_i_SM = indices_sp[intdptr_sp[i]:intdptr_sp[i + 1]]
        data_i_SM = data_sp[intdptr_sp[i]:intdptr_sp[i + 1]]
        for n in range(n_update):
            inds_nz = []
            inds_pot = []
            col_ids_A = []
            for k, ind in enumerate(row_i_SM):
                col_ids_A.extend(indices[indptr[ind]:indptr[ind + 1]])

            col_ids_A = set(col_ids_A)
            res = []
            for k, ind in enumerate(col_ids_A):
                val = dot_sp(row_i_SM, data_i_SM, indices_T[indptr_T[ind]:indptr_T[ind + 1]], data_T[indptr_T[ind]:indptr_T[ind + 1]]) - (i == ind)
                if val != 0:
                    inds_nz.append(ind)
                    res.append(val)

            if not len(res):
                cont = True
                break
            for ind in inds_nz:
                inds_pot.extend(indices_T[indptr_T[ind]:indptr_T[ind + 1]])

            st_time = time()
            rsq = sum([ val ** 2 for val in res ])
            inds_pot = set(inds_pot)
            inds_pot_new = [ ind for ind in inds_pot if ind not in row_i_SM ]
            res_pot = [ 
                          rsq - 
                          dot_sp(indices[indptr[ind]:indptr[ind + 1]], data[indptr[ind]:indptr[ind + 1]], inds_nz, res) ** 2 / 
                          A_k[ind] ** 2 for ind in inds_pot_new 
                      ]
            res_pot_inds_min = np.argsort(res_pot, kind='mergesort')
            update_inds = [inds_pot_new[res_pot_inds_min[n]] for n in range(min(n_add, len(res_pot))) ]
            row_i_SM.extend(update_inds)
            st_time = time()
            data_sub = []
            rowids_sub = []
            colids_sub = []
            ncols = len(row_i_SM)
            for j, ind in enumerate(row_i_SM):
                data_sub.extend(data[indptr[ind]:indptr[ind + 1]])
                rowids_sub.extend(indices[indptr[ind]:indptr[ind + 1]])
                colids_sub.extend((j for k in xrange(0, indptr[ind + 1] - indptr[ind], 1)))
#            rowids_sub.append(i)
            row_comp = list(set(rowids_sub))

            nrows = len(row_comp)
            ncols = len(row_i_SM)
            
            A_sub = np.empty([nrows, ncols])
            A_sub.fill(0.0)
            for j, ind in enumerate(rowids_sub):
                row = row_comp.index(ind)
                A_sub[row, colids_sub[j]] = data_sub[j]

            ei = np.empty([nrows])
            ei.fill(0.0)
            ei[row_comp.index(i)] = 1.0
            x = lstsq(A_sub, ei, overwrite_a = True, overwrite_b=True, lapack_driver='gelsy')[0]
            data_i_SM = x

        if cont == True:
            cont = False
            continue
        data_sp_new.extend(x)
        row_sp_new.extend([i for val in x])
        col_sp_new.extend(row_i_SM)

    end_time = time()

    SPAI_new = csr_matrix((data_sp_new, (row_sp_new, col_sp_new)), shape)
    return SPAI_new



def dot_sp(sp_ind, sp_vals, sp_ind2, sp_vals2):
    """
    An implementation for a sparse dot product 
    (needs to be rewritten and compiled with cython for
    good performance).

    sp_ind, sp_ind2... indices of non-zeros in first and second vector
    sp_vals, sp_vals2... nown-zeros in first and second vector    
    sp_dot... returned dot product
    """

    sp_dot = 0.0
    for k, ind in enumerate(sp_ind):
        if ind in sp_ind2:
            pos = sp_ind2.index(ind)
            sp_dot += sp_vals[k] * sp_vals2[pos]
            del sp_ind2[pos]
            del sp_vals2[pos]

    return sp_dot   


def make_interpol_V_F(nz, ny, nx, dims, transpose=False):
    """
    interpolates a scalar cell centred field to
    the cell faces
 
    nz, ny, nx... dimension sizes
    dims... tuple to indicate on which faces to interpolate:
    (1,0,0)  to u-faces
    (0,1,0)  to v-faces
    (0,0,1)  to w-faces
    (1,1,1)  to all faces
    transpose... reverse operation
    """

    col_ids = []
    row_ids = []
    data = []
   
    if not transpose:
        if dims[0] == 1:
            for n in xrange(0,(nx + 1) * ny * nz):
                k = int(n / (ny * (nx + 1)))
                i = int((n - k * ny * (nx + 1)) / (nx + 1))
                j = n - k * ny * (nx + 1) - i * (nx + 1)
                m = k * ny * nx + i * nx + j

                if j == 0:
                    data.append(0.5)
                    col_ids.append(m)
                    row_ids.append(n)
                elif j == nx:
                    data.append(0.5)
                    m = k * ny * nx + i * nx + j - 1
                    col_ids.append(m)
                    row_ids.append(n)
                else:
                    col_ids.append(m)
                    row_ids.append(n)
                    data.append(0.5)
                    row_ids.append(n)
                    m = k * ny * nx + i * nx + j - 1
                    col_ids.append(m)
                    data.append(0.5)

        if dims[1] == 1:
            for n in xrange(0, nx * (ny + 1) * nz):
                k = int(n / ((ny + 1) * nx))
                i = int((n - k * (ny + 1) * nx) / nx)
                j = n - k * (ny + 1) * nx - i * nx
                m = k * ny * nx + i * nx + j            

                if i == 0:
                    col_ids.append(m)
                    row_ids.append(n + (nx + 1) * ny * nz)
                    data.append(0.5)
                elif i == ny:
                    data.append(0.5)
                    m = k * ny * nx + (i - 1) * nx + j
                    col_ids.append(m)
                    row_ids.append(n + (nx + 1) * ny * nz)
                else:
                    col_ids.append(m)
                    row_ids.append(n + (nx + 1) * ny * nz)
                    data.append(0.5)
                    row_ids.append(n + (nx + 1) * ny * nz)
                    m = k * ny * nx + (i - 1) * nx + j
                    col_ids.append(m)
                    data.append(0.5)

        if dims[2] == 1:
            for n in xrange(0, nx * ny * (nz + 1)):
                k = int(n / (ny * nx))
                i = int((n - k * ny * nx) / nx)
                j = n - k * ny * nx - i * nx
                m = k * ny * nx + i * nx + j

                if k <= 1:
                    data.append(1.0)
                    row_ids.append(n + nx * (ny + 1) * nz + (nx + 1) * ny * nz)
                    col_ids.append(m)            
                elif k >= nz - 1:
                    m = (k - 1) * ny * nx + i * nx + j
                    col_ids.append(m)
                    row_ids.append(n + nx * (ny + 1) * nz + (nx + 1) * ny * nz)
                    data.append(1.0)
                else:
                    col_ids.append(m)
                    row_ids.append(n + nx * (ny + 1) * nz + (nx + 1) * ny * nz)
                    data.append(0.5)
                    row_ids.append(n + nx * (ny + 1) * nz + (nx + 1) * ny * nz)
                    m = (k - 1) * ny * nx + i * nx + j 
                    col_ids.append(m)
                    data.append(0.5)

        shape = [(nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1), nx * ny * nz]
    if transpose:
        for m in range(nx * ny * nz):
            k = int(m / (ny * nx))
            i = int((m - k * ny * nx) / nx)
            j = m - k * ny * nx - i * nx      
            if dims[0] == 1:
                data.append(0.5)
                data.append(0.5)
                col_ids.append(find_nx(k, i, j, nz, ny, nx))
                col_ids.append(find_nx(k, i, j + 1, nz, ny, nx))
                row_ids.append(m)
                row_ids.append(m)
            if dims[1] == 1:
                data.append(0.5)
                data.append(0.5)
                col_ids.append(find_ny(k, i, j, nz, ny, nx))
                col_ids.append(find_ny(k, i + 1, j, nz, ny, nx))
                row_ids.append(m)
                row_ids.append(m)
            if dims[2] == 1:
                if k >= nz - 3:
                    data.append(1.0)
                    col_ids.append(find_nz(k, i, j, nz, ny, nx))
                    row_ids.append(m)
                elif k <= 2:
                    data.append(1.0)
                    col_ids.append(find_nz(k + 1, i, j, nz, ny, nx))
                    row_ids.append(m) 
                else:
                    data.append(0.5)
                    data.append(0.5)
                    col_ids.append(find_nz(k, i, j, nz, ny, nx))
                    col_ids.append(find_nz(k + 1, i, j, nz, ny, nx))
                    row_ids.append(m)
                    row_ids.append(m)
        shape = [nx * ny * nz, (nx + 1) * ny * nz + nx * (ny + 1) * nz + nx * ny * (nz + 1)]

    INT_OP = csr_matrix((data,(row_ids, col_ids)), shape)

    return INT_OP


def make_expansion_op(field):
    """
    This function returns an expansion operator to insert
    a smaller scalar field into a larger.
    field... the larger field whose non-zero entries correspond to
    the field part.
    
    operator... returned expansion operator
    """
   
    nnzr = (np.where(field)[0]).size
    
    shape = (field.size, nnzr)
    
    data = []
    row_ids = []
    col_ids = []

    n = 0
    for m in range(field.size):
        if field[m] != 0:
            data.append(1.0)
            col_ids.append(n)         
            row_ids.append(m)
            n += 1

    operator = csr_matrix((data, (row_ids, col_ids)), shape)
    return operator


def make_trans_zs_z(comm, hsurf, nx, ny, param_dict):
    """
    Constructs  linear interpolation matrices
    to interpolate from terrain coordinates to
    z=const levels and reverse.

    comm... communicator
    hsurf... terrain function
    nx, ny... horizontal dimension sizes of subdomain
    param_dict... parameter dictionary
    """

    mpicomm = comm.mpicomm
    rank = mpicomm.Get_rank()
    pids = comm.pids

    hl = param_dict['z2coord']
    dz = param_dict['dz'][0]

    zmin = np.min(hsurf) + hl[0]
    zmax = np.max(hsurf) + hl[0]
    zmin = ddcp.min_para(mpicomm, zmin, pids[1:], pids[0])
    zmax = ddcp.max_para(mpicomm, zmax, pids[1:], pids[0])

    nz = int((zmax + hl[-1] - zmin) / dz)

    hhl_z = np.linspace(zmin, nz * dz, nz + 1)

    hfl_z = 0.5 * (hhl_z[1:] + hhl_z[:-1])
    hhl_s  = hsurf.reshape(1, ny, nx) + hl.reshape(hl.size, 1, 1)
    hfl_s = 0.5 * (hhl_s[1:] + hhl_s[:-1])

    param_dict.update({'zscoord':hfl_z})

    data = []
    col_ids = []
    row_ids = []
    shape = (nz * ny * nx, (hl.size - 1) * ny * nx)

    for n in range(nx * ny * nz):
        k = int(n / (nx * ny))
        i = int((n - k * nx * ny) / nx)
        j = n - k * nx * ny - i * nx

        zind1 = np.argmin(np.absolute(hfl_z[k] - hhl_s[:, i, j])) - 1
        if zind1 < 0 and hfl_z[k] < hhl_s[0, i, j]:
            continue
        zind2 = min([zind1 + 1, hl.size - 2])
        if zind2 > hl.size - 2 and hfl_z[k] > hhl_s[-1, i, j]:
           continue
        zind1 = max([zind1, 0])
        if zind1 == zind2:
            fac1 = 1.0
        else:
            fac1 = (hfl_s[zind2, i, j] - min([hfl_z[k], hfl_s[zind2, i, j]])) / (hfl_s[zind2, i, j] - hfl_s[zind1, i, j] + 1e-50)

        fac2 = 1.0 - fac1
        data.append(fac1)
        data.append(fac2)
        row_ids.append(n)
        row_ids.append(n)
        col_ids.append(zind1 * nx * ny + i * nx + j)
        col_ids.append(zind2 * nx * ny + i * nx + j)

    trans_z_zs = csr_matrix((data, (row_ids, col_ids)), shape)

    trans_zs_z = csr_matrix(trans_z_zs.transpose(copy=True))

    data = trans_zs_z.data
    indptr = trans_zs_z.indptr.tolist()
    norm_c = []

    for i in range(len(indptr[:-1])):
        norm_c.append(np.sum(data[indptr[i]:indptr[i + 1]]))

    norm_c = np.array(norm_c)
    norm_cinv = 1.0 / (norm_c + 1e-100) * np.array(np.array(norm_c, dtype=bool), dtype=float)
    NORMinv = diags(norm_cinv, format='csr')
    trans_zs_z = NORMinv * trans_zs_z

    return trans_z_zs, trans_zs_z, hhl_z


def merge_operators(OP1, OP2):
   """
   This function merges two csr operators of the same shape
   in the sense by selecting the rows with the higher number of nz
   of both operators. Both operators need to have the same shape.  
 
   OP1... first operator part
   OP2... second operator part

   OP_merged... returned merged operator
   """

   nrows = OP1.shape[0]


   data_new = []
   col_inds_new = []
   row_inds_new = []

   data1 = OP1.data.tolist()
   data2 = OP2.data.tolist()

   indptr1 = OP1.indptr.tolist()
   indptr2 = OP2.indptr.tolist()

   indices1 = OP1.indices.tolist()
   indices2 = OP2.indices.tolist()
   
   for m in range(nrows):       
        if indptr1[m + 1] - indptr1[m] > indptr2[m + 1] - indptr2[m]:
            data_new.extend(data1[indptr1[m]:indptr1[m + 1]])
            col_inds_new.extend(indices1[indptr1[m]:indptr1[m + 1]])
            row_inds_new.extend([m for ind in range(indptr1[m + 1] - indptr1[m])])
        else:
            data_new.extend(data2[indptr2[m]:indptr2[m + 1]])
            col_inds_new.extend(indices2[indptr2[m]:indptr2[m + 1]])
            row_inds_new.extend([m for ind in range(indptr2[m + 1] - indptr2[m])])

   OP_merged = csr_matrix((data_new, (row_inds_new, col_inds_new)), OP1.shape)
   return OP_merged


def calc_volume_intersection(cell_coarse, cells_fine):
    """
    Function to derive the volume intersections needed in 
    the 3d prolongation operator based on the volume law.

    cell_coarse... cell geometry and (effective) volume of a coarse cell
    cells_coarse... cell geometries and (effective) volumes of a list of potentially 
                    intersecting fine-grid cells with cell_coarse

    int_volume... returned intersection volume
    int_volume_eff... returned effective intersection volume
    """

    zc_s, zc_e, yc_s, yc_e, xc_s, xc_e = cell_coarse[:6]
    cells_fine_kl, cells_fine_kh, cells_fine_il, cells_fine_ih, cells_fine_jl, cells_fine_jh, volume_weights = cells_fine

    dz = np.maximum(np.minimum(cells_fine_kh, zc_e) - np.maximum(cells_fine_kl, zc_s), 0.0)
    dy = np.maximum(np.minimum(cells_fine_ih, yc_e) - np.maximum(cells_fine_il, yc_s), 0.0)
    dx = np.maximum(np.minimum(cells_fine_jh, xc_e) - np.maximum(cells_fine_jl, xc_s), 0.0)

    vols = dz * dy * dx
    int_volume = np.sum(vols)
    int_volume_eff = np.sum(vols * volume_weights)

    return int_volume, int_volume_eff


def invert_string(string, inversion_rule = {'s':'n', 'n':'s', 'w':'e', 'e':'w', 'b':'t', 't':'b'}):
   """
   Function needed to  assign intersection volumes to opposite grid nodes in the volume law.
   Used to invert relative node locations like 'bsw', tne', ecc.

   string... input string containing exclusively letters defined in inversion_rule
   inversion_rule... dictionary containing the assignment between reciprocal letters.
   string_inv... returned inverted string
   """

   string_inv = ''
   for letter in string:
       string_inv += inversion_rule[letter]
   return string_inv

def find_nx(k, i, j, nz, ny, nx):
    """
    Auxilliary function to find the index position
    of u-face k, i, j in the 1d-expanded field

    k, i, j... index triple of face in 3d-field
    nz, ny, nx... 3d-shape of grid (number of cells)
    """

    return k * (nx + 1) * ny + i * (nx + 1) + j

def find_ny(k, i, j, nz, ny, nx):
    """
    Auxilliary function to find the index position
    of v-face k, i, j in the 1d-expanded field

    k, i, j... index triple of face in 3d-field
    nz, ny, nx... 3d-shape of grid (number of cells)
    """

    return k * (ny + 1) * nx + i * nx + j + (nx + 1) * ny * nz

def find_nz(k, i, j, nz, ny, nx):
    """
    Auxilliary function to find the index position
    of w-face k, i, j in the 1d-expanded field

    k, i, j... index triple of face in 3d-field
    nz, ny, nx... 3d-shape of grid (number of cells)
    """

    return k * nx * ny + i * nx + j + (nx + 1) * ny * nz + nx * (ny + 1) * nz

def find_nlam(k, i, j, nz, ny, nx):
    """
    Auxilliary function to find the index position
    of cell volume k, i, j in the 1d-expanded field

    k, i, j... index triple of cell in 3d-field
    nz, ny, nx... 3d-shape of grid (number of cells)
    """

    return int(k * nx * ny + i * nx + j)

def retrieve_kij(nz, ny, nx, n):
    """
    Auxilliary function to find the triple of indices 
    k, i, j refering to the cell volume position in a 3d field
    from the index position n in the 1d-expanded field.

    nz, ny, nx... 3d-shape of grid (number of cells)
    n... index position in 1d-expanded field
    k, i, j... triple of indices of cell in 3d field
    """

    k = int(n / (ny * nx))
    i = int((n - k * ny * nx) / nx)
    j = n - k * ny * nx - i * nx
    return k, i, j


def put_1d(fields):
    """
    Expands a 3d face-centred vector field into 1d-shape by flatten each component
    and appending the fields in 'u', 'v', 'w' ordering.
    fields... tuple of vector components as 3d scalar fields

    field1d... returned 1d-expanded vector field
    """

    field1d = []
    n = 0
    for field in fields:
        field1d.extend(field.flatten())

    field1d = np.array(field1d)
    return field1d


def put_3d(field1d, shape):
    """
    Reshapes an expanded face-centred vector field in its
    3d-shape.
    
    field1d... 1d-expanded vector field
    shape... 3d-shape (number of cells in each dimension)
    
    field_u, field_v, field_w... returned vector components as 3d scalar fields
    """

    field_u = field1d[:shape[0] * shape[1] * (shape[2] + 1)].reshape(shape[0], shape[1], shape[2] + 1)
    field_v = field1d[shape[0] * shape[1] * (shape[2] + 1):shape[0] * shape[1] * (shape[2] + 1) + shape[0] * (shape[1] + 1) * shape[2]].reshape(shape[0], shape[1] + 1, shape[2])
    field_w = field1d[-(shape[0] + 1) * shape[1] * shape[2]:].reshape(shape[0] + 1, shape[1], shape[2])

    return field_u, field_v, field_w
