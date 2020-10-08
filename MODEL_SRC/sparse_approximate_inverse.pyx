# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020


from scipy.sparse import csr_matrix
from scipy.sparse.linalg import norm
import scipy
import numpy as np

from copy import copy

def make_SPAI(A, A_P):
    """
    Constructs a sparse-approximate inverse
    of A with the sparsity pattern of A_P.
    See Huckle et al. (1995). 
    
    A... matrix whose inverse is approximated
    A_P... sparisity pattern of SPAI

    SPAI... returned sparse approximate inverse
    """


    cdef tuple shape = A.shape
    cdef int size = shape[0]
    cdef int[:] indptr_view = A.indptr
    cdef int[:] indices_view = A.indices
    cdef double[:] data_view = A.data

    cdef int[:] indptr_2_view = A_P.indptr
    cdef int[:] indices_2_view = A_P.indices

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k

    cdef Py_ssize_t mat_size = shape[0]
    cdef Py_ssize_t row_i_size

    cdef Py_ssize_t size_tmp
    cdef Py_ssize_t size_tmp2

    cdef unsigned int l
    cdef unsigned int ind
    cdef unsigned short c
    cdef unsigned int row
    cdef unsigned short count
    cdef unsigned int nrows


    cdef unsigned int ptr_spai


    colids_new = np.empty([indices_2_view.size], dtype=np.uint32)
    rowids_new = np.empty([indices_2_view.size], dtype=np.uint32)
    data_sp = np.empty([indices_2_view.size], dtype=np.float)

    cdef double[:] data_sp_view = data_sp
    cdef unsigned int[:] rowids_new_view = rowids_new
    cdef unsigned int[:] colids_new_view = colids_new

    cdef double[:] data_sub_view
    cdef unsigned int[:] rowids_sub_view
    cdef unsigned int[:] colids_sub_view
    
    cdef int[:] row_i_view
    cdef double[:, :] A_sub_view 
    cdef double[:] x_view
    cdef double[:] ei_view
    cdef unsigned int[:] row_comp_view


    ptr_spai = 0

    for i in range(mat_size):
        
        row_i = indices_2_view[indptr_2_view[i]:indptr_2_view[i + 1]]
        row_i_view = row_i
        row_i_size = row_i.size
        size_tmp = 0
        for j in range(row_i_size):
            ind = row_i_view[j]
            size_tmp += indptr_view[ind + 1] - indptr_view[ind]

        if size_tmp == 0:
            continue
        data_sub = np.empty([size_tmp], dtype=float)
        rowids_sub = np.empty([size_tmp], dtype=np.uint32)
        colids_sub = np.empty([size_tmp], dtype=np.uint32)
 
        data_sub_view = data_sub
        rowids_sub_view = rowids_sub
        colids_sub_view = colids_sub
  

        if row_i_size == 0:
            continue
        c = 0
        count = 0
        for j in range(row_i_size):
            ind = row_i_view[j]
            l = indptr_view[ind]
            size_tmp2 = indptr_view[ind + 1] - indptr_view[ind]
            for k in range(size_tmp2):
                data_sub_view[count] = data_view[l]
                rowids_sub_view[count] = indices_view[l]
                colids_sub_view[count] = c
                l += 1
                count += 1
            c+=1

        row_comp = np.unique(rowids_sub_view)
        row_comp_view = row_comp
        nrows = row_comp.size

        A_sub = np.full([nrows, row_i_size], 1e-40)
        A_sub_view = A_sub
        for j in range(size_tmp):
            ind = rowids_sub[j]
            for k in range(nrows):
                if ind == row_comp_view[k]:
                    A_sub_view[k, colids_sub_view[j]] = data_sub_view[j]
                    break

        ei = np.zeros([nrows])
        ei_view = ei
        for k in range(nrows):
           if row_comp_view[k] == i:
               ei_view[k] = 1.0

        x = scipy.linalg.lstsq(A_sub_view, ei_view, overwrite_a = True, overwrite_b=True, lapack_driver='gelsy')[0]
        x_view = x
        l = ptr_spai
        for k in range(row_i_size):
            rowids_new_view[l] = i
            colids_new_view[l] = row_i[k]
            data_sp_view[l] = x_view[k]
            l += 1
            ptr_spai += 1

    SPAI = csr_matrix((data_sp_view[:ptr_spai], [rowids_new_view[:ptr_spai], colids_new_view[:ptr_spai]]), shape)

    return SPAI


def construct_sparsity_stencil(MAT, int nz, int ny, int nx, int l=0):
    """
    This function selects the sparsity
    pattern based on the stencil of MAT
    and additional l surrounding neighbor nodes.
   
    MAT... matrix to approximately invert
    nz, ny, nx... 3d shape of grid (number of cells)
    l... number of neighbor nodes included in the sparsity pattern

    PAT... returned matrix representing the sparsity pattern
    """

    cdef int mat_size = MAT.shape[0]
    cdef int[:] indices_MAT_view = MAT.indices
    cdef int[:] indptr_MAT_view = MAT.indptr

    cdef int nnz = MAT.data.size

    stencil_rowids = np.empty([nnz * (2 * l + 1) ** 3], dtype=np.int32)
    stencil_colids = np.empty([nnz * (2 * l + 1) ** 3], dtype=np.int32)
    stencil_data = np.ones([nnz * (2 * l + 1) ** 3], dtype=np.float64)
     
    cdef int[:] stencil_rowids_view = stencil_rowids
    cdef int[:] stencil_colids_view = stencil_colids
    cdef double[:] stencil_data_view = stencil_data

    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t n
    cdef Py_ssize_t m
    cdef Py_ssize_t kk
    cdef Py_ssize_t ii
    cdef Py_ssize_t jj

    cdef int ptr_stenc_mat
    cdef int row_size
    cdef int col_ind
    cdef int col_ind_new

    ptr_stenc_mat = 0
    for n in range(mat_size):
        col_ids_sub = indices_MAT_view[indptr_MAT_view[n]:indptr_MAT_view[n + 1]]
        row_size = col_ids_sub.size

        for m in range(row_size):
            col_ind = col_ids_sub[m]             
            col_ind_new = min(max(k * nx * ny + i * nx + j, 0), mat_size - 1)
            stencil_rowids_view[ptr_stenc_mat] = n
            stencil_colids_view[ptr_stenc_mat] = col_ind_new
            ptr_stenc_mat += 1

            k = col_ind / (nx * ny)
            i = (col_ind - k * (nx * ny) ) / nx
            j = col_ind - k * nx * ny - i * nx

            for kk in range(-l, l):
                for ii in range(-l, l):
                    for jj in range(-l, l):
                        col_ind_new = min(max((k + kk) * nx * ny + (i + ii) * nx + j + jj, 0), mat_size - 1)
                        stencil_rowids_view[ptr_stenc_mat] = n
                        stencil_colids_view[ptr_stenc_mat] = col_ind_new
                        ptr_stenc_mat += 1
                        
    PAT = csr_matrix((stencil_data_view[:ptr_stenc_mat], [stencil_rowids_view[:ptr_stenc_mat], stencil_colids_view[:ptr_stenc_mat]]), (mat_size, mat_size))

    return PAT


def make_SPAIE(SMO, MAT, int n_update):
    """
    Improves a sparse approximate inverse
    using the SPAI algorithm (Huckle et al.,1995).
   
    SPAI... initial sparse approximate inverse
    A... matrix whose inverse is approximated
    n_update... number of update steps 
    n_add... number of indices per row to add at each update

    SPAI_new... returned new sparse approximate inverse
    """

    cdef tuple shape = MAT.shape
    cdef double[:] data_SMO_view = SMO.data
    cdef int[:] indices_SMO_view = SMO.indices
    cdef int[:] indptr_SMO_view = SMO.indptr

    cdef double[:] data_MAT_view = MAT.data
    cdef int[:] indices_MAT_view = MAT.indices
    cdef int[:] indptr_MAT_view = MAT.indptr

    cdef double[:] A_k_view = norm(MAT, axis=1)
    
    MAT_T = MAT.transpode(copy=True)

    cdef double[:] data_MAT_T_view = MAT_T.data
    cdef int[:] indices_MAT_T_view = MAT_T.indices
    cdef int[:] indptr_MAT_T_view = MAT_T.indptr

    cdef int[:] indices_SMO_new_view

    cdef int size_inds_SMO_old = SMO.data.size
    cdef int size_inds_SMO_new 
    
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t k
    cdef Py_ssize_t n
 

    size_inds_SMO_new = size_inds_SMO_old + n_update * shape[0]

    indices_SMO_new = np.empty([size_inds_SMO_new], dtype=np.uint32)
    indices_SMO_new_view = indices_SMO_new

    for i in range(shape[0]):
        row_i_SMO = indices_SMO_view[indptr_SMO_view[i]:indptr_SMO_view[i + 1]]
        data_i_SMO = data_SMO_view[indptr_SMO_view[i]:indptr_SMO_view[i + 1]]

#        for n in range(n_update):


def sparsificate(SPAI, unsigned int nfill, double drop_tol=1e-4):
    '''
    This function removes small nz-values from a sparse 
    matrix, such that only nfill values are contained in 
    each row.

    SPAI... the sparse matrix to sparsificate
    nfill... number of remaining nz-values in each row
    drop_tol... drop tolerance to remove additional values below drop_tol in absolute value
   
    SPAI_new... returned sparsificated matrix 
    '''
    
    cdef double[:] data_view = SPAI.data
    cdef int[:] indices_view = SPAI.indices
    cdef int[:] indptr_view = SPAI.indptr

    cdef int[:] indices_new_view
    cdef int[:] indptr_new_view
    cdef double[:] data_new_view

    cdef double[:] new_datarow_view
    cdef int[:] new_indsrow_view

    cdef int[:] inds_sorted_full_view
    cdef int[:] inds_sorted_view

    cdef unsigned int row_size
    cdef unsigned int size_inds_new   
    cdef short rowpos_diag = -1
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    cdef Py_ssize_t nrows

    cdef unsigned int l

    cdef unsigned short diag_contained
    cdef unsigned int ptr_spai

    nrows = SPAI.shape[0]
    
    data_new = []
    row_inds = []
    col_inds = []

    size_inds_new = 0

    for i in range(nrows):
        size_inds_new += min(nfill, indptr_view[i + 1] - indptr_view[i])

    data_new = np.empty([size_inds_new], dtype=float)
    colinds_new = np.empty([size_inds_new], dtype=np.int32)
    rowinds_new = np.empty([size_inds_new], dtype=np.int32)
  
    data_new_view = data_new
    colinds_new_view = colinds_new
    rowinds_new_view = rowinds_new

    ptr_spai = 0

    for i in range(nrows):
        data_r = data_view[indptr_view[i]:indptr_view[i + 1]]
        indices_r = indices_view[indptr_view[i]:indptr_view[i + 1]]
        row_size = indptr_view[i + 1] - indptr_view[i]        
 
        l = ptr_spai
        if row_size <= nfill:
            for j in range(row_size):
                if np.absolute(data_r[j]) >= drop_tol:
                    data_new_view[l] = data_r[j]
                    colinds_new_view[l] = indices_r[j]
                    rowinds_new_view[l] = i
                    l += 1
                    ptr_spai += 1
            continue
                  
        data_abs = np.absolute(data_r)

        
        inds_sorted_full = (np.argsort(data_abs)).astype(dtype=np.int32)
        inds_sorted_full_view = inds_sorted_full
 
        inds_sorted = np.empty([nfill], dtype=np.int32)
        inds_sorted_view = inds_sorted

        for j in range(nfill):
            inds_sorted_view[j] = inds_sorted_full_view[-j - 1]

        # keep always the diagonal element if available
        for j in range(row_size):
            if indices_r[j] == i:
                rowpos_diag = j

        diag_contained = 0
        for j in range(nfill):
            a = inds_sorted_view[j]
            if indices_r[inds_sorted_view[j]] == i:
                diag_contained = 1
        if diag_contained == 0 and rowpos_diag > -1:
            inds_sorted_view[0] = rowpos_diag           
            
        l = ptr_spai  
        for j in range(nfill):
            if np.absolute(data_r[inds_sorted_view[j]]) >= drop_tol:
                data_new_view[l] = data_r[inds_sorted_view[j]]
                colinds_new_view[l] = indices_r[inds_sorted_view[j]]
                rowinds_new_view[l] = i
                l += 1
                ptr_spai += 1
    
    SPAI_new = csr_matrix((data_new_view[:ptr_spai], (rowinds_new_view[:ptr_spai], colinds_new_view[:ptr_spai])), SPAI.shape)
    return SPAI_new

