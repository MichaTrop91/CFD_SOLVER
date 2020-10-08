# Code design: 
# Michael Weger
# weger@tropos.de
# Permoserstrasse 15
# 04318 Leipzig                   
# Germany
# Last modified: 10.10.2020


def csr_gauss_seidel(Py_ssize_t dim_size, int[:] A_indptr, int[:] A_indices, double[:] A_data, double[:] diag_inv, double[:] bvec, double[:] xvec):
    """
    A fast lexicographic Gauss-Seidel (over-relaxation) solver 
    for A * x = b exploiting the compressed sparse row format of A.

    dim_size:   integer, dimension size of the square matrix A
    A_indptr:   integer array, pointers to rows of A
    A_indices:  integer array, comlumn indices of A
    A_data:     double array, non-zeros of A
    diag_inv:   double array, inverted diagonal of A scaled with an over-relaxation factor
    bvec:       doubla array, right-hand side vector
    xvec:       double array, first-guess and approximate-solution vector
    """

    cdef double ax
    cdef Py_ssize_t i
    cdef Py_ssize_t j
    for i in range(dim_size):
        ax = 0.0        
        for j in range(A_indptr[i], A_indptr[i + 1]):
            ax += A_data[j] * xvec[A_indices[j]]
        xvec[i] += diag_inv[i] * (bvec[i] - ax)


def csr_gauss_seidel_reverse(Py_ssize_t dim_size, int[:] A_indptr, int[:] A_indices, double[:] A_data, double[:] diag_inv, double[:] bvec, double[:] xvec):
    """
    A fast reverse-lexicographic Gauss-Seidel (over-relaxation) solver 
    for A * x = b exploiting the compressed sparse row format of A.
   

    dim_size:   integer, dimension size of the square matrix A
    A_indptr:   integer array, pointers to rows of A
    A_indices:  integer array, comlumn indices of A
    A_data:     double array, non-zeros of A
    diag_inv:   double array, inverted diagonal of A scaled with an over-relaxation factor
    bvec:       doubla array, right-hand side vector
    xvec:       double array, first-guess and approximate-solution vector
    """

    cdef double ax
    cdef Py_ssize_t i
    cdef Py_ssize_t j 
    for i in range(dim_size - 1, -1, -1):
        ax = 0.0
        for j in range(A_indptr[i], A_indptr[i + 1]):
            ax += A_data[j] * xvec[A_indices[j]]
        xvec[i] += diag_inv[i] * (bvec[i] - ax)
