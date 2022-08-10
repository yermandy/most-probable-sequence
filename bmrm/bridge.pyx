#!python3
#cython: language_level=3

import numpy as np

cdef extern from "libqp.h":
    ctypedef struct libqp_state_T:
        unsigned int nIter;
        double QP;
        double QD;
        char exitflag;

    libqp_state_T libqp_splx_solver(
            const double* (*get_col)(unsigned int),
            const double *diag_H,
            double *f,
            double *b,
            unsigned int *I,
            unsigned char *S,
            double *x,
            unsigned int n,
            unsigned int MaxIter,
            double TolAbs,
            double TolRel,
            double QP_TH,
            void (*print_state)(libqp_state_T state)
    );

cdef double[:, ::1] view


cdef const double *get_col(unsigned int col):
    return &view[col, 0]


def qp_splx(
        H,
        f,
        x0=None,
        unsigned int maxiter=np.iinfo(np.int32).max,
        double tolabs=0 ,
        double tolrel=1e-9,
        double qp_th=-np.inf,
        verb=False
):
    b = np.array([1], dtype=np.float64)
    I = np.ones(H.shape[0], dtype=np.uint32)
    S = np.array([0], dtype=np.uint8)
    if x0 is None:
        x = np.zeros(H.shape[0], dtype=np.float64)
        x[0] = 1
    else:
        x = x0.copy()

    global view
    view = H

    cdef unsigned int nvar = H.shape[0]
    cdef const double[::1] diag_H = np.ascontiguousarray(np.diagonal(H))
    cdef double[::1] vec_f = f
    cdef double[::1] vec_b = b
    cdef unsigned int[::1] vec_I = I
    cdef unsigned char[::1] vec_S = S
    cdef double[::1] vec_x = x

    if verb:
        print("Settings of LIBQP_SPLX solver:")
        print(f"MaxIter  : {maxiter}" )
        print(f"TolAbs   : {tolabs}")
        print(f"TolRel   : {tolrel}")
        print(f"QP_TH    : {qp_th}")
        print(f"nVar     : {nvar}")

    cdef libqp_state_T state = libqp_splx_solver(
        &get_col,
        &diag_H[0],
        &vec_f[0],
        &vec_b[0],
        &vec_I[0],
        &vec_S[0],
        &vec_x[0],
        nvar,
        maxiter,
        tolabs,
        tolrel,
        qp_th,
        NULL)

    return x


