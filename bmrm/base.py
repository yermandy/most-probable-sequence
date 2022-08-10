import numpy as np
from qpsolvers import solve_qp
from timeit import default_timer as timer
import sys
from bmrm.libqp_bridge import qp_splx


def bmrm(
        risk_func,
        n_dim,
        lmbda,
        solver="franc",
        tol_rel=1e-3,
        tol_abs=0,
        max_iter=np.inf,
        buff_size=500,
        cp_cln=np.inf,
        verb=True,
        store_W=False
):

    t0 = timer()
    stats = {}

    if buff_size > max_iter:
        buff_size = max_iter + 1

    W = np.zeros(n_dim)

    # compute risk and its gradient
    tmp_time = timer()
    R, subgrad = risk_func(W)
    risktime1 = timer() - tmp_time

    # if gradient is zero, return current parameters
    if np.all(subgrad == 0):
        stats['Fp'] = stats['Fd'] = stats['hist_Fp'] = stats['hist_Fd'] = stats['hist_R'] = R
        stats['nIter'] = 0
        stats['hist_qtime'] = 0
        stats['hist_risktime'] = risktime1
        stats['hist_hessiantime'] = 0
        stats['hist_innerlooptime'] = 0
        stats['hist_wtime'] = 0
        stats['hist_runtime'] = timer() - t0
        stats['flag'] = 2
        if store_W:
            stats['hist_W'] = W

        return W, stats

    # Allocate matrices A,b,H according to buff_size
    A = np.zeros((n_dim, buff_size + 1), dtype=np.float64)
    b = np.zeros((buff_size + 1, 1), dtype=np.float64)
    H = np.zeros((buff_size + 1, buff_size + 1), dtype=np.float64)

    A[:, 0] = subgrad.flatten()
    b[0] = R
    alpha = np.array([])

    clean_cp = not np.isinf(cp_cln) and solver == 'franc'
    if clean_cp:
        alpha_cd = np.zeros(buff_size + 1)

    itr = 0
    nCP = 0
    exitflag = -1
    Fp = Fd = -np.inf

    # allocate buffers for statistics
    hist_Fd = np.zeros(buff_size + 1)
    hist_Fp = np.zeros(buff_size + 1)
    hist_R = np.zeros(buff_size + 1)
    hist_runtime = np.zeros(buff_size + 1)
    hist_risktime = np.zeros(buff_size + 1)
    hist_qptime = np.zeros(buff_size + 1)
    hist_hessiantime = np.zeros(buff_size + 1)
    hist_innerlooptime = np.zeros(buff_size + 1)
    hist_wtime = np.zeros(buff_size + 1)
    if store_W:
        histW = np.zeros((buff_size, n_dim))

    hist_risktime[0] = risktime1
    hist_runtime[0] = timer() - t0
    hist_Fd[0] = -np.inf
    hist_Fp[0] = R + 0.5 * lmbda * np.linalg.norm(W) ** 2
    hist_R[0] = R

    if verb:
        print(f'Buffers allocated for {buff_size} cutting planes:')
        print(f'Cutting plane buffer: {sys.getsizeof(A)/1024**2} MB')
        print(f'Hessian: {sys.getsizeof(H) / 1024 ** 2} MB')
        print(f'{itr}: '
              f'time={round(hist_runtime[0], 4)}, '
              f'Fp={round(hist_Fp[0], 4)}, '
              f'Fd={round(hist_Fd[0], 4)}, '
              f'R={round(hist_R[0], 4)}')

    # main optimization loop
    while exitflag == -1:
        itr_start_time = timer()
        itr += 1
        nCP += 1
        tmp_time = timer()

        if nCP > 1:
            H[:nCP-1, nCP-1] = A[:, :nCP-1].T @ A[:, nCP-1] / lmbda
            H[nCP-1, :nCP-1] = H[:nCP-1, nCP-1].T
        H[nCP-1, nCP-1] = A[:, nCP-1].T @ A[:, nCP-1] / lmbda
        hist_hessiantime[itr] = timer() - tmp_time

        tmp_time = timer()
        if np.all(b[:nCP] == 0):
            alpha = np.zeros(nCP)
        elif solver == 'franc':
            x0 = np.pad(alpha, (0, 1))
            if nCP == 1:
                x0[0] = 1
            alpha = qp_splx(np.ascontiguousarray(H[:nCP, :nCP]), np.ascontiguousarray(-b[:nCP].reshape(nCP,)), x0)
        else:
            alpha = solve_qp(H[:nCP, :nCP],
                             -b[:nCP].reshape(nCP,),
                             A=np.ones(nCP),
                             b=np.array([1.]),
                             lb=np.zeros(nCP),
                             solver=solver)
        hist_qptime[itr] = timer() - tmp_time

        tmp_time = timer()
        W = - alpha @ A[:, :nCP].T / lmbda
        hist_wtime[itr] = timer() - tmp_time

        nzA = sum(alpha > 0)

        tmp_time = timer()
        R, subgrad = risk_func(W)
        hist_risktime[itr] = timer() - tmp_time
        A[:, nCP] = subgrad.flatten()
        b[nCP] = R - A[:, nCP].T @ W.T

        Fp = R + 0.5 * lmbda * np.linalg.norm(W) ** 2
        Fd = - 0.5 * alpha @ H[:nCP, :nCP] @ alpha[:, np.newaxis] + alpha @ b[:nCP]

        # check for terminal conditions
        if Fp - Fd <= tol_rel*np.abs(Fp):
            exitflag = 1
        elif Fp-Fd <= tol_abs:
            exitflag = 2
        elif itr >= max_iter:
            exitflag = 0

        hist_runtime[itr] = timer() - t0
        hist_Fp[itr] = Fp
        hist_Fd[itr] = Fd
        hist_R[itr] = R
        if store_W:
            histW[itr-1, :] = W

        # purge unused cutting planes
        if clean_cp:
            alpha_cd[np.nonzero(alpha == 0)] += 1
            alpha_cd[np.nonzero(alpha)] = 0

            if np.any(alpha_cd >= cp_cln):
                old_nCP = nCP
                idx = np.nonzero(alpha_cd[:old_nCP] < cp_cln)[0]
                nCP = len(idx)

                alpha = alpha[idx]
                H[:nCP, :nCP] = H[idx, :][:, idx]
                A[:, :nCP] = A[:, idx]
                b[:nCP] = b[idx]

                #alpha_cd[:nCP] = alpha_cd[idx]
                #alpha_cd[nCP:] = 0
                alpha_cd[:] = 0

                if verb:
                    print(f'Cutting plane buffer cleaned up (old_nCP={old_nCP}, new_nCP={nCP})')

        # Expand buffers if needed
        if nCP >= A.shape[1] - 1:
            A = np.pad(A, ((0, 0), (0, 1)))
            H = np.pad(H, ((0, 1), (0, 1)))
            b = np.pad(b, ((0, 1), (0, 0)))
            if clean_cp:
                alpha_cd = np.pad(alpha_cd, (0, 1))
            if verb:
                print(f'Cutting plane buffer size increased to {A.shape[1]}')

        if itr >= buff_size:
            buff_size += 1
            hist_Fd = np.pad(hist_Fd, (0, 1))
            hist_Fp = np.pad(hist_Fp, (0, 1))
            hist_R = np.pad(hist_R, (0, 1))
            hist_runtime = np.pad(hist_runtime, (0, 1))
            hist_risktime = np.pad(hist_risktime, (0, 1))
            hist_qptime = np.pad(hist_qptime, (0, 1))
            hist_hessiantime = np.pad(hist_hessiantime, (0, 1))
            hist_innerlooptime = np.pad(hist_innerlooptime, (0, 1))
            hist_wtime = np.pad(hist_wtime, (0, 1))
            if store_W:
                histW = np.pad(histW, ((0, 1), (0, 0)))
            if verb:
                print(f'Stats buffer size increased to {buff_size + 1}')

        hist_innerlooptime[itr] = timer() - itr_start_time

        if verb:
            print(f'{itr}: '
                  f'time={round(hist_runtime[itr], 4)}, '
                  f'Fp={round(hist_Fp[itr], 4)}, '
                  f'Fd={round(hist_Fd[itr], 4)}, '
                  f'(Fp-Fd)={round((Fp-Fd).item(), 4)}, '
                  f'(Fp-Fd)/Fp={round(((Fp-Fd)/Fp).item(), 4)}, '
                  f'R={round(hist_R[itr], 4)}, '
                  f'nCP={round(nCP, 4)}, '
                  f'nzA={round(nzA, 4)}, '
                  f'time_inner={round(hist_innerlooptime[itr], 4)}, '
                  f'time_risk={round(hist_risktime[itr], 4)}, '
                  f'time_w={round(hist_wtime[itr], 4)}, '
                  f'time_qp={round(hist_qptime[itr], 4)}, '
                  f'time_hes={round(hist_hessiantime[itr], 4)}, ')

    if verb:
        print('Accumulated times:')
        print(f'risk time       : {hist_risktime[:itr+1].sum()}')
        print(f'qp time         : {hist_qptime[:itr+1].sum()}')
        print(f'hessian time    : {hist_hessiantime[:itr+1].sum()}')
        print(f'w time          : {hist_wtime[:itr+1].sum()}')
        print(f'inner loop time : {hist_innerlooptime[:itr+1].sum()}')
        print(f'total runtime   : {hist_runtime[itr]}')

    # set up statistics
    stats['Fp'] = Fp
    stats['Fd'] = Fd
    stats['hist_Fp'] = hist_Fp[:itr+1]
    stats['hist_Fd'] = hist_Fd[:itr+1]
    stats['hist_R'] = hist_R[:itr+1]
    stats['nIter'] = itr
    stats['hist_qtime'] = hist_qptime[:itr+1]
    stats['hist_risktime'] = hist_risktime[:itr+1]
    stats['hist_hessiantime'] = hist_hessiantime[:itr+1]
    stats['hist_innerlooptime'] = hist_innerlooptime[:itr+1]
    stats['hist_wtime'] = hist_wtime[:itr+1]
    stats['hist_runtime'] = hist_runtime[:itr+1]
    stats['flag'] = exitflag
    if store_W:
        stats['hist_W'] = histW[:itr, :]

    return W, stats
