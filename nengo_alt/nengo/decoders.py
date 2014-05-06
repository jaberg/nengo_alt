"""
Functions concerned with solving for decoders or full weight matrices.

Many of the solvers in this file can solve for decoders or weight matrices,
depending on whether the post-population encoders `E` are provided (see below).
Solvers that are only intended to solve for either decoders or weights can
remove the `E` parameter or make it manditory as they see fit.

All solvers take following arguments:
  A : array_like (M, N)
    Matrix of the N neurons' activities at the M evaluation points
  Y : array_like (M, D)
    Matrix of the target decoded values for each of the D dimensions,
    at each of the M evaluation points.

All solvers have the following optional keyword parameters:
  rng : numpy.RandomState
    A random number generator to use as required. If none is provided,
    numpy.random will be used.
  E : array_like (D, N2)
    Array of post-population encoders. Providing this tells the solver
    to return an array of connection weights rather than decoders.

Many solvers have the following optional keyword parameters:
  noise_amp : float
    This generally governs the amount of L2 regularization, as a fraction
    of the maximum firing rate of the population.

All solvers return the following:
  X : np.ndarray (N, D) or (N, N2)
    (N, D) array of decoders if E is none, or (N, N2) array of weights
    if E is not none.
"""

import logging
import numpy as np
import nengo.utils.numpy as npext

logger = logging.getLogger(__name__)


def _get_solver():
    # NOTE: this should be settable by a defaults manager
    return _cholesky


def lstsq(A, Y, rng=np.random, E=None, rcond=0.01):
    """Unregularized least-squares."""
    Y = np.dot(Y, E) if E is not None else Y
    X, residuals2, rank, s = np.linalg.lstsq(A, Y, rcond=rcond)
    return X, {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
               'residuals': np.sqrt(residuals2),
               'rank': rank,
               'singular_values': s}


def lstsq_noise(
        A, Y, rng=np.random, E=None, noise_amp=0.1, solver=None, **kwargs):
    """Least-squares with additive white noise."""
    sigma = noise_amp * A.max()
    A = A + rng.normal(scale=sigma, size=A.shape)
    Y = np.dot(Y, E) if E is not None else Y
    solver = _get_solver() if solver is None else solver
    return solver(A, Y, 0, **kwargs)


def lstsq_multnoise(
        A, Y, rng=np.random, E=None, noise_amp=0.1, solver=None, **kwargs):
    """Least-squares with multiplicative white noise."""
    A = A + rng.normal(scale=noise_amp, size=A.shape) * A
    Y = np.dot(Y, E) if E is not None else Y
    solver = _get_solver() if solver is None else solver
    return solver(A, Y, 0, **kwargs)


def lstsq_L2(
        A, Y, rng=np.random, E=None, noise_amp=0.1, solver=None, **kwargs):
    """Least-squares with L2 regularization."""
    Y = np.dot(Y, E) if E is not None else Y
    sigma = noise_amp * A.max()
    solver = _get_solver() if solver is None else solver
    return solver(A, Y, sigma, **kwargs)


def lstsq_L2nz(
        A, Y, rng=np.random, E=None, noise_amp=0.1, solver=None, **kwargs):
    """Least-squares with L2 regularization on non-zero components."""
    Y = np.dot(Y, E) if E is not None else Y

    # Compute the equivalent noise standard deviation. This equals the
    # base amplitude (noise_amp times the overall max activation) times
    # the square-root of the fraction of non-zero components.
    sigma = (noise_amp * A.max()) * np.sqrt((A > 0).mean(axis=0))

    # sigma == 0 means the neuron is never active, so won't be used, but
    # we have to make sigma != 0 for numeric reasons.
    sigma[sigma == 0] = sigma.max()

    solver = _get_solver() if solver is None else solver
    return solver(A, Y, sigma, **kwargs)


def lstsq_L1(A, Y, rng=np.random, E=None, l1=1e-4, l2=1e-6):
    """Least-squares with L1 and L2 regularization (elastic net).

    This method is well suited for creating sparse decoders or weight matrices.
    """
    import sklearn.linear_model

    # TODO: play around with these regularization constants (I just guessed).
    #   Do we need to scale regularization by number of neurons, to get same
    #   level of sparsity? esp. with weights? Currently, setting l1=1e-3 works
    #   well with weights when connecting 1D populations with 100 neurons each.
    a = l1 * A.max()      # L1 regularization
    b = l2 * A.max()**2   # L2 regularization
    alpha = a + b
    l1_ratio = a / (a + b)

    # --- solve least-squares A * X = Y
    if E is not None:
        Y = np.dot(Y, E)

    model = sklearn.linear_model.ElasticNet(
        alpha=alpha, l1_ratio=l1_ratio, fit_intercept=False, max_iter=1000)
    model.fit(A, Y)
    X = model.coef_.T
    X.shape = (A.shape[1], Y.shape[1]) if Y.ndim > 1 else (A.shape[1],)
    infos = {'rmses': npext.rms(Y - np.dot(A, X), axis=0)}
    return X, infos


def lstsq_drop(A, Y, rng, E=None, noise_amp=0.1, drop=0.25, solver=lstsq_L2nz):
    """Find sparser decoders/weights by dropping small values.

    This solver first solves for coefficients (decoders/weights) with
    L2 regularization, drops those nearest to zero, and retrains remaining.
    """
    Y, m, n, d, matrix_in = _format_system(A, Y)

    # solve for coefficients using standard solver
    X, info0 = solver(A, Y, rng=rng, noise_amp=noise_amp)
    X = np.dot(X, E) if E is not None else X

    # drop weights close to zero, based on `drop` ratio
    Xabs = np.sort(np.abs(X.flat))
    threshold = Xabs[int(np.round(drop * Xabs.size))]
    X[np.abs(X) < threshold] = 0

    # retrain nonzero weights
    Y = np.dot(Y, E) if E is not None else Y
    for i in range(X.shape[1]):
        nonzero = X[:, i] != 0
        if nonzero.sum() > 0:
            X[nonzero, i], info1 = solver(
                A[:, nonzero], Y[:, i], rng=rng, noise_amp=0.1 * noise_amp)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'info0': info0, 'info1': info1}
    return X if matrix_in else X.flatten(), info


def nnls(A, Y, rng, E=None):
    """Non-negative least-squares without regularization.

    Similar to `lstsq`, except the output values are non-negative.
    """
    import scipy.optimize

    Y, m, n, d, matrix_in = _format_system(A, Y)
    Y = np.dot(Y, E) if E is not None else Y
    X = np.zeros((n, d))
    residuals = np.zeros(d)
    for i in range(d):
        X[:, i], residuals[i] = scipy.optimize.nnls(A, Y[:, i])

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'residuals': residuals}
    return X if matrix_in else X.flatten(), info


def nnls_L2(A, Y, rng, E=None, noise_amp=0.1):
    """Non-negative least-squares with L2 regularization.

    Similar to `lstsq_L2`, except the output values are non-negative.
    """
    # form Gram matrix so we can add regularization
    sigma = noise_amp * A.max()
    G = np.dot(A.T, A)
    Y = np.dot(A.T, Y)
    np.fill_diagonal(G, G.diagonal() + sigma)
    return nnls(G, Y, rng, E=E)


def nnls_L2nz(A, Y, rng, E=None, noise_amp=0.1):
    """Non-negative least-squares with L2 regularization on nonzero components.

    Similar to `lstsq_L2nz`, except the output values are non-negative.
    """
    sigma = (noise_amp * A.max()) * np.sqrt((A > 0).mean(axis=0))
    sigma[sigma == 0] = 1

    # form Gram matrix so we can add regularization
    G = np.dot(A.T, A)
    Y = np.dot(A.T, Y)
    np.fill_diagonal(G, G.diagonal() + sigma)
    return nnls(G, Y, rng, E=E)


def _cholesky(A, y, sigma, transpose=None):
    """Solve the least-squares system using the Cholesky decomposition."""
    m, n = A.shape
    if transpose is None:
        # transpose if matrix is fat, but not if we have sigmas for each neuron
        transpose = m < n and sigma.size == 1

    if transpose:
        # substitution: x = A'*xbar, G*xbar = b where G = A*A' + lambda*I
        G = np.dot(A, A.T)
        b = y
    else:
        # multiplication by A': G*x = A'*b where G = A'*A + lambda*I
        G = np.dot(A.T, A)
        b = np.dot(A.T, y)

    # add L2 regularization term 'lambda' = m * sigma**2
    np.fill_diagonal(G, G.diagonal() + m * sigma**2)

    try:
        import scipy.linalg
        factor = scipy.linalg.cho_factor(G, overwrite_a=True)
        x = scipy.linalg.cho_solve(factor, b)
    except ImportError:
        L = np.linalg.cholesky(G)
        L = np.linalg.inv(L.T)
        x = np.dot(L, np.dot(L.T, b))

    x = np.dot(A.T, x) if transpose else x
    info = {'rmses': npext.rms(y - np.dot(A, x), axis=0)}
    return x, info


def _conjgrad_scipy(A, Y, sigma, tol=1e-4):
    import scipy.sparse.linalg
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = m * sigma**2
    calcAA = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    G = scipy.sparse.linalg.LinearOperator(
        (n, n), matvec=calcAA, matmat=calcAA, dtype=A.dtype)
    B = np.dot(A.T, Y)

    X = np.zeros((n, d), dtype=B.dtype)
    infos = np.zeros(d, dtype='int')
    itns = np.zeros(d, dtype='int')
    for i in range(d):
        def callback(x):
            itns[i] += 1  # use the callback to count the number of iterations

        X[:, i], infos[i] = scipy.sparse.linalg.cg(G, B[:, i], tol=tol,
                                                   callback=callback)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': itns,
            'info': infos}
    return X if matrix_in else X.flatten(), info


def _lsmr_scipy(A, Y, sigma, tol=1e-4):
    import scipy.sparse.linalg
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = sigma * np.sqrt(m)
    X = np.zeros((n, d), dtype=Y.dtype)
    itns = np.zeros(d, dtype='int')
    for i in range(d):
        X[:, i], _, itns[i], _, _, _, _, _ = scipy.sparse.linalg.lsmr(
            A, Y[:, i], damp=damp, atol=tol, btol=tol)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': itns}
    return X if matrix_in else X.flatten(), info


def _conjgrad_iters(calcAx, b, x, maxiters=None, rtol=1e-6):
    """Solve the single-RHS linear system using conjugate gradient."""

    if maxiters is None:
        maxiters = b.shape[0]

    r = b - calcAx(x)
    p = r.copy()
    rsold = np.dot(r, r)

    for i in range(maxiters):
        Ap = calcAx(p)
        alpha = rsold / np.dot(p, Ap)
        x += alpha * p
        r -= alpha * Ap

        rsnew = np.dot(r, r)
        beta = rsnew / rsold

        if np.sqrt(rsnew) < rtol:
            break

        if beta < 1e-12:  # no perceptible change in p
            break

        # p = r + beta*p
        p *= beta
        p += r
        rsold = rsnew

    return x, i+1


def _conjgrad(A, Y, sigma, X0=None, maxiters=None, tol=1e-2):
    """Solve the least-squares system using conjugate gradient."""
    Y, m, n, d, matrix_in = _format_system(A, Y)

    damp = m * sigma**2
    rtol = tol * np.sqrt(m)
    G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    B = np.dot(A.T, Y)

    X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
    iters = -np.ones(d, dtype='int')
    for i in range(d):
        X[:, i], iters[i] = _conjgrad_iters(
            G, B[:, i], X[:, i], maxiters=maxiters, rtol=rtol)

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': iters}
    return X if matrix_in else X.flatten(), info


def _block_conjgrad(A, Y, sigma, X0=None, tol=1e-2):
    """Solve a least-squares system with multiple-RHS."""
    Y, m, n, d, matrix_in = _format_system(A, Y)
    sigma = np.asarray(sigma, dtype='float')
    sigma = sigma.reshape(sigma.size, 1)

    damp = m * sigma**2
    rtol = tol * np.sqrt(m)
    G = lambda x: np.dot(A.T, np.dot(A, x)) + damp * x
    B = np.dot(A.T, Y)

    # --- conjugate gradient
    X = np.zeros((n, d)) if X0 is None else np.array(X0).reshape((n, d))
    R = B - G(X)
    P = np.array(R)
    Rsold = np.dot(R.T, R)
    AP = np.zeros((n, d))

    maxiters = int(n / d)
    for i in range(maxiters):
        AP = G(P)
        alpha = np.linalg.solve(np.dot(P.T, AP), Rsold)
        X += np.dot(P, alpha)
        R -= np.dot(AP, alpha)

        Rsnew = np.dot(R.T, R)
        if (np.diag(Rsnew) < rtol**2).all():
            break

        beta = np.linalg.solve(Rsold, Rsnew)
        P = R + np.dot(P, beta)
        Rsold = Rsnew

    info = {'rmses': npext.rms(Y - np.dot(A, X), axis=0),
            'iterations': i + 1}
    return X if matrix_in else X.flatten(), info


def _format_system(A, Y):
    m, n = A.shape
    matrix_in = Y.ndim > 1
    d = Y.shape[1] if matrix_in else 1
    Y = Y.reshape((Y.shape[0], d))
    return Y, m, n, d, matrix_in
