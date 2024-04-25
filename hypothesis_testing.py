# -*- coding: utf-8 -*-
"""
A set of additional functions to assist with the auto-correlation and
hypothesis testing method described in [1]. Functions from `np_redef` are
re-exported here to try and avoid any confusion with `numpy` in scripts which
use both this module and `np_redef`.

References
----------
[1] - new paper citation.
[2] - R. DE MAESSCHALCK, et al., The Mahalanobis distance, Chemometrics and
        Intelligent Laboratory Systems, Volume 50, Issue 1, p. 1-18, Jan 2000,
        doi:10.1016/S0169-7439(99)00047-7
[3] - K. V. MARDIA, Measures of multivariate skewness and kurtosis with
        applications, Biometrika, Volume 57, Issue 3, p. 519–530, Dec 1970,
        doi:10.1093/biomet/57.3.519
[4] - L. J. NELSON, et al., Ply-orientation measurements in composites using
        structure-tensor analysis of volumetric ultrasonic data, Composites A,
        Volume 104, p. 108-119, Jan 2018, doi:10.1016/j.compositesa.2017.10.027
        
"""
__all__ = [
    'auc', 'autocorrelate', 'convolve', 'correlate', 'cov', 'mahalanobis', 
    'mardia_test', 'nancov', 'opheim_simpl', 'ply_orientation', 'roc'
]
__version__ = '0.1'
__author__ = 'Matthew G. Chandler'

import numpy as np
from scipy.stats import chi2, norm
import warnings

from np_redef import convolve, correlate, cov


def auc(x, y):
    """
    Compute the area under the curve of a given set of `x` and `y` values. This
    is designed to work with the result of the `roc` function, thus it is
    expected that the function is monotonic (increasing or decreasing).

    Parameters
    ----------
    x : ndarray (N,)
        Input x-coordinate values.
    y : ndarray (N,)
        Input y-coordinate values.

    Returns
    -------
    area : float
        Area under the curve described by `x` and `y`.

    """
    dx = np.diff(x)
    direction = 1
    if np.any(dx < 0):
        if np.all(dx <= 0):
            direction = -1
        else:
            raise ValueError('x is neither increasing nor decreasing : {}.'.format(x))
    return direction * np.trapz(y, x)


def autocorrelate(x, mode="same", method="auto", axes=None):
    """
    Autocorrelates by passing in `x` as both input arguments to `correlate`,
    and normalises the result by 1/N (not included in scipy's implementation).
    """
    # Do autocorrelation: c_{av}[k] = sum_n x[n+k] * conj(x[n])
    acf = correlate(x, x, mode, method, axes)

    # Normalise: divide each term by n
    if axes is None:
        axes = tuple(range(x.ndim))
    else:
        axes = tuple(axes)

    N = np.ones(x.ndim, dtype=int)
    for ax in axes:
        N[ax] = acf.shape[ax]

    div = np.ones(N)
    if mode == "valid":
        for n in N:
            div *= n
    elif mode in {"same", "full"}:
        for ax, dim in enumerate(N):
            shape = np.ones(len(N), dtype=int)
            shape[ax] = dim
            d = (dim - np.abs(np.asarray(range(dim)) - int(dim / 2))).reshape(shape)
            div *= d

    return acf / div


def mahalanobis(x, mu, sig):
    """
    Compute the square of Mahalanobis distance [3] of all observations in `x`:
        s² = (x - μ) Σ⁻¹ (x - μ)*

    Parameters
    ----------
    x : ndarray (M, N)
        N-feature array of M observations of test points, for which the distance
        from the multivariate normal distribution described by `mu` and `sig` will
        be computed.
    mu : ndarray (N,)
        N-feature means.
    sig : ndarray (N, N)
        N-feature covariance.

    Returns
    -------
    s2 : ndarray (M,)
        Square of distance from feature means to test points, normalised by
        covariance.
    
    References
    ----------
    [2] - R. DE MAESSCHALCK, et al., The Mahalanobis distance, Chemometrics and
            Intelligent Laboratory Systems, Volume 50, Issue 1, p. 1-18, Jan
            2000, doi:10.1016/S0169-7439(99)00047-7

    """
    x, mu, sig = np.asarray(x), np.asarray(mu), np.asarray(sig)
    return np.einsum(
        'ij,ji->i',
        (x - mu.reshape(1, -1)).conj(),
        np.linalg.solve(np.atleast_2d(sig), (x - mu.reshape(1, -1)).transpose()),
    ).real


def mardia_test(x, alpha=0.05):
    """
    Perform the Mardia test [2] for normality with multi-dimensional data.
    Alternatively, check that Mahalanobis distance of `x` is distributed with χ².

    Parameters
    ----------
    x : ndarray (d, n)
        Data to be tested for normality, with features in axis 0 and
        observations in axis 1. If 1D, this may be relaxed to shape (n,).
    alpha : float, optional
        Significance level on which normality will be tested. The default is
        0.05.

    Returns
    -------
    result : bool
        True if both of the tests performed do not reject the null hypothesis
        (i.e. both p1 and p2 > alpha), False otherwise. Treat it as True/False
        of whether the provided data is normal.

    References
    ----------
    [3] - K. V. MARDIA, Measures of multivariate skewness and kurtosis with
            applications, Biometrika, Volume 57, Issue 3, p. 519–530, Dec 1970,
            doi:10.1093/biomet/57.3.519

    """
    x = np.asarray(x)
    if x.ndim < 2:
        x = x.reshape(1, -1)
    elif x.ndim > 2:
        raise ValueError(
            'x must be 2D containing rows of features and columns of observations.'
        )

    (d, n) = x.shape

    if x.dtype == complex:
        x = np.vstack([x.real, x.imag])

    mean = np.mean(x, axis=1).reshape(-1, 1)
    covr = np.atleast_2d(cov(x))

    a_to_sum = np.einsum('ji,jk', (x - mean).conj(), np.linalg.solve(covr, x - mean))
    a = 1 / (6 * n) * np.einsum('ij,ij,ij', a_to_sum, a_to_sum, a_to_sum).real
    b = np.sqrt(n / (8 * d * (d + 2))) * (
        1 / n * np.einsum('ii', a_to_sum**2) - d * (d + 2)
    )
    p1 = 1 - chi2.cdf(a, d * (d + 1) * (d + 2) / 6)
    p2 = 2 * (1 - norm.cdf(abs(b)))

    return p1 >= alpha and p2 >= alpha


def nancov(x, y=None):
    """
    A method to compute covariance which does not take nan values into account.
    This is not guaranteed to be any better or worse than the regular cov
    function - it will be about as good as numpy's nanmean function.

    Parameters
    ----------
    x : ndarray (M, N)
        M-feature array of N observations. If 1D, it is assumed that the array
        contains multiple observations of a single feature. May contain nan
        values.
    y : ndarray (O, P), optional
        Optional additional O-feature array of P observations. May not be the
        same size as `x`, but if not included then autocovariance of `x` is
        computed. The default is None.

    Returns
    -------
    cov : ndarray (M, O)
        Covariance matrix between `x` and `y`.

    """
    if y is None:
        y = x
    if x.ndim <= 1:
        x = x.reshape(1, -1)
    if y.ndim <= 1:
        y = y.reshape(1, -1)

    return np.nanmean(
        (
            (x - np.nanmean(x, axis=1).reshape(-1, 1)).reshape(
                x.shape[0], 1, x.shape[1]
            )
            * (y - np.nanmean(y, axis=1).reshape(-1, 1))
            .reshape(1, y.shape[0], y.shape[1])
            .conj()
        ),
        axis=-1,
    ).squeeze()


def opheim_simpl(x, y, tol):
    """
    Performs Opheim path simplification algorithm on (x, y) data. A path
    contained in advancing `x`, `y` will be simplified by tolerance `tol` with
    the algorithm:
    1) Select the first vertex as a `key`.
    2) Find the first vertex after `key` which is more than distance `tol`
       away (call it `next`) and links these two vertices with `line`.
    3) Find the vertices which exist between `key` and `next`. Find the last
       one which sits within `tol` of `line` found in (2). Alternatively, find
       the vertex which (when connected to its previous point) forms a segment
       which has an angle with `line` greater than 90° (to deal with spikes
       which should typically be preserved). Call it `last`. Remove all points
       between `key` and `last`.
    4) Set `key = last` and repeat from (2), until all points are exhausted.
    
    This function is mostly useful for plotting curves which may have a large
    number of points but are relatively smooth, e.g. the results of the `roc`
    function. Not all of the data is needed to visually represent it. Consider 
    using this if `matplotlib.pyplot` is really slow at plotting curves with a
    lot of points.

    Parameters
    ----------
    x : ndarray[float] (N,)
        x-coordiantes.
    y : ndarray[float] (N,)
        y-coordinates.
    tol : float
        Magnitude tolerance used to iterate through (x, y).

    Returns
    -------
    mask : ndarray[bool] (N,)
        Mask to be applied to `x` and `y` to remove the relevant data points.

    """
    x, y = np.asarray(x), np.asarray(y)
    if x.ndim != 1 or y.ndim != 1:
        raise ValueError('`x` and `y` must be 1-dimensional ndarrays.')
    if x.shape[0] != y.shape[0]:
        raise ValueError('`x` and `y` must have the same shape.')
    mask = np.full(x.shape, False)
    mask[0], mask[-1] = True, True

    i, N = 0, x.shape[0]
    while i < N - 2:
        # Find the first vertex beyond `tol`
        j = i + 1
        v = [x[j] - x[i], y[j] - y[i]]
        while j < N and np.linalg.norm(v) <= tol:
            j = j + 1
            v = [x[j] - x[i], y[j] - y[i]]
        v = v / np.linalg.norm(v)

        # Unit normal between `i`, `j`.
        norm = [v[1], -v[0]]

        # Find the last point which is within `tol` of the line connecting
        # point `i` to point `j`. Alternatively, the last point within a
        # direction change of pi/2.
        while j < N - 1:
            # Perpendicular distance from `i -> j` line
            v1 = [x[j + 1] - x[i], y[j + 1] - y[i]]
            d = np.abs(np.dot(norm, v1))
            if d > tol:
                break

            # Angle between line and current segment.
            v2 = [x[j + 1] - x[j], y[j + 1] - y[j]]
            cos = np.dot(v, v2)
            if cos <= 0:
                break

            j += 1
        i = j
        mask[i] = True

    return mask


def ply_orientation(grid, tfm, p=0.4e-3):
    """
    Computes the measured ply orientation from a TFM image of a composite
    material using the structure tensor. Note if `arim` is used to compute this
    `tfm`, then `tfm.res` should be passed to this function.

    Smoothing functions are hard-coded to follow the recommendations of Nelson
    et al. [4].

    Parameters
    ----------
    grid : arim.core.Grid
        The grid over which the TFM is computed. Only spacing is used, so
        technically it can be any object with attributes `dx`, `dy`, `dz`.
    tfm : ndarray[complex]
        TFM image produced over a composite material. In theory, it may be 2D
        or 3D, but this work relied on 2D data so the 3D case is not fully 
        implemented.
    p : float, optional
        Approximate ply thickness. Used in smoothing functions to produce the
        recommended kernel. The default is .4e-3.

    Returns
    -------
    ply_angle : ndarray[float]
        Measured ply angle relative to the last axis of the TFM image. Will
        have the same shape as `tfm`.

    References
    ----------
    [4] - L. J. NELSON, et al., Ply-orientation measurements in composites
            using structure-tensor analysis of volumetric ultrasonic data,
            Composites A, Volume 126, p. 105581, November 2019,
            doi:10.1016/j.compositesa.2017.10.027

    """
    # Compute sine and cosine of phase angle.
    phase = np.angle(tfm)
    sin, cos = np.sin(phase), np.cos(phase)

    # Smooth sine and cosine w/ Gaussian kernel:
    #   ```G1(r) = A * exp(-(r.T * C * r)/2)```
    # Standard deviation model: recommendation.
    sigxx, sigyy, sigzz = p, p, p / 15
    if tfm.ndim == 2:
        # Go up to 6σ, very little going on outside due to exponential decay.
        x = np.arange(0, 12 * sigxx, grid.dx)
        z = np.arange(0, 12 * sigzz, grid.dz)
        [x, z] = np.meshgrid(x - x.mean(), z - z.mean(), indexing="ij")
        r = np.vstack([x.ravel(), z.ravel()])
        # Covariance: `C = cov**(-1)`. Will invert inside `mahalanobis`.
        cov1 = [sigxx**2, sigzz**2] * np.eye(2)
        cov2 = [(2 * sigxx) ** 2, (2 * sigzz) ** 2] * np.eye(2)
        # Get nominal normal direction, used in ply angle calc later. Use z_hat.
        normal = np.asarray([0, 1]).reshape(-1, 1)
    elif tfm.ndim == 3:
        x = np.arange(0, 12 * sigxx, grid.dx)
        y = np.arange(0, 12 * sigyy, grid.dy)
        z = np.arange(0, 12 * sigzz, grid.dz)
        [x, y, z] = np.meshgrid(x - x.mean(), y - y.mean(), z - z.mean(), indexing="ij")
        r = np.vstack([x.ravel(), y.ravel(), z.ravel()])
        cov1 = [sigxx**2, sigyy**2, sigzz**2] * np.eye(3)
        cov2 = [(2 * sigxx) ** 2, (2 * sigyy) ** 2, (2 * sigzz) ** 2] * np.eye(3)
        normal = np.asarray([0, 0, 1]).reshape(-1, 1)
    else:
        raise ValueError(
            'Too many dimensions in `tfm` - should contain a single 2D or 3D TFM image.'
        )
    g1 = (
        # `A` defined s.t. ∫ G1(r) dr = 1
        (2 * np.pi) ** (-len(x.shape) / 2)
        * np.sqrt(np.linalg.det(cov1))
        * np.exp(-mahalanobis(r.transpose(), 0, cov1) / 2).reshape(x.shape)
    )
    sin_smoothed = convolve(sin, g1, mode='same')
    cos_smoothed = convolve(cos, g1, mode='same')

    # Compute derivatives:
    if tfm.ndim == 2:
        d = np.asarray([-1, 0, 1])
        s = np.asarray([3, 10, 3])
        # Could probably automate this better.
        scharr_kern = np.zeros((2, 3, 3))
        scharr_kern[0, :, :] = np.dot(d.reshape(-1, 1), s.reshape(1, -1)) / 32
        scharr_kern[1, :, :] = np.dot(s.reshape(-1, 1), d.reshape(1, -1)) / 32
    elif tfm.ndim == 3:
        raise NotImplementedError(
            '3D implementation not yet working - need to work out the Scharr kernel in 3D.'
        )
        # d = np.asarray([0])
        # s = np.asarray([0])
        # scharr_kern = np.zeros((3, 3, 3, 3))
    grad = np.zeros((tfm.ndim, *tfm.shape))
    for dim in range(tfm.ndim):
        grad[dim] = (
            cos_smoothed * convolve(sin_smoothed, scharr_kern[dim], mode='same')
            - sin_smoothed * convolve(cos_smoothed, scharr_kern[dim], mode='same')
        )

    # Create structure tensor
    structure = grad.reshape(tfm.ndim, 1, *tfm.shape) * grad.reshape(
        1, tfm.ndim, *tfm.shape
    )
    # Smoothing (`g2`) goes here.
    g2 = (
        (2 * np.pi) ** (-len(x.shape) / 2)
        * np.sqrt(np.linalg.det(cov2))
        * np.exp(-mahalanobis(r.transpose(), 0, cov2) / 2).reshape(x.shape)
    ).reshape(1, 1, *x.shape)
    # g2 = np.asarray([1]).reshape([1 for _ in structure.shape])
    structure_smoothed = convolve(structure, g2, mode='same').reshape(
        tfm.ndim, tfm.ndim, -1
    )

    # Compute eigenvectors and ply angle - 2D.
    evecs = np.vstack([
        2 * structure_smoothed[0, 1],
        (
            structure_smoothed[1, 1]
            - structure_smoothed[0, 0]
            + np.sqrt(
                (structure_smoothed[0, 0] - structure_smoothed[1, 1]) ** 2
                + 4 * structure_smoothed[0, 1] ** 2
            )
        ),
    ]).transpose()
    ply_angle = (
        np.arccos(
            np.dot(evecs, normal).squeeze()
            / (np.linalg.norm(evecs, axis=1) * np.linalg.norm(normal))
        )
        * np.sign([
            np.linalg.det(np.hstack([normal, evecs[i : i + 1, :].transpose()]))
            for i in range(evecs.shape[0])
        ])
    ).reshape(tfm.shape)

    return ply_angle


def roc(pos, neg, reduce_size=False):
    """
    Compute receiver-operating characteristic curve from known positive and
    negative distributions.

    This works by sorting `pos` and `neg` into ascending order and take the
    union:
    ```
    x, idxs = np.unique(np.vstack([pos, neg]), return_index=True)
    ```
    where `idxs` points to the location in the composite array that each
    element in `x` originally came from. As such, the part of `idxs` which
    comes from `pos` will be monotonically increasing as `pos` is ascending;
    and the part from `neg` will be also, with these two increasing arrays
    spliced together in some way. Flip the order so that `x` and `idxs` are
    decreasing.

    Considering a position `loc` in both `x` and `idxs` which originally comes
    from `pos`, then all of the values in the slice `pos[:idxs[loc]]` must be
    less than `x[loc]` because `pos` is ascending. Therefore `idxs[loc]` is
    equal to the number of elements in `pos` which are less than `x[loc]`.
    This is exactly the false -ve rate when divided by the total number of
    elements in `pos`, so subtract from 1 to get true +ve.

    Suppose that `loc+1` in `x` originally comes from `neg`. As `x` is
    descending, then `neg[idxs[loc+1]] < pos[idxs[loc]]`, thus the number of
    values in `pos` less than `x[loc+1]` is the same as the number of values
    less than `x[loc]`.

    Parameters
    ----------
    pos : ndarray (N1,)
        Observations made when the null hypothesis is true. Multiple features
        is not currently supported.
    neg : ndarray (N2,)
        Observations made when the null hypothesis is false. Multiple features
        is not currently supported.
    reduce_size : bool
        Arrays of thresholds, true positives and false positives can be huge.
        Set this to `True` to reduce the overall size of the array using Opheim
        simplification, with a tolerance of `1e-5`, i.e. steps should now be
        ~1e-5 between points.

    Returns
    -------
    x : ndarray (M,)
        The list of thresholds which are used, equivalent to the ascending
        values in the union of `pos` and `neg`.
    tps : ndarray (M,)
        True positive rate, or the percentage of weighted values in `pos` which
        exceed the value in `x`.
    fps : ndarray (M,)
        False positive rate, or the percentage of weighted values in `neg`
        which exceed `x`.

    """
    # Check data
    pos, neg = np.squeeze(pos), np.squeeze(neg)
    if pos.ndim < 2:
        pos = pos.reshape(-1, 1)
    elif pos.ndim > 2:
        raise ValueError(
            'pos expected to have 2 dimensions, found {}.'.format(pos.ndim)
        )
    elif pos.shape[1] != 1:
        warnings.warn(
            'pos should only have one feature, currently has {}.'.format(pos.shape[0])
        )
    if neg.ndim < 2:
        neg = neg.reshape(-1, 1)
    elif neg.ndim > 2:
        raise ValueError(
            'neg expected to have 2 dimensions, found {}.'.format(neg.ndim)
        )
    elif neg.shape[1] != 1:
        warnings.warn(
            'neg should only have one feature, currently has {}.'.format(neg.shape[0])
        )

    # # Check weights - leave out for now as working with weights will require
    # # some reformulation.
    # if pos_wts is None:
    #     pos_wts = np.ones(pos.shape[1])
    # else:
    #     pos_wts = np.squeeze(pos_wts)
    #     if pos_wts.ndim > 1 or pos_wts.shape[0] != pos.shape[1]:
    #         raise ValueError(
    #             'pos_wts expected to have shape equal to number of features in pos, found shape {}'
    #             .format(pos_wts.shape)
    #         )
    # if neg_wts is None:
    #     neg_wts = np.ones(neg.shape[1])
    # else:
    #     neg_wts = np.squeeze(neg_wts)
    #     if neg_wts.ndim > 1 or neg_wts.shape[0] != neg.shape[1]:
    #         raise ValueError(
    #             'neg_wts expected to have shape equal to number of features in neg, found shape {}'
    #             .format(neg_wts.shape)
    #         )

    # Sort into ascending order.
    pos, neg = np.sort(pos, axis=0)[::-1, :], np.sort(neg, axis=0)[::-1, :]
    # Sort unique values into descending order.
    x, idxs = np.unique(np.vstack([pos, neg]), return_index=True)
    # x, idxs = x[::-1], idxs[::-1]
    tps, fps = idxs.copy(), idxs.copy()

    for loc, idx in enumerate(idxs):
        # Find the parts of `tps` which come from `neg` and set them to the
        # most recently seen value in `pos`.
        if idx >= pos.size:
            # At the start, all values in the array must be < x[0].
            if loc == 0:
                tps[loc] = pos.size
            # At any other time, we must have either just seen a value from
            # `pos` on the previous iteration, or set the value on the previous
            # iteration. Either way, `loc-1` must contain a value from `pos`.
            else:
                tps[loc] = tps[loc - 1]
        # Find the parts of `fps` from `pos` and set them to the most recently
        # seen value in `neg`.
        else:
            if loc == 0:
                fps[loc] = pos.size + neg.size
            else:
                fps[loc] = fps[loc - 1]
    # `fps` must now point to the part of `np.vstack([pos, neg])` which comes
    # from `neg`, so correct for the presence of `pos`.
    fps -= pos.size

    # `tps` and `fps` contain the number of elements in `pos` and `neg` which
    # are less than a given threshold in `x`. To get true +ve and -ve rates,
    # want the number of elements larger than the threshold in `x`.
    tps, fps = tps / pos.size, fps / neg.size

    # Ensure that `tps` and `fps` both start at 1 and end at 0.
    tps, fps = np.hstack([1, tps, 0]), np.hstack([1, fps, 0])
    x = np.hstack([x[0], x, x[-1]])

    if reduce_size and x.shape[0] > 10000:
        try:
            mask = opheim_simpl(fps, tps, 1e-5)
        except Exception:
            try:
                mask = opheim_simpl(fps, tps, 2e-5)
            except Exception:
                mask = np.full(x.shape, True)
        x, tps, fps = x[mask], tps[mask], fps[mask]

    return x, tps, fps
