# -*- coding: utf-8 -*-
"""
A set of functions which make slight adaptations to some of those found in
`numpy`, to make life a bit easier. The useful functions are listed below, the
helper functions are not. 

nancov :        Compute the covariance, ignoring NaN values. Takes inputs and
                returns outputs in the same way as `np.cov()`.
cov :           Entirely identical to `np.cov`, apart from the fact that it
                takes `pseudo` as a keyword argument. Boolean value expected,
                if `True` then compute Cov[X, X*], if `False` then compute 
                Cov[X, X].
correlate :     Identical to `np.correlate` but also takes `axes` as a keyword
                argument, the axes to perform correlation over.
convolve :      Identical to `np.convolve`, but also takes `axes` as a keyword
                argument.
autocorrelate : Take a single input and correlate, but also do the 1/N
                normalisation which numpy neglects.
                
"""
__all__ = ['convolve', 'correlate', 'cov']
__version__ = '0.1'
__author__ = 'Matthew G. Chandler'

import numpy as np
from scipy.signal import choose_conv_method, fftconvolve
from scipy.signal import _sigtools
import warnings


def convolve(in1, in2, mode="full", method="auto", axes=None):
    """
    Copy of scipy's `convolve` to work with the additional `axes` parameter.
    """

    volume = np.asarray(in1)
    kernel = np.asarray(in2)

    if volume.ndim == kernel.ndim == 0:
        return volume * kernel
    elif volume.ndim != kernel.ndim:
        raise ValueError("volume and kernel should have the same " "dimensionality")

    if _inputs_swap_needed(mode, volume.shape, kernel.shape):
        # Convolution is commutative; order doesn't have any effect on output
        volume, kernel = kernel, volume

    if method == "auto":
        method = choose_conv_method(volume, kernel, mode=mode)

    if method == "fft":
        out = fftconvolve(volume, kernel, mode=mode, axes=axes)
        result_type = np.result_type(volume, kernel)
        if result_type.kind in {"u", "i"}:
            out = np.around(out)

        if np.isnan(out.flat[0]) or np.isinf(out.flat[0]):
            warnings.warn(
                "Use of fft convolution on input with NAN or inf"
                " results in NAN or inf output. Consider using"
                " method='direct' instead.",
                category=RuntimeWarning,
                stacklevel=2,
            )

        return out.astype(result_type)
    elif method == "direct":
        # fastpath to faster numpy.convolve for 1d inputs when possible
        if _np_conv_ok(volume, kernel, mode):
            return np.convolve(volume, kernel, mode)

        return correlate(volume, _reverse_and_conj(kernel), mode, "direct")
    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


_modedict = {"same": 1, "full": 2}

def correlate(in1, in2, mode="same", method="auto", axes=None):
    """
    Copy of scipy.signal's fftconvolve, with an additional axes parameter.
    """
    in1 = np.asarray(in1)
    in2 = np.asarray(in2)

    if in1.ndim == in2.ndim == 0:
        return in1 * in2.conj()
    elif in1.ndim != in2.ndim:
        raise ValueError("in1 and in2 should have the same dimensionality")

    # Don't use _valfrommode, since correlate should not accept numeric modes
    try:
        val = _modedict[mode]
    except KeyError as e:
        raise ValueError("Acceptable mode flags are" " 'same', or 'full'.") from e

    # this either calls fftconvolve or this function with method=='direct'
    if method in ("fft", "auto"):
        return convolve(in1, _reverse_and_conj(in2, axes=axes), mode, method, axes)

    elif method == "direct":
        # fastpath to faster numpy.correlate for 1d inputs when possible
        if _np_conv_ok(in1, in2, mode):
            return np.correlate(in1, in2, mode)

        # _correlateND is far slower when in2.size > in1.size, so swap them
        # and then undo the effect afterward if mode == 'full'.  Also, it fails
        # with 'valid' mode if in2 is larger than in1, so swap those, too.
        # Don't swap inputs for 'same' mode, since shape of in1 matters.
        swapped_inputs = (
            (mode == "full")
            and (in2.size > in1.size)
            or _inputs_swap_needed(mode, in1.shape, in2.shape, axes=axes)
        )

        if swapped_inputs:
            in1, in2 = in2, in1

        if mode == "valid":
            ps = [i - j + 1 for i, j in zip(in1.shape, in2.shape)]
            out = np.empty(ps, in1.dtype)

            z = _sigtools._correlateND(in1, in2, out, val)

        else:
            ps = [i + j - 1 for i, j in zip(in1.shape, in2.shape)]

            # zero pad input
            in1zpadded = np.zeros(ps, in1.dtype)
            sc = tuple(slice(0, i) for i in in1.shape)
            in1zpadded[sc] = in1.copy()

            if mode == "full":
                out = np.empty(ps, in1.dtype)
            elif mode == "same":
                out = np.empty(in1.shape, in1.dtype)

            z = _sigtools._correlateND(in1zpadded, in2, out, val)

        if swapped_inputs:
            # Reverse and conjugate to undo the effect of swapping inputs
            z = _reverse_and_conj(z, axes=axes)

        return z

    else:
        raise ValueError("Acceptable method flags are 'auto'," " 'direct', or 'fft'.")


def cov(
    m,
    y=None,
    rowvar=True,
    bias=False,
    ddof=None,
    fweights=None,
    aweights=None,
    *,
    dtype=None,
    pseudo=False
):
    """
    Alternative implementation of numpy's cov function, which may compute the
    pseudo-covariance matrix (in which Cov[X, X*] is computed instead of
    Cov[X, X]).
    """
    # Check inputs
    if ddof is not None and ddof != int(ddof):
        raise ValueError("ddof must be integer")

    # Handles complex arrays too
    m = np.asarray(m)
    if m.ndim > 2:
        raise ValueError("m has more than 2 dimensions")

    if y is not None:
        y = np.asarray(y)
        if y.ndim > 2:
            raise ValueError("y has more than 2 dimensions")

    if dtype is None:
        if y is None:
            dtype = np.result_type(m, np.float64)
        else:
            dtype = np.result_type(m, y, np.float64)

    X = np.array(m, ndmin=2, dtype=dtype)
    if not rowvar and X.shape[0] != 1:
        X = X.T
    if X.shape[0] == 0:
        return np.array([]).reshape(0, 0)
    if y is not None:
        y = np.array(y, copy=False, ndmin=2, dtype=dtype)
        if not rowvar and y.shape[0] != 1:
            y = y.T
        X = np.concatenate((X, y), axis=0)

    if ddof is None:
        if bias == 0:
            ddof = 1
        else:
            ddof = 0

    # Get the product of frequencies and weights
    w = None
    if fweights is not None:
        fweights = np.asarray(fweights, dtype=float)
        if not np.all(fweights == np.around(fweights)):
            raise TypeError("fweights must be integer")
        if fweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional fweights")
        if fweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and fweights")
        if any(fweights < 0):
            raise ValueError("fweights cannot be negative")
        w = fweights
    if aweights is not None:
        aweights = np.asarray(aweights, dtype=float)
        if aweights.ndim > 1:
            raise RuntimeError("cannot handle multidimensional aweights")
        if aweights.shape[0] != X.shape[1]:
            raise RuntimeError("incompatible numbers of samples and aweights")
        if any(aweights < 0):
            raise ValueError("aweights cannot be negative")
        if w is None:
            w = aweights
        else:
            w *= aweights

    avg, w_sum = np.average(X, axis=1, weights=w, returned=True)
    w_sum = w_sum[0]

    # Determine the normalization
    if w is None:
        fact = X.shape[1] - ddof
    elif ddof == 0:
        fact = w_sum
    elif aweights is None:
        fact = w_sum - ddof
    else:
        fact = w_sum - ddof * sum(w * aweights) / w_sum

    if fact <= 0:
        warnings.warn("Degrees of freedom <= 0 for slice", RuntimeWarning, stacklevel=2)
        fact = 0.0

    X -= avg[:, None]
    if w is None:
        X_T = X.T
    else:
        X_T = (X * w).T
    if not pseudo:
        c = np.dot(X, X_T.conj())
    else:
        c = np.dot(X, X_T)
    c *= np.true_divide(1, fact)
    return c.squeeze()


def _inputs_swap_needed(mode, shape1, shape2, axes=None):
    """
    Helper function from scipy.signal._signaltools
    """
    if mode != "valid":
        return False

    if not shape1:
        return False

    if axes is None:
        axes = range(len(shape1))

    ok1 = all(shape1[i] >= shape2[i] for i in axes)
    ok2 = all(shape2[i] >= shape1[i] for i in axes)

    if not (ok1 or ok2):
        raise ValueError(
            "For 'valid' mode, one must be at least "
            "as large as the other in every dimension"
        )

    return not ok1


def _np_conv_ok(volume, kernel, mode):
    """
    Helper function from scipy.signal._signaltools
    """
    if volume.ndim == kernel.ndim == 1:
        if mode in ("full", "valid"):
            return True
        elif mode == "same":
            return volume.size >= kernel.size
    else:
        return False


def _reverse_and_conj(x, axes=None):
    """
    Helper function from scipy.signal._signaltools
    """
    if axes is None:
        reverse = (slice(None, None, -1),) * x.ndim
    else:
        slices = [
            [
                slice(None),
            ]
            * x.ndim,
            [
                slice(None, None, -1),
            ]
            * x.ndim,
        ]
        reverse = tuple(slices[ax in np.mod(axes, x.ndim)][ax] for ax in range(x.ndim))
    return x[reverse].conj()