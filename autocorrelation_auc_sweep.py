#!/usr/bin/env python3
# encoding: utf-8
"""
An overall script encapsulating the process steps for the auto-correlation and
hypothesis testing method for detection of area-based defects [1]. For a given
window, determine the deviation from a fitted multivariate normal distribution
within a set of images, and compute the ROC curve.

Call this from a terminal with 
```
>> python autocorrelation_auc_sweep.py <box_idx, opt.>
```
where <box_idx> is an integer which selects a window size from the unravelled
arrays `i_widths` and `j_widths` defined on line 173. If one is not provided, a
default value of 208 is used.

Arim [2] is a module used for TFM image generation. While the original repo
(https://github.com/ndtatbristol/arim) should work for this, I am working in a
different fork (https://github.com/mgchandler/arim-mgc).

This project relied on BluePebble, an HPC system made available by the ACRC at
the University of Bristol (https://www.bristol.ac.uk/acrc). It is intended that
this repo will be stored at data.bris, the University of Bristol data
repository, on publication of this work.

References
----------
[1] - M. G. CHANDLER, et al., A multivariate statistical approach to wrinkling
        detection in composites, (unpublished)
[2] - N. BUDYN, et al., A Model for Multiview Ultrasonic Array Inspection of
        Small Two-Dimensional Defects, IEEE Transactions on Ultrasonics,
        Ferroelectrics, and Frequency Control, Volume 66, p. 1129-1139, Jun
        2019, doi:10.1109/TUFFC.2019.2909988

"""
import arim.im
import arim.models.block_in_immersion as bim
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
import sys

import hypothesis_testing as ht


if __name__ == '__main__':
    # Allow iteration by passing in an argument at the system level. Else
    # manually change it below.
    if len(sys.argv) > 1:
        box_idx = int(sys.argv[1])
    else:
        box_idx = 208
        
    dirname = os.path.join('..', 'Raw Ultrasonic Data')

    # TFM parameters
    xmin, xmax = -15.0e-3, 15.0e-3
    zmin, zmax = 0.75e-3, 3.675e-3
    pix   = 0.11e-3
    nload = 100
    
    wrinkled_positions = {'x3200', 'x4800'}#
    ignored_positions = ()#{'x3200'}
    nsamples   = len(os.listdir(dirname))
    npositions = len(os.listdir(
        os.path.join(dirname, os.listdir(dirname)[0])
    ))
    nwrinkled  = len(wrinkled_positions)
    nignored   = len(ignored_positions)
    
    grid = arim.geometry.Grid(
        xmin=xmin,
        xmax=xmax,
        ymin=0.0,
        ymax=0.0,
        zmin=zmin,
        zmax=zmax,
        pixel_size=pix,
    )

    # How arim computes nx, nz
    nx = round((abs(xmax - xmin) + pix) / pix)
    nz = round((abs(zmax - zmin) + pix) / pix)
    
    conf = arim.io.load_conf_file('conf_imm_composite.yaml')
    conf_frame = arim.io.frame_from_conf(conf)
    
    # TFM storage. Start by trying to load TFMs if they have been precomputed
    # and saved.
    try:
        wrinkled = np.load(os.path.join('..', 'TFMs', 'wrinkled_tfm_{:n}mm.npy'.format(pix*1e3)))
        # wrinkled = wrinkled[:, 900:]
        pristine = np.load(os.path.join('..', 'TFMs', 'pristine_tfm_{:n}mm.npy'.format(pix*1e3)))
    except FileNotFoundError:
        wrinkled = np.zeros((nx*nz, 0), dtype=complex)
        pristine = np.zeros((nx*nz, 0), dtype=complex)
        
        
        # %% Produce TFMs
        for sample in sorted(os.listdir(dirname)):
            gs = int(sample.split(' ')[0][1:])
            
            for position in sorted(os.listdir(os.path.join(dirname, sample))):
                folder = os.path.join(dirname, sample, position)
                I = np.zeros((nx, nz, nload), dtype=complex) # TFM storage
                
                for i, file in enumerate(sorted(os.listdir(folder))):
                    filepath = os.path.join(folder, file)
                    
                    # Import data, preprocess
                    frame = arim.io.load_expdata(filepath)
                    frame = frame.apply_filter(
                        arim.signal.Hanning(
                            frame.numsamples,
                            frame.probe.frequency/2,
                            frame.probe.frequency/2,
                            frame.time,
                        )
                    )
                    frame = frame.expand_frame_assuming_reciprocity()
        
                    # Apply consistent options
                    frame.probe.dead_elements = conf_frame.probe.dead_elements
                    frame.examination_object = conf_frame.examination_object
                    # frame.time = frame.time.from_vect(conf_frame.time.samples)
        
                    # Measure probe position by measuring frontwall
                    (
                        probe_standoff,
                        probe_angle, 
                        time_to_surface
                    ) = arim.measurement.find_probe_loc_from_frontwall(
                        frame, frame.examination_object.couplant_material,  # tmin=7.5e-6
                    )
                    
                    # Do the imaging
                    views = bim.make_views(
                        frame.examination_object,
                        frame.probe.to_oriented_points(),
                        grid.to_oriented_points(),
                        tfm_unique_only=True,
                        max_number_of_reflection=0,
                    )
                    # Inconsistency between my fork of arim and the upstream one
                    # wrt spaces in view names - as we only use direct views, it
                    # doesn't matter. Force spaces if none exist
                    if 'L-L' in views.keys():
                        views['L - L'] = views['L-L']
                    arim.ray.ray_tracing(views.values(), convert_to_fortran_order=True)
                    
                    tfm = arim.im.tfm.tfm_for_view(
                        frame,
                        grid,
                        views['L - L'],
                        interpolation='linear',
                    )
                    I[:, :, i] = tfm.res.squeeze()
                
                # Store the data. Append is a bit slow, but it's more readable
                # than predefining storage of the right size and incrementing
                # counters.
                flatI = I.reshape((nx*nz, nload))
                
                if position in wrinkled_positions:
                    wrinkled = np.append(wrinkled, flatI, axis=1)
                else:
                    pristine = np.append(pristine, flatI, axis=1)
                    
        np.save(os.path.join('..', 'TFMs', 'wrinkled_tfm_{:n}mm.npy'.format(pix*1e3)), wrinkled)
        np.save(os.path.join('..', 'TFMs', 'pristine_tfm_{:n}mm.npy'.format(pix*1e3)), pristine)
            
            
    # %% Fit multivariate normal distributions to autocorrelation. Only do this
    #    one box, defined at the top.
    
    i_widths, j_widths = np.meshgrid(
        np.arange(1, int((zmax-zmin)/pix), 1, dtype=int),
        np.arange(1, int((zmax-zmin)/pix), 1, dtype=int),
    )
    i_width, j_width = i_widths.ravel()[box_idx], j_widths.ravel()[box_idx]
    
    # Subset of pristine data for training.
    subset = np.full(pristine.shape[1], True)
    subset[np.random.choice(pristine.shape[1], wrinkled.shape[1], replace=False)] = False
    training_sliding_view = sliding_window_view(
        pristine[:, subset].reshape(nx, nz, -1),
        (i_width, j_width),
        (0, 1),
    )
    x_strides, z_strides = training_sliding_view.shape[:2]
    
    # bool array for unique part of autocorrelation
    unique = np.full((i_width, j_width), False)
    unique[:int(i_width/2), :] = True
    unique[int(i_width/2), :int(j_width/2)+1] = True
    
    # Storage
    mean = np.zeros((z_strides, unique.sum()), dtype=complex)
    cov  = np.zeros((z_strides, unique.sum(), unique.sum()), dtype=complex)
    pcov = np.zeros((z_strides, unique.sum(), unique.sum()), dtype=complex)
    
    # Depth dependent distribution parameters
    for j in range(z_strides):
        acf = ht.autocorrelate(
            training_sliding_view[:, j, :, :, :],
            mode='same',
            axes=(-2, -1),
        )
        acf = acf[:, :, unique]
        
        mean[j, :] = np.nanmean(acf, axis=(0, 1))
        cov[j, :, :]  = ht.cov(
            acf.reshape(acf.shape[0] * acf.shape[1], unique.sum()).transpose(),
        )
        pcov[j, :, :] = ht.cov(
            acf.reshape(acf.shape[0] * acf.shape[1], unique.sum()).transpose(),
            pseudo=True,
        )
        

    # %% s-maps. Function of sample (`nsamples`), acquisition (`nload`),
    #    overall position (`nwrinkled`), and position within the TFM
    #    (`xstrides`, `z_strides`).
    
    # Reshape TFMs so that we are stacking samples.
    subset = subset.reshape((nsamples, (npositions - nwrinkled - nignored) * nload))
    wrinkled = wrinkled.reshape((nx, nz, nsamples, nwrinkled * nload))
    pristine = pristine.reshape((nx, nz, nsamples, (npositions - nwrinkled - nignored) * nload))
    
    # Storage
    wrinkled_s2 = np.zeros(
        (x_strides, z_strides, nsamples, nwrinkled * nload),
        dtype=float,
    )
    # Variable length in last axis due to randomness in `subset`. Fill blanks
    # with NaN and then ignore these values later.
    pristine_s2 = np.full(
        (x_strides, z_strides, nsamples, (npositions - nwrinkled - nignored) * nload),
        np.nan,
        dtype=float,
    )
    
    for gs in range(nsamples):
        wrinkled_sliding_view = sliding_window_view(
            wrinkled[:, :, gs, :],
            (i_width, j_width),
            (0, 1),
        )
        pristine_sliding_view = sliding_window_view(
            pristine[:, :, gs, ~subset[gs, :]],
            (i_width, j_width),
            (0, 1),
        )
        pristine_subset_size = (~subset).sum(axis=1)[gs]
        
        for j in range(z_strides):
            
            # Wrinkled image
            wrinkled_acf = ht.autocorrelate(
                wrinkled_sliding_view[:, j, :, :, :],
                mode='same',
                axes=(-2, -1),
            )
            wrinkled_acf = wrinkled_acf[:, :, unique].reshape(
                x_strides * nwrinkled * nload, unique.sum()
            )
            wrinkled_acf = np.hstack([wrinkled_acf.real, wrinkled_acf.imag[:, :-1]])
            
            # Pristine image
            pristine_acf = ht.autocorrelate(
                pristine_sliding_view[:, j, :, :, :],
                mode='same',
                axes=(-2, -1),
            )
            pristine_acf = pristine_acf[:, :, unique].reshape(
                x_strides * pristine_subset_size, unique.sum()
            )
            pristine_acf = np.hstack([pristine_acf.real, pristine_acf.imag[:, :-1]])
            
            # Distribution parameters
            mu = mean[j, :].reshape(1, -1)
            mu = np.hstack([mu.real, mu.imag[:, :-1]])
                            
            g = cov[j, :, :]
            c = pcov[j, :, :]
            sg = 0.5 * np.vstack([
                np.hstack([(g + c).real, (-g + c).imag[:, :-1]]),
                np.hstack([(g + c).imag[:-1, :], (g - c).real[:-1, :-1]]),
            ])
            
            # Normalised distance from distribution
            wrinkled_s2[:, j, gs, :] = ht.mahalanobis(wrinkled_acf, mu, sg).reshape(
                x_strides, nwrinkled * nload
            )
            pristine_s2[:, j, gs, ~subset[gs, :]] = ht.mahalanobis(pristine_acf, mu, sg).reshape(
                x_strides, pristine_subset_size
            )
            
    
    # %% ROCs and AUCs. Take the mean value of s2 within a given image as the
    #    test statistic. Gives an impression of spatial extent and overall
    #    departure from N_k.
    
    # Storage. Expect that each set might be different lengths, so can't use
    # numpy.
    aucs = {}
    xs, tps, fps = {}, {}, {}
    
    for gs in range(nsamples):
        x, tp, fp = ht.roc(
            wrinkled_s2[:, :, gs, :].mean(axis=(0, 1)).ravel(),
            pristine_s2[:, :, gs, ~subset[gs, :]].mean(axis=(0, 1)).ravel(),
            reduce_size=True,
        )
        aucs['g{}'.format(gs+1)] = ht.auc(fp, tp)
        xs['g{}'.format(gs+1)]   = x
        tps['g{}'.format(gs+1)]  = tp
        fps['g{}'.format(gs+1)]  = fp
    
    x, tp, fp = ht.roc(
        wrinkled_s2[:, :, :, :].mean(axis=(0, 1)).ravel(),
        pristine_s2[:, :, ~subset].mean(axis=(0, 1)).ravel(),
        reduce_size=True,
    )
    aucs['agg'] = ht.auc(fp, tp)
    xs['agg']   = x
    tps['agg']  = tp
    fps['agg']  = fp