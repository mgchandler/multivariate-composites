#!/usr/bin/env python3
# encoding: utf-8
"""
A script which computes the ply orientation from the eigenvectors of the
structure tensor of a given TFM. The ROC curve is then calculated.

This project relied on BluePebble, an HPC system made available by the ACRC at
the University of Bristol (https://www.bristol.ac.uk/acrc).

References
----------
[1] - L. J. NELSON, et al., Ply-orientation measurements in composites using
        structure-tensor analysis of volumetric ultrasonic data, Composites A,
        Volume 104, p. 108-119, Jan 2018, doi:10.1016/j.compositesa.2017.10.027
[2] - M. G. CHANDLER, et al., A multivariate statistical approach to wrinkling
        detection in composites, (unpublished)
"""
import arim.im
import arim.models.block_in_immersion as bim
from concurrent.futures import ProcessPoolExecutor
from functools import partial
import numpy as np
from numpy.lib.stride_tricks import sliding_window_view
import os
import sys

import hypothesis_testing as ht


if __name__ == '__main__':
        
    dirname = os.path.join('..', 'Raw Ultrasonic Data')

    # TFM parameters
    xmin, xmax = -15.0e-3, 15.0e-3
    zmin, zmax = 0.75e-3, 3.675e-3
    pix   = 0.11e-3
    nload = 100
    
    wrinkled_positions = {'x3200', 'x4800'}
    nsamples   = len(os.listdir(dirname))
    npositions = len(os.listdir(
        os.path.join(dirname, os.listdir(dirname)[0])
    ))
    nwrinkled  = len(wrinkled_positions)
    
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
            
            
    # %% Compute the ply orientation. This has been wrapped up inside `ht`.
    #    Still need a subset to have comparable distribution size.
    subset = np.full(pristine.shape[1], True)
    subset[np.random.choice(pristine.shape[1], wrinkled.shape[1], replace=False)] = False
    subset = subset.reshape((nsamples * (npositions - nwrinkled) * nload))
    
    wrinkled = wrinkled.reshape(
        (nx, nz, nsamples * nwrinkled * nload)
    ).transpose(2, 0, 1)
    pristine = pristine.reshape(
        (nx, nz, nsamples * (npositions - nwrinkled) * nload)
    ).transpose(2, 0, 1)

    wrinkled_theta =np.zeros(
        (nx, nz, nsamples * nwrinkled * nload),
        dtype=float,
    )
    pristine_theta = np.zeros(
        (nx, nz, nsamples * (npositions - nwrinkled) * nload),
        dtype=float,
    )
    
    where = np.where(~subset)[0]
    with ProcessPoolExecutor() as pool:
        for i, res in enumerate(
            pool.map(partial(ht.ply_orientation, grid), wrinkled)
        ):
            wrinkled_theta[:, :, i] = res
        for i, res in enumerate(
            pool.map(partial(ht.ply_orientation, grid), pristine[~subset, :, :])
        ):
            pristine_theta[:, :, where[i]] = res
    
    wrinkled_theta = wrinkled_theta.reshape(nx, nz, nsamples, nwrinkled * nload)
    pristine_theta = pristine_theta.reshape(nx, nz, nsamples, (npositions - nwrinkled) * nload)
    subset = subset.reshape(nsamples, (npositions - nwrinkled) * nload)
    
    
    # %% ROCs and AUCs. Take the absolute value of orientation and calculate
    #    the mean to use as the test statistic.
    
    # Storage. Expect that each set might be different lengths, so can't use
    # numpy.
    aucs = {}
    xs, tps, fps = {}, {}, {}
    
    for gs in range(nsamples):
        x, tp, fp = ht.roc(
            np.abs(wrinkled_theta[:, :, gs, :]).mean(axis=(0, 1)).ravel(),
            np.abs(pristine_theta[:, :, gs, ~subset[gs, :]]).mean(axis=(0, 1)).ravel(),
            reduce_size=True,
        )
        aucs['g{}'.format(gs+1)] = ht.auc(fp, tp)
        xs['g{}'.format(gs+1)]   = x
        tps['g{}'.format(gs+1)]  = tp
        fps['g{}'.format(gs+1)]  = fp
    
    x, tp, fp = ht.roc(
        np.abs(wrinkled_theta[:, :, :, :]).mean(axis=(0, 1)).ravel(),
        np.abs(pristine_theta[:, :, ~subset]).mean(axis=(0, 1)).ravel(),
        reduce_size=True,
    )
    aucs['agg'] = ht.auc(fp, tp)
    xs['agg']   = x
    tps['agg']  = tp
    fps['agg']  = fp