# Multivariate Distribution Fitting for Ultrasonic Data Acquired in Composite Material

A working repository encapsulating the process steps for the auto-correlation and
hypothesis testing method for detection of area-based defects [1]. For a given
window, determine the deviation from a fitted multivariate normal distribution
within a set of images, and compute the ROC curve.

Call this from a terminal with 
```
>> python autocorrelation_auc_sweep.py <box_idx, opt.>
```
where `<box_idx>` is an integer which selects a window size from the unravelled
arrays `i_widths` and `j_widths` defined on line 173. If one is not provided, a
default value of 208 is used.

Arim [2] is a module used for TFM image generation. While the [original repo](https://github.com/ndtatbristol/arim)
should work for this, I am working in a [different fork](https://github.com/mgchandler/arim-mgc).

The results of this method are compared with the structure tensor method for the
extraction of ply orientation [3].

This project relied on BluePebble, an HPC system made available by the [ACRC at
the University of Bristol](https://www.bristol.ac.uk/acrc). It is intended that
this repo will be stored at data.bris, the University of Bristol data
repository, on publication of this work.

## References
----------
1. M. G. Chandler, et al., A multivariate statistical approach to wrinkling
   detection in composites, (unpublished)
2. [N. Budyn, et al., A Model for Multiview Ultrasonic Array Inspection of
   Small Two-Dimensional Defects, IEEE Transactions on Ultrasonics,
   Ferroelectrics, and Frequency Control, Volume 66, p. 1129-1139
   (2019)](https://doi.org/10.1109/TUFFC.2019.2909988)
4. [L. J. Nelson, et al., Ply-orientation measurements in composites using
   structure-tensor analysis of volumetric ultrasonic data, Composites A,
   Volume 104, pp. 108-119 (2018)](https://doi.org/10.1016/j.compositesa.2017.10.027)
