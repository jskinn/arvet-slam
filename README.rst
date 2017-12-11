==============================================
ARVET: A Robotic Vision Evaluation Tool - SLAM
==============================================

A set of extras for ARVET for working with SLAM systems.
Includes importers for common slam datasets:

 - KITTI_
 - `TUM RGBD evaluation datasets`_
 - `EuRoC drone dataset`_

Includes bindings for two common robotic vision systems. Acutally using either of these systems requires python
bindings to link them from the original C++

 - ORB_SLAM2_ (Python bindings at https://github.com/jskinn/ORB_SLAM2-PythonBindings )
 - LibVisO2_ (Python bindings at https://github.com/jlowenz/pyviso2 )

This module also includes 4 evaluation benchmarks:

- The absolute trajectory error and relative pose error implemented bu Jurgen Sturm,
and distributed as part of the TUM RGBD evaluation tools
(see https://vision.in.tum.de/data/datasets/rgbd-dataset/tools#evaluation )

- The trajectory drift evaluation used as part of the KITTI benchmark
(see http://www.cvlibs.net/datasets/kitti/eval_odometry.php )

- Tracking statistics for how often the system reports itself as lost (particularly relevant for ORB_SLAM2)

.. _KITTI: http://www.cvlibs.net/datasets/kitti/eval_odometry.php
.. _TUM RGBD evaluation datasets: https://vision.in.tum.de/data/datasets/rgbd-dataset
.. _EuRoC drone dataset: http://projects.asl.ethz.ch/datasets/doku.php?id=kmavvisualinertialdatasets

.. _ORB_SLAM2: https://github.com/raulmur/ORB_SLAM2
.. _LibVisO2: http://www.cvlibs.net/software/libviso/

License
=======

Except where otherwise noted in the relevant file, this code is licensed under the BSD 2-Clause licence, see LICENSE.
In particular, this module repackages code from Jurgen Strum, also under a BSD license
