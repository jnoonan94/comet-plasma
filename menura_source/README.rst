Introduction
=================

``Menura`` is an open-source multi-GPU numerical model developed for various space physics applications.

.. image:: ../../gallery/illustration_B_perp_iono.png
  :width: 800
  :alt: B_perp

``Menura`` is built around a hybrid Particle-In-Cell solver,
treating electrons as a charge-neutralising fluid, and ions as massive particles.
It solves iteratively the particles' dynamics, gathers particle moments at the nodes of a grid,
at which the magnetic field is also computed, and then solves the Maxwell equations.
This solver uses the popular Current Advance Method (CAM).
