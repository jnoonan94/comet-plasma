# Plasma model styles used in comet science (tutorials D–F)

Three self-contained, runnable notebooks, one per model family in Goetz et al. (2022),
*The plasma environment of comet 67P/Churyumov-Gerasimenko*, Space Sci. Rev. 218:65 (Sect. 3.3, Table 5).
Every notebook builds its model from scratch in numpy/scipy, checks it against an analytic or
textbook result, and links the toy code to the production codes used in the literature.

| Folder | Model family | What is simulated | Built-in validation | Runtime* |
|---|---|---|---|---|
| `D_fluid_models/` | **Fluid / MHD**: all species are fluids | (1) Inner-coma ion continuity with ionisation and recombination; (2) 1-D mass-loaded MHD along the Sun–comet line, sub- and super-critical | Galand et al. (2016) and Beth et al. (2019) analytic profiles; exact Biermann-type steady MHD solution and critical mass loading | ~20 s |
| `E_hybrid_models/` | **Hybrid**: ions are particles, electrons a massless fluid | 2-D hybrid PIC of a weakly outgassing comet: water-ion pickup and solar-wind deflection | Boris pusher E×B drift and cycloid test; opposite deflection of the two ion populations (Behar et al. 2016, 2018) | ~2 min |
| `F_fully_kinetic_models/` | **Fully kinetic PIC**: ions and electrons are particles | 1-D electrostatic PIC of an ionising coma and its ambipolar potential, at three ion/electron mass ratios | Plasma-oscillation frequency; Boltzmann relation; Bohm criterion; ln√μ scaling, extrapolated to H₂O⁺ | ~1 min |

\*Measured on a cloud CPU; a laptop should be similar.

Suggested order: D → E → F. Each regime fails where the next one begins: fluid models break once
pickup gyroradii exceed the interaction region, and hybrid models cannot give electron distributions.
Tutorial E maps directly onto the Menura source (`menura_source/`, see `B_menura_guide/`).

Requirements: Python ≥ 3.9, numpy, scipy, matplotlib, jupyter.
