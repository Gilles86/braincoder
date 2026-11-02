==================
Migrating to 0.7
==================

Since braincoder 0.7, a model object holds only its configuration: the tuning function, the
stimulus grid, the HRF. It never stores data, parameters, weights or a noise model, so nothing you
did earlier can change what a later call computes.

Passing a removed argument to a constructor raises a ``TypeError`` that points here, and leaving
out required inputs raises a ``ValueError``, so old code fails loudly.

In short
========

.. list-table::
   :header-rows: 1
   :widths: 50 50

   * - Before 0.7
     - Since 0.7
   * - ``model = GaussianPRF(paradigm=x, parameters=p)``

       ``model.predict()``
     - ``model = GaussianPRF()``

       ``model.predict(x, p)``
   * - ``model.simulate(noise=.1)``
     - ``model.simulate(x, p, noise=.1)``
   * - ``GaussianPRF2D(grid_coordinates=g, paradigm=x)``
     - ``GaussianPRF2D(grid_coordinates=g)``
   * - ``GaussianPRFWithHRF(paradigm=x, hrf_model=h)``
     - ``GaussianPRFWithHRF(hrf_model=h)``
   * - ``RegressionGaussianPRF(paradigm=x, regressors=r)``
     - ``RegressionGaussianPRF(design_data=x, regressors=r)``
   * - ``model.init_pseudoWWT(grid, p)``

       ``ResidualFitter(model, data, x, parameters=p).fit()``
     - ``ResidualFitter(model, data, x, parameters=p).fit(stimulus_range=grid)``
   * - ``model.get_pseudoWWT()``
     - ``model.get_pseudoWWT(grid, p)``
   * - ``model.weights = w``

       ``model.get_WWT()``
     - ``model.get_WWT(w)``
   * - ``model.get_stimulus_pdf(data, grid)``
     - ``model.get_stimulus_pdf(data, grid, parameters=p, omega=omega, dof=dof)``
   * - ``model.get_rf()``
     - ``model.get_rf(p)``
   * - ``linear = model.to_linear_model()``
     - ``linear, baseline, weights = model.to_linear_model(p)``
   * - ``xy_model = model.to_xy_model()``
     - ``xy_model, xy_parameters = model.to_xy_model(p)``
   * - ``CustomStimulusFitter(data, model, stimulus, omega)``
     - ``CustomStimulusFitter(data, model, stimulus, omega, parameters=p, weights=w)``
   * - ``SPMHRFModel(tr=1.6, unique_hrfs=True)``
     - ``SPMHRFModel(tr=1.6)``
   * - ``model.apply_mask(mask)``
     - Select the voxels in your own ``data``, ``parameters`` and ``weights``.

``ParameterFitter(model, data, paradigm)`` and ``WeightFitter(model, parameters, data, paradigm)``
are unchanged. ``ResidualFitter`` and ``StimulusFitter`` keep the same arguments, but ``parameters``
is now required for any model that has parameters.

New fitting default
===================

``ParameterFitter.fit`` now defaults to ``r2_atol=1e-4``, so fits stop earlier. To reproduce a fit from an
earlier version, pass ``r2_atol=1e-6``.

Custom models
=============

Subclasses need two changes:

* Remove ``paradigm``, ``data``, ``parameters``, ``weights`` and ``omega`` from ``__init__``, and from the
  call to ``super().__init__``.
* Replace ``self.parameters``, ``self.weights``, ``self.omega`` and ``self.paradigm`` with method arguments.

``_basis_predictions(paradigm, parameters)`` and ``transformations`` work as before.

A subclass whose paradigm has more than one stimulus dimension and that uses ``StimulusFitter`` should
pass ``n_dimensions`` to the base constructor.
