.. opstool documentation master file, created by
   sphinx-quickstart on Fri Dec  2 02:37:37 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to gmspy's documentation!
===================================

``gmspy`` is a python package for dealing with ``ground motions time-histories`` induced by earthquakes,
including various ``intensity measures (IMs)``, ``elastic response spectra``, ``constant ductility response spectra``, ``pre-processing``, etc.

To use, install `gmspy` from 
`gmspy-PyPI <https://pypi.org/project/gmspy/>`_:

.. code-block:: console

   pip install --upgrade gmspy

It is recommended that you use `Anaconda <https://www.anaconda.com/>`_ to avoid library version incompatibilities.

.. toctree::
   :maxdepth: 5
   :caption: Instructions

   CHANGELOG
   src/intro.ipynb

.. toctree::
   :maxdepth: 3
   :caption: Gallery

   sphinx_gallery_examples/index


.. toctree::
   :maxdepth: 5
   :caption: Opstool Package Index

   src/gmspy.rst

