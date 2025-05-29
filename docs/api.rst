API Reference
=============

This section provides detailed documentation for the MLArena API.

PreProcessor
-----------

.. autoclass:: mlarena.preprocessor.PreProcessor
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: fit

   .. automethod:: transform

   .. automethod:: fit_transform

   .. automethod:: analyze_features

   .. automethod:: get_encoding_recommendations

   .. automethod:: visualize_target_encoding

MLPipeline
----------

.. autoclass:: mlarena.pipeline.MLPipeline
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: fit

   .. automethod:: predict

   .. automethod:: evaluate

   .. automethod:: explain_model

   .. automethod:: explain_case

   .. automethod:: tune

   .. automethod:: threshold_analysis

Utils
-----

Input/Output
~~~~~~~~~~~~

.. automodule:: mlarena.utils.io_utils
   :members:
   :undoc-members:
   :show-inheritance:

Data Utilities
~~~~~~~~~~~~~~

.. automodule:: mlarena.utils.data_utils
   :members:
   :undoc-members:
   :show-inheritance:

Plot Utilities
~~~~~~~~~~~~~~

.. automodule:: mlarena.utils.plot_utils
   :members:
   :undoc-members:
   :show-inheritance:

Exceptions
---------

.. automodule:: mlarena.exceptions
   :members:
   :undoc-members:
   :show-inheritance: 