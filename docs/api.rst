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

ML_PIPELINE
----------

.. autoclass:: mlarena.pipeline.ML_PIPELINE
   :members:
   :undoc-members:
   :show-inheritance:

   .. automethod:: __init__

   .. automethod:: fit

   .. automethod:: predict

   .. automethod:: predict_proba

   .. automethod:: evaluate

   .. automethod:: cross_validate

   .. automethod:: save

   .. automethod:: load

   .. automethod:: get_feature_importance

Exceptions
---------

.. automodule:: mlarena.exceptions
   :members:
   :undoc-members:
   :show-inheritance:

Utilities
--------

.. automodule:: mlarena.utils
   :members:
   :undoc-members:
   :show-inheritance: 