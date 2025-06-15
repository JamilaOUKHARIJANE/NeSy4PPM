Tutorials
-------------------

Here, you will find tutorials explaining how to utilize the different functionalities of NeSy4PPM.
All tutorials can be downloaded and run from our
`Github. <>`_
List of available tutorials:

.. toctree::
   :maxdepth: 1
   :glob:

   tutorials/*

Tutorials
=========
The ``docs/tutorials/`` folder contains hands-on Jupyter notebooks that guide users through the main functionalities of NeSy4PPM.
The tutorials use the `Sepsis cases log <https://data.4tu.nl/articles/dataset/Sepsis_Cases_-_Event_Log/12707639>`_ as a running example.

1. Learning:

   a. `Event log encoding <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/docs/tutorials/1.Encoding_Event_Logs.ipynb>`_: methods for encoding event logs into numeric vectors;
   b. `NN model training <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/docs/tutorials/2.Training_Neural_Model.ipynb>`_: training LSTM or Transformer models using encoded log;
   c. `Full learning <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/docs/tutorials/3.Learning.ipynb>`_: full pipeline for Neural model learning.
2. `Prediction <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/docs/source/tutorials/4.Predicting_Suffix.ipynb>`_: methods for predicting suffix using a trained Neural model and an external BK;
3. `End to end pipeline <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/docs/source/tutorials/4.Learning_and_Predicting_Suffix.ipynb>`_: combines Neural model learning and prediction.


.. toctree::
   :maxdepth: 2
   :caption: Notebooks

   tutorials/*