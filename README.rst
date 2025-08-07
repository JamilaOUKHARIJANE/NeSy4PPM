NeSy4PPM documentation
======================

NeSy4PPM is the first Python package designed for both single-attribute (e.g., activity) and multi-attribute (e.g., activity and resource) suffix prediction in predictive process monitoring. It implements a Neuro-Symbolic (NeSy) system that integrates neural models with various types of symbolic background knowledge (BK), enabling accurate and compliant predictions even under concept drift.

NeSy4PPM offers the following key features:

1. **Symbolic knowledge integration**: supports declarative and procedural BK, including DECLARE, MP-DECLARE (multi-perspective DECLARE), ProbDECLARE (probabilistic DECLARE), and Petri nets.

2. **Flexible learning**: provides multiple prefix encoding methods and supports LSTM (Long Short-Term Memory) and Transformer architectures.

3. **Drift-aware prediction**: contextualizes neural predictions using BK in real-time, enhancing prediction accuracy and compliance in dynamic environments.

Installation
============
For installing NeSy4PPM, you have the following three alternative methods:

1. Local Installation
---------------------

1. Clone or download the NeSy4PPM project::

     git clone https://github.com/JamilaOUKHARIJANE/NeSy4PPM.git
     cd NeSy4PPM

2. Create and activate a virtual environment (Python 3.10 required).

   **Using venv:**

   .. code-block:: bash

      python -m venv nesy4ppm-env
      source nesy4ppm-env/bin/activate  # On Windows use: nesy4ppm-env\Scripts\activate

   **Using Conda:**

   .. code-block:: bash

      conda create -n nesy4ppm-env python=3.10
      conda activate nesy4ppm-env

3. Install dependencies::

     pip install -r docs/source/requirements.txt

4. Install the package in editable mode::

     pip install -e .

2. PyPI Installation
--------------------
You can install NeSy4PPM directly from `PyPI <https://pypi.org/project/nesy4ppm/>`_. We always recommend using a virtual environment to avoid conflicts with your global Python packages::

    pip install nesy4ppm

3. Docker Installation
----------------------

You can run NeSy4PPM in an isolated Docker container:

1. `Install Docker <https://www.docker.com/get-started>`_

2. Pull and run the Docker image::

     docker pull jamilaoukharijane/nesy4ppm:latest
     docker run -it -p 8888:8888 jamilaoukharijane/nesy4ppm:latest

3. Open Jupyter Notebook in your browser: navigate to http://127.0.0.1:8888/tree to access the NeSy4PPM code and tutorials.

Usage
=====

You will find tutorials explaining how to utilize the different functionalities of NeSy4PPM.

- All tutorials are available in the ``docs/source/tutorials/Suffix_Prediction_tutorial.ipynb`` or on `Github. <https://github.com/JamilaOUKHARIJANE/NeSy4PPM/blob/master/docs/source/tutorials/Suffix_Prediction_tutorial.ipynb>`_
- You can download and run them locally in a Jupyter environment.
- The tutorials use the
  `Helpdesk log <https://data.4tu.nl/articles/_/12675977/1>`_ as a running example, but you can easily adapt it to work with other event logs.

Repository Structure
====================
- ``NeSy4PPM/Data_preprocessing``: contains the implementation of event log loading and data preparation for Neural Networks model training.
- ``NeSy4PPM/Training``: contains the implementation of Neural Networks model training.
- ``NeSy4PPM/ProbDeclmonitor``: contains the implementation of Probabilistic Declare conformance checking.
- ``NeSy4PPM/Prediction``: contains the implementation of suffix prediction using a contextualized Neural predictions with BK.
- ``Evaluation.py``: provides evaluation script for assessing the NeSy4PPM prediction performance.
- ``docs/source/tutorials/``: contains step-by-step guides and examples to help users get started with NeSy4PPM.
