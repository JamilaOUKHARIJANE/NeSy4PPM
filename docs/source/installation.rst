Installation
============


Local Installation
------------------

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

PyPI Installation
-----------------
You can also install NeSy4PPM directly from `PyPI <https://pypi.org/project/nesy4ppm/>`_. We recommend using a virtual environment to avoid conflicts with your global Python packages::

    pip install nesy4ppm

Docker Installation
-------------------

You can also run NeSy4PPM using Docker:

1. `Install Docker <https://www.docker.com/get-started>`_

2. Pull and run the Docker image::

     docker pull jamilaoukharijane/nesy4ppm:latest
     docker run -p 8888:8888 jamilaoukharijane/nesy4ppm:latest

3. Open http://127.0.0.1:8888/tree in your browser to launch Jupyter to run tutorial