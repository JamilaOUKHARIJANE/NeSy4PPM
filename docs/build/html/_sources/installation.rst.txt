Installation
============

We recommend using a virtual environment to avoid conflicts between your local Python setup and required libraries. You can use `Conda <https://conda.io>`_ or `venv <https://docs.python.org/3/library/venv.html>`_:

1. Clone/download the NeSy4PPM project.
2. Activate your virtual environment.
3. Install the dependencies listed in ``requirements.txt`` using:: pip install -r requirements.txt

Alternatively, you can install NeSy4PPM directly from `PyPi <https://pypi.org/project/nesy4ppm/>`_.

For ProbDECLARE BK conformance checking, install the `Lydia <https://github.com/whitemech/lydia>`_ backend using Docker:

1. `Install Docker <https://www.docker.com/get-started>`_
2. Pull the Lydia image:

   .. code-block:: bash

      docker pull whitemech/lydia:latest


3. Make the Docker image executable under the name ``lydia``. On Linux and macOS machines, the following commands should work:

   .. code-block:: bash

      echo '#!/usr/bin/env sh' > lydia
      echo 'docker run -v$(pwd):/home/default whitemech/lydia lydia "$@"' >> lydia
      sudo chmod u+x lydia
      sudo mv lydia /usr/local/bin/


More information can be found at `Logaut repository <https://github.com/whitemech/logaut>`_.
