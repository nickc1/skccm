Install
=======


pip
^^^
::

  pip install skccm


Conda (Recommended)
^^^^^^^^^^^^^^^^^^^

To create a conda environment, you can use the following conda environment.yml file::

  name: skccm_env
  dependencies:
    - python=3
    - numpy
    - scikit-learn
    - scipy
    - pip:
      - skccm

Then you can simply create the environment with::

  conda env create -f environment.yml

And activate it with::

  source activate skccm_env

Contribute, Report Issues, Support
^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^

To contribute, we suggest making a `pull request`_.

To report issues, we suggest `opening an issue`_.

For support, email cortalen@uncw.edu.



.. _github: https://github.com/NickC1/skccm
.. _pull request: https://github.com/NickC1/skccm/pulls
.. _opening an issue: https://github.com/NickC1/skccm/issues
