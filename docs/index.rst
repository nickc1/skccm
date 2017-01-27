.. skccm documentation master file, created by
   sphinx-quickstart on Thu Jan  5 14:38:25 2017.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

skccm
=================================


|License Type|

**Scikit Convergent Cross Mapping**

Scikit Convergent Cross Mapping (skccm) can be used as a way to detect causality
between time series.

For a quick explanation of this package, I suggest checking out the :ref:`example`
section as well as the wikipedia article on `convergent cross mapping`_ . Additionally,
`Dr. Sugihara's lab`_ has produced some good summary videos about the topic:

1. `Time Series and Dynamic Manifolds`_
2. `Reconstructed Shadow Manifold`_
3. `State Space Reconstruction: Convergent Cross Mapping`_


For a more complete background, I suggest checking out the following papers:

1. `Detecting Causality in Complex Ecosystems by Sugihara`_
2. `Distinguishing time-delayed causal interactions using convergent cross mapping by Ye`_

Sugihara also have a good `talk about about Correlation and Causation`_.




.. toctree::
   :maxdepth: 2
   :caption: Contents:

   install
   quick-example
   generate-data
   embed
   predict
   score
   module-reference
   acknowledgements


.. _convergent cross mapping: https://www.wikiwand.com/en/Convergent_cross_mapping
.. _Detecting Causality in Complex Ecosystems by Sugihara: http://science.sciencemag.org/content/338/6106/496
.. _Distinguishing time-delayed causal interactions using convergent cross mapping by Ye: https://www.ncbi.nlm.nih.gov/pmc/articles/PMC4592974/

.. _Scikit Emperical Dynamic Modeling: https://github.com/NickC1/skedm
.. _Nonlinear Analysis by Kantz: https://www.amazon.com/Nonlinear-Time-Analysis-Holger-Kantz/dp/0521529026/ref=sr_1_1?s=books&ie=UTF8&qid=1475599671&sr=1-1&keywords=nonlinear+time+series+analysis

.. _Practical implementation of nonlinear time series methods\: The TISEAN package : http://scitation.aip.org/content/aip/journal/chaos/9/2/10.1063/1.166424

.. _example: http://skedm.readthedocs.io/en/latest/example.html
.. _nonlinear analysis: https://www.wikiwand.com/en/Nonlinear_functional_analysis

.. _dr. sugihara's lab: http://deepeco.ucsd.edu/

.. _Time Series and Dynamic Manifolds: https://www.youtube.com/watch?v=fevurdpiRYg

.. _Reconstructed Shadow Manifold: https://www.youtube.com/watch?v=rs3gYeZeJcw
.. _State Space Reconstruction\: Convergent Cross Mapping: https://youtu.be/iSttQwb-_5Y

.. _phase spaces: https://github.com/ericholscher/reStructuredText-Philosophy

.. _talk about about Correlation and Causation: https://youtu.be/uhONGgfx8Do

.. |License Type| image:: https://img.shields.io/github/license/mashape/apistatus.svg
    :target: https://github.com/NickC1/skedm/blob/master/LICENSE
.. |Travis CI| image:: https://travis-ci.org/NickC1/skedm.svg?branch=master
    :target: https://travis-ci.org/NickC1/skedm
