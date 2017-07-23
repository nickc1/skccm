..  _embed:

Embed
==========

Before performing convergent cross mapping, both time series must be appropriately embed. As a quick recap, the lag is picked as the first minimum in the mutual information and the embedding dimension is picked using a false near neighbors test. In practice, however, it is acceptable to use the embedding that gives the highest forecast skill.

Lag Value
^^^^^^^^^

In attempting to use the time series to reconstruct the state space behavior of a complete system, a lag is needed to form the embedding vector. This lag is most commonly found from the first minimum in the mutual information between the time series and a shifted version of itself. The first minimum in the mutual information can be thought of as jumping far enough away in the time series that new information is gained. A more inituitive but less commonly used procedure to finding the lag is using the first minimum in the autocorrelation. The mutual information calculation can be done using the embed class provided by skccm.

::

  from skccm import Embed
  lag = 1
  embed = 2
  e1 = Embed(x1)
  e2 = Embed(x2)
  X1 = e1.embed_vectors_1d(lag,embed)
  X2 = e2.embed_vectors_1d(lag,embed)

.. image:: /_static/ccm/lorenz_mutual_info.png
   :align: center

The figure above shows the mutual information for the :math:`x` values of the lorenz time series. We can see a minimum around 16. If we were interested in reconstructing the state space behavior we would use a lag of 16.

Embedding Dimension
^^^^^^^^^^^^^^^^^^^

Traditionally, the embedding dimension is chosen using a `false near neighbor test`_. This checks to see when the reconstructed attractor is fully "unfolded". This functionality is not in skccm currently, but will be added in the future. In practice, the embedding dimension that gives the highest forecast skill is chosen. The false near neighbor test can be noisy for real world systems.


Examples
^^^^^^^^

An example of a 1D embedding is shown in the gif below. This is the same thing as rebuilding the attractor. It shows a lag of 2 and an embedding dimension of 3. Setting the problem up this way allows us to use powerful near neighbor libraries such as the one implemented in scikit-learn.

.. image:: /_static/ccm/embedding.gif
   :align: center


Using this package, this would be represented as:

::

  from skccm import Embed
  lag = 1
  embed = 2
  e1 = Embed(x1)
  e2 = Embed(x2)
  X1 = e1.embed_vectors_1d(lag,embed)
  X2 = e2.embed_vectors_1d(lag,embed)


More examples of 1d embeddings are shown below. L is the lag and E is the embedding dimension.

.. image:: /_static/ccm/embedding_examples.png
   :align: center

.. _false near neighbor test: https://www.wikiwand.com/en/False_nearest_neighbor_algorithm
