..  _embed:

Embed
==========

Embedding the time series is required in order to determine causation. I would suggest reading `this post by a professor at emory`_ to understand which lag and embedding dimension is appropriate.

As a quick recap, the lag is picked as the first minimum in the mutual information and the embedding dimension is picked using a false near neighbors test. In practice, however, it is acceptable to use the embedding that gives the highest forecast skill. As a starting point, an embedding dimension of 3 is a good value for 1D systems.

Lag Value
^^^^^^^^^

Mutual information is used as a way to jump far enough in time that new information about the system can be gained. A similar idea is calculating the autocorrelation. Systems that don't change much from one time step to the next would have higher autocorrelation and thus a larger lag value would be necessary to gain new information about the system. It turns out that using mutual information over autocorrelation allows for better predictions to be made [Chaos Paper Link].

::

  import skccm as ccm

  E = edm.Embed(X) #initiate the class

  max_lag = 100
  mi = E.mutual_information(max_lag)

.. image:: /_static/ccm/lorenz_mutual_info.png
   :align: center

The figure above shows the mutual information for the :math:`x` values of the lorenz time series. We can see a minimum around 16.

Embedding Dimension
^^^^^^^^^^^^^^^^^^^

Ideally you want to find the best embedding dimension for a specific time series. A good rule of thumb is to use an embedding dimension of three as your first shot. After the initial analysis, you can tweak this hyperparameter until you achieve the best prediction skill.

Alternatively, you can use a [false near neighbor][fnn] test when the reconstructed attractor is fully "unfolded". This functionality is not in skccm currently, but will be added in the future.


Examples
^^^^^^^^

An example of a 1D embedding is shown in the gif below. This is the same thing as rebuilding the attractor. It shows a lag of 2 and an embedding dimension of 3. Setting the problem up this way allows us to use powerful near neighbor libraries such as the one implemented in scikit-learn.

.. image:: /_static/ccm/embedding.gif
   :align: center


Using this package, this would be represented as:

::

  E = ccm.Embed(X)

  lag = 2
  embed = 3
  X,y = E.embed_vectors_1d(lag, emb)


More examples of 1d embeddings are shown below. E is the embedding dimension and L is the lag.

.. image:: /_static/ccm/embedding_examples.png
   :align: center


.. _this post by a professor at emory: http://www.physics.emory.edu/faculty/weeks//research/tseries3.html
