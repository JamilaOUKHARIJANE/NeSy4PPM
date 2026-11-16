AdaptiveNeSy4PPM
================

AdaptiveNeSy4PPM combines a neural next-activity predictor with process
background knowledge (BK).  During suffix prediction, the system can either
use a fixed BK weight or compute a new weight for every decoded step.  The
adaptive mode is designed to give more influence to the source that is more
reliable for the current prefix.

The implementation supports activity-only, beam search, and several BK representations, including Declare and Petri-net/procedural models.

Adaptive weighting is controlled by boolean options passed to
``predict_evaluate``.  The available signals are:

``nn_entropy``
    Use ``1 - U_NN`` as a neural reliability factor, where ``U_NN`` is the
    normalized entropy of the neural probability distribution.  A confident
    neural distribution has a larger factor.

``bk_entropy``
    Use ``1 - U_BK`` as a BK reliability factor.

``prefix_support``
    Use the normalized frequency of the current prefix in the training log.
    An unseen prefix has support ``0``.

``prefix_sim``
    Use the best normalized Damerau-Levenshtein similarity between the current
    prefix and a prefix observed in the training data.

``bk_conformance``
    Use the current prefix's BK alignment/conformance score when the selected
    BK model provides one.  The exact score depends on the BK type: for
    example, it can be Declare compliance or Petri-net token-replay fitness.

An adaptive run that enables all available signal families can therefore be
configured with ``nn_entropy=True``, ``bk_entropy=True``,
``prefix_support=True``, ``prefix_sim=True``, and ``bk_conformance=True``.
The flags also make it possible to run signal ablations.