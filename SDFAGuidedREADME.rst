Neuro-Symbolic Suffix Prediction Guided by Stochastic Automata at Decoding Time
====================



NeSy4PPM combines neural next-activity probabilities with a Stochastic
Deterministic Finite Automaton (SDFA) during suffix prediction.  The SDFA
provides stochastic process knowledge that guides beam search toward
admissible continuations while preserving the neural model's predictions.

An SDFA can be built from declarative constraints or from a procedural event
log.  The resulting ``.sdfa`` model is loaded as background knowledge with
``load_bk`` and passed to ``predict_evaluate``.

The complete workflow, including SDFA construction, training, prediction, and
evaluation, is available in the
`Stochastic guided suffix prediction tutorial
<docs/source/tutorials/Stochastic_guided_suffix_prediction.ipynb>`_.

Decoding configuration
----------------------

SDFA-guided decoding is controlled by the following arguments passed to
``predict_evaluate``:

``bk_model``
    The SDFA loaded with ``load_bk``.  If it is ``None``, the decoder uses the
    neural model without SDFA guidance.

``beam_size``
    The number of alternative suffixes retained during beam search.  A value
    of ``1`` performs greedy decoding.

``weight``
    A value in ``[0, 1]`` that balances the neural and SDFA scores.  A value
    of ``0`` uses the neural score only, while larger values give more weight
    to the SDFA transition probabilities.

``feasibility_pruning``
    Remove candidate activities that have no admissible SDFA transition or a
    zero SDFA transition probability.

``termination_pruning``
    Remove candidate activities whose SDFA state cannot reach an allowed
    termination within the remaining prediction horizon.

``bk_end``
    Apply SDFA conformance after decoding rather than using SDFA guidance
    during the search.  Completed suffixes are checked for admissible
    transitions and termination.

Example
-------

The following configuration uses soft SDFA guidance together with feasibility
and termination pruning:

.. code-block:: python

    from pathlib import Path

    from NeSy4PPM.Data_preprocessing.utils import load_bk
    from NeSy4PPM.Prediction import predict_suffix

    sdfa_model = load_bk(Path("path/to/model.sdfa"))
    prediction_filename = "helpdesk_SDFA_guided_prediction.csv"
    beam_size = len(log_data.act_to_int) + 1

    predict_suffix.predict_evaluate(
        log_data,
        max_trace_len,
        models_folder=model_folder,
        encoder=encoder,
        output_filename=prediction_filename,
        bk_model=sdfa_model,
        beam_size=beam_size,
        weight=0.5,
        bk_end=False,
        feasibility_pruning=True,
        termination_pruning=True,
    )

The same decoder can be used for the main comparison modes:

``SDFA_guided``
    Use ``weight=0.5`` with ``feasibility_pruning=True`` and
    ``termination_pruning=True``.

``SDFA_nopruning``
    Use ``weight=0.5`` with both pruning options set to ``False``.

``neural``
    Set ``bk_model=None`` and ``weight=0.0`` for neural-only decoding.

``SDFA_postdecoding``
    Set ``weight=0.0`` and ``bk_end=True`` to check the completed prediction
    against the SDFA after decoding.

Evaluation Results
------------------

The evaluation reports prediction time, Damerau-Levenshtein similarity, SDFA
feasibility, SDFA termination, and the normalized explored branch count
(``nEBC``).

The supplementary experimental results are available in the
`SDFA_guided_NeSy_Supplement`_ file.

.. _SDFA_guided_NeSy_Supplement:
   https://github.com/JamilaOUKHARIJANE/NeSy4PPM/blob/master/ICPM_paper_supplements.pdf
