.. NeSy4PPM documentation master file, created by
   sphinx-quickstart on Sat Jun 28 16:00:57 2025.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

NeSy4PPM documentation
======================

NeSy4PPM is the first Python package designed for both single-attribute (e.g., activity) and multi-attribute (e.g., activity and resource) suffix prediction in predictive process monitoring. It implements a Neuro-Symbolic (NeSy) system that integrates neural models with various types of symbolic background knowledge (BK), enabling accurate and compliant predictions even under concept drift.

NeSy4PPM offers the following key features:

1. **Symbolic knowledge decoding-time integration**: contextualizes neural predictions with declarative or procedural knowledge, including DECLARE, MP-DECLARE (multi-perspective DECLARE), ProbDECLARE (probabilistic DECLARE), and Petri nets.

2. **Flexible learning**: provides multiple prefix encoding methods and supports LSTM (Long Short-Term Memory) and Transformer architectures.

3. **SDFA-constrained decoding**: combines neural next-event probabilities with a stochastic deterministic finite automaton (SDFA) during beam search, with optional pruning of infeasible transitions and suffixes that cannot terminate within the remaining prediction horizon.

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   modules.rst
   installation.rst
   tutorials.rst
   structure.rst
   publications.rst
   citing.rst