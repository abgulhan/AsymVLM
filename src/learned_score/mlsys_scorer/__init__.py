"""Isolated, auditable scorer experiments for the MLSys revision.

Backbone loading is deliberately lazy so protocol checks and CLI help work
without GPU/Transformers dependencies.
"""

RETENTIONS = (0.5, 0.7, 0.9)
VARIANTS = ("original", "rank32", "rank32_coverage")
