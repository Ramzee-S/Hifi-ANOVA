"""Per-stage training modules split out of the monolithic ``trainer.py``.

Each module here holds one stage of the staged fit as a mixin the
``HiFiANOVATrainer`` composes, keeping ``self`` semantics identical to the
pre-split methods (behavior-preserving decomposition).
"""
