"""
Grid Infrastructure Detection Dataset Builder

A modular pipeline for generating training data for electrical substation
detection from NAIP aerial imagery, using OpenStreetMap labels.

Supports two sampling strategies (curated counties vs. randomized US-wide),
per-substation-type budgets, and geographic train/val/test splitting.
"""

__version__ = "0.2.0"
__author__ = "Nishan Sah, Arthur Spirou, Jordan Gutterman, Arturo Arias"
