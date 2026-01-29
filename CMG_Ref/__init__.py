"""
CMG Reference Motion Generator Package

This package provides the Conditional Motion Generator (CMG) for generating
reference motions based on velocity commands.
"""

from .module.cmg import CMG
from .cmg_trainer import CMGTrainer

__all__ = ['CMG', 'CMGTrainer']
