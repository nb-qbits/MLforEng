# mlforeng/__init__.py

"""
MLforEng: minimal ML library for the workshop.

This package is environment-agnostic: it runs locally, in containers,
and on Kubernetes/OpenShift as-is.
"""

from . import config, data, models, train
