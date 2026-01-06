"""
Synthetic Q&A generation from documents.

Generates realistic question-answer pairs for RAG evaluation
without requiring manual labeling.
"""

from autorag.synthetic.generator import SyntheticQAGenerator

__all__ = ["SyntheticQAGenerator"]
