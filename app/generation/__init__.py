"""Generation: prompt construction, streaming assembly, citation validation.

This package turns retrieved passages into a grounded, cited answer
(architecture.md 3.4, 3.5). It is deliberately separate from `app.api`: the
prompt, the sentence splitter, and the validator are pure functions with no
FastAPI or database dependency, so they can be tested and reasoned about without
a running app.
"""
