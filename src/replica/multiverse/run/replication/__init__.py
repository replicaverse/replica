"""Execute caller-declared replication cells and nested contrasts.

``contracts`` validates each declaration, ``runner`` handles persistence and
failures, ``contrast`` evaluates configured focal predictors, and ``sources``
loads local or published predictor outputs. Campaign discovery, corpus lists,
predictor definitions, and analysis presets belong to the caller.
"""
