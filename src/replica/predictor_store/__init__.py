"""Export, verification and publication of the predictor store.

The store is a Hugging Face dataset repository of long-format Parquet tables
built from compute pickles. :mod:`~replica.predictor_store.export` stages a
release from output directories, :mod:`~replica.predictor_store.verify` reads
the staged tree back against the pickles, and
:mod:`~replica.predictor_store.publish` uploads it as a pull request.
:mod:`~replica.predictor_store.layout` fixes the paths and schemas they share.
"""
