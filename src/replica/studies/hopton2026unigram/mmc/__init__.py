"""Sharded producer for marginal Monte Carlo unigram-surprisal pickles.

``p(w)`` depends only on the word type, the LM and the sample pool, so the unit
of work is a word type and every dataset containing it reuses one estimate.

Three sources share the stages. ``--source mmc`` sums every boundary of a
small pool. The two pooled sources sweep a large pool once and draw each word's
boundaries by priority sampling, which adds the pool, index and sweep stages
between ``samples`` and ``shard``; they differ in the numerator. ``mmc_pool``
pairs the canonical sum against the take-store estimate of itself and adds the
exact text tally (``counts``, ``takes``). ``mmc_pool_ee`` charges the residual
count directly and prices the continuation against the sampler's horizon
(``splits``, ``--cap``). Each publishes under its own tag family.

Stages::

    samples      draw one sample shard and write it to disk
    pool         merge sample shards into a token store plus a manifest
    words        collect the union of surfaces over datasets
    index        tokenize one shard and label its boundaries
    index-merge  tile the shard indices and collapse duplicate prefixes
    plan         split the union into equal-cost slices and size the buffers
    counts       tally one shard's occurrences of each surface on the text
    counts-merge add the shard tallies
    splits       split one shard's occurrences of each surface by spelling
    splits-merge add the shard splits and the pool's unit total
    sweep        score one shard's boundaries against the first-token columns
    sweep-merge  fold the shard sweeps into one pooled selection
    takes        enumerate the boundaries whose successor opens a tracked word
    shard        estimate one slice of the union (the parallel stage)
    merge        combine shard tables into one type -> p table per LM
    audit        re-estimate a fraction of the union under a different layout
    materialize  write each dataset's pickle as a lookup against the table

Every stage except ``materialize`` writes its output through a temporary
file; ``materialize`` writes each pickle in place. The array stages skip work
whose output is already on disk, so a partially completed array can be
resubmitted as is; ``merge`` and ``materialize`` always rebuild from their
inputs. The pickles ``materialize`` writes carry no provenance manifest, so
``replica predictors export`` reports each as a discovery problem and does not
stage it.
"""

from .cli import build_parser, main

__all__ = ["build_parser", "main"]
