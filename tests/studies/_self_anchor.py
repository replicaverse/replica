"""Regeneration affordance for REPLICA self-anchored (drift-guard) references.

Set ``PSYCHO_REGEN_SELF_ANCHOR=1`` and run the test file with ``-s`` to print
each freshly computed reference as a copy-pasteable literal before the
assertions run. Assertions always run, so refreshing a frozen value takes a
reviewed edit.
"""

import os
import pprint


def maybe_print_reference(label: str, key, computed: dict) -> None:
    """Print *computed* as a ``REFERENCE`` literal when regen mode is on."""
    if not os.environ.get("PSYCHO_REGEN_SELF_ANCHOR"):
        return
    print(f"\n[self-anchor regen] {label} REFERENCE[{key!r}] =")
    pprint.pprint(computed, sort_dicts=False)
