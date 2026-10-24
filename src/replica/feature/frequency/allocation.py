"""Deterministic allocation of pooled unigram draws across caller-defined strata."""

import hashlib
import json
from collections.abc import Mapping
from dataclasses import dataclass, field
from types import MappingProxyType

import numpy as np

from .strata import _validate_labels

# --- allocation ------------------------------------------------------------


@dataclass(frozen=True, slots=True)
class AllocationSpec:
    """Deterministic per-word split of ``k_eval`` draws across strata.

    Every eligible stratum gets ``k_floor``; its explicit home weights split
    ``k_home``. What remains is spread proportional to the exact
    per-stratum column mass under a per-stratum cap. The split is a function of
    the pool alone, so the estimator stays unbiased conditional on the pool.
    """

    strata: tuple[str, ...]
    k_eval: int
    k_floor: int = 0
    k_home: int = 0
    replicates: int = 1
    remainder_cap_factor: float = 1.0
    home_weights: Mapping[str, Mapping[str, float]] = field(default_factory=dict)

    def __post_init__(self) -> None:
        object.__setattr__(self, "strata", tuple(self.strata))
        _validate_labels(self.strata)
        for name in ("k_eval", "k_floor", "k_home", "replicates"):
            value = getattr(self, name)
            if not isinstance(value, (int, np.integer)):
                raise ValueError(f"{name} must be an integer")
            object.__setattr__(self, name, int(value))
        if self.k_eval < 1 or self.replicates < 1:
            raise ValueError("k_eval and replicates must be positive")
        if self.k_floor < 0 or self.k_home < 0:
            raise ValueError("k_floor and k_home must be non-negative")
        if not np.isfinite(self.remainder_cap_factor) or self.remainder_cap_factor < 0:
            raise ValueError("remainder_cap_factor must be finite and non-negative")
        weights = {}
        for word, row in self.home_weights.items():
            unknown = row.keys() - set(self.strata)
            if unknown:
                raise ValueError(f"unknown home strata for {word!r}: {sorted(unknown)}")
            copied = {label: float(weight) for label, weight in row.items()}
            if any(not np.isfinite(w) or w < 0 for w in copied.values()):
                raise ValueError("home weights must be finite and non-negative")
            weights[word] = MappingProxyType(copied)
        object.__setattr__(self, "home_weights", MappingProxyType(weights))

    def sha(self) -> str:
        """Digest of every field, so a run records the split it used."""
        payload = {
            "k_eval": self.k_eval,
            "k_floor": self.k_floor,
            "k_home": self.k_home,
            "strata": self.strata,
            "replicates": self.replicates,
            "remainder_cap_factor": self.remainder_cap_factor,
            "home_weights": {w: dict(d) for w, d in self.home_weights.items()},
        }
        blob = json.dumps(payload, sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()

    def home_strata(self, word: str) -> tuple[tuple[int, float], ...]:
        """Positive home weights in partition order; absent words fund none.

        Args:
            word: Surface whose explicit weights to look up.
        """
        weights = self.home_weights.get(word, {})
        return tuple(
            (i, weights[label])
            for i, label in enumerate(self.strata)
            if weights.get(label, 0.0) > 0.0
        )

    def budget(
        self,
        word: str,
        colsum_col: np.ndarray,
        capacity: "np.ndarray | None" = None,
    ) -> np.ndarray:
        """Draws per stratum for *word*, summing to ``k_eval`` exactly.

        A stratum with no column mass is ineligible and gets nothing; its share
        goes to the eligible ones. When no stratum has mass the result is all
        zeros and the word's estimate is its BOS term alone.

        ``capacity`` bounds how many draws each stratum's stored cells can
        serve. The leftover above the per-stratum caps is spread by column
        mass, so in a pool dominated by one script that stratum can be handed
        far more draws than its buffers hold; the draws move to strata with
        room instead. Any deterministic split keeps the estimator unbiased.

        Args:
            word: Surface being funded; its home strata take the ``k_home``
                share.
            colsum_col: Exact per-stratum column mass of the word's first token,
                shape ``(len(self.strata),)``. Every entry has to be finite.
            capacity: Draws each stratum's stored cells can serve, same shape.
                ``None`` leaves the split unbounded.

        Returns:
            Integer draws per stratum, summing to ``k_eval`` or, under a
            *capacity* too small for it, to the room the cells have.
        """
        colsum = np.asarray(colsum_col, dtype=np.float64)
        if colsum.shape != (len(self.strata),):
            raise ValueError(f"colsum_col must have shape ({len(self.strata)},)")
        if not bool(np.isfinite(colsum).all()):
            raise ValueError(f"colsum_col for {word!r} is not finite")
        if bool((colsum < 0).any()):
            raise ValueError("column mass must be non-negative")
        if capacity is not None:
            capacity = np.asarray(capacity)
            if capacity.shape != colsum.shape:
                raise ValueError(f"capacity must have shape {colsum.shape}")
            if not np.issubdtype(capacity.dtype, np.integer) or (capacity < 0).any():
                raise ValueError("capacity must contain non-negative integers")
        eligible = colsum > 0.0
        if not bool(eligible.any()):
            return np.zeros(len(self.strata), dtype=np.int64)
        if capacity is not None and (capacity[eligible] < self.replicates).any():
            raise ValueError(
                "capacity must fund every replicate of each eligible stratum"
            )
        base = np.where(eligible, float(self.k_floor), 0.0)
        home: np.ndarray = np.zeros(len(self.strata), dtype=np.float64)
        pairs = [(i, w) for i, w in self.home_strata(word) if eligible[i]]
        if pairs:
            total_weight = float(np.array([w for _, w in pairs]).sum())
            for i, w in pairs:
                home[i] = self.k_home * w / total_weight
        floors = self._floors(base + home, eligible)
        target = floors + self._remainder(colsum, eligible, floors)
        if capacity is not None:
            target = self._respect_capacity(target, capacity, eligible)
            # A word whose mass sits in strata with shallow cells cannot spend
            # the whole budget; drawing what the cells hold keeps the estimator
            # unbiased and only widens its interval.
            room = int(np.where(eligible, np.asarray(capacity), 0).sum())
            return _largest_remainder(
                target, min(self.k_eval, room), eligible, capacity
            )
        return _largest_remainder(target, self.k_eval, eligible)

    def _respect_capacity(
        self, target: np.ndarray, capacity: np.ndarray, eligible: np.ndarray
    ) -> np.ndarray:
        """Move draws off strata that ask for more than their cells hold."""
        cap = np.asarray(capacity, dtype=np.float64)
        if cap.shape != (len(self.strata),):
            raise ValueError(f"capacity must have shape ({len(self.strata)},)")
        out = np.asarray(target, dtype=np.float64).copy()
        for _ in range(len(self.strata)):
            over = eligible & (out > cap)
            if not bool(over.any()):
                break
            excess = float((out[over] - cap[over]).sum())
            out[over] = cap[over]
            room = eligible & (out < cap)
            if not bool(room.any()):
                break
            head = (cap - out) * room
            out += excess * head / float(head.sum())
        return out

    def _floors(self, floors: np.ndarray, eligible: np.ndarray) -> np.ndarray:
        """Per-stratum lower bounds, never below one draw per replicate.

        The replicates partition the frame, so a funded stratum that cannot
        reach every one of them would drop the mass of the cells left empty.
        Squeezing a budget shrinks only what sits above that minimum.
        """
        low = float(self.replicates)
        need = low * float(eligible.sum())
        if need > self.k_eval:
            raise ValueError(
                f"k_eval={self.k_eval} cannot fund {int(eligible.sum())} strata at "
                f"{self.replicates} replicates; raise k_eval or lower replicates"
            )
        held: np.ndarray = np.where(eligible, np.maximum(floors, low), 0.0)
        if float(held.sum()) <= self.k_eval:
            return held
        slack = np.maximum(held - low, 0.0)
        total = float(slack.sum())
        share = (self.k_eval - need) / total if total > 0.0 else 0.0
        return np.where(eligible, low + slack * share, 0.0)

    def _remainder(
        self, colsum: np.ndarray, eligible: np.ndarray, floors: np.ndarray
    ) -> np.ndarray:
        """Proportional split of what the floors leave, capped per stratum."""
        left = float(self.k_eval - floors.sum())
        extra: np.ndarray = np.zeros(len(self.strata), dtype=np.float64)
        if left <= 0.0:
            return extra
        cap = self.remainder_cap_factor * floors
        free = eligible.copy()
        for _ in range(len(self.strata)):
            weight = np.where(free, colsum, 0.0)
            total = float(weight.sum())
            if total <= 0.0:
                break
            share = left * weight / total
            over = free & (share > cap)
            if not bool(over.any()):
                extra += np.where(free, share, 0.0)
                left = 0.0
                break
            extra[over] = cap[over]
            left -= float(cap[over].sum())
            free &= ~over
        if left > 0.0:
            # Every eligible stratum sits at its cap, and leaving draws unspent
            # would silently shrink k_eval, so the rest goes by mass.
            weight = np.where(eligible, colsum, 0.0)
            extra += left * weight / float(weight.sum())
        return extra

    def replicate_budget(self, k_per_stratum: np.ndarray) -> np.ndarray:
        """Split each stratum's draws over the replicates, low index first.

        The replicates partition the frame, so a stratum that is funded at all
        has to fund every one of them. A cell left at zero draws contributes
        nothing while its share of the stratum's mass stays in the estimand,
        which is a downward bias no draw can correct.

        Args:
            k_per_stratum: Draws each stratum is funded for, shape
                ``(len(self.strata),)``, as :meth:`budget` returns them.

        Returns:
            Shape ``(len(self.strata), replicates)``; row sums equal the input.
        """
        k = np.asarray(k_per_stratum)
        if k.shape != (len(self.strata),):
            raise ValueError(f"k_per_stratum must have shape ({len(self.strata)},)")
        if not np.issubdtype(k.dtype, np.integer) or (k < 0).any():
            raise ValueError("k_per_stratum must contain non-negative integers")
        k = k.astype(np.int64)
        r = int(self.replicates)
        short = k[(k > 0) & (k < r)]
        if short.size:
            raise ValueError(
                f"k_eval={self.k_eval} funds a stratum with {int(short.min())} "
                f"draws across {r} replicates; raise k_eval or k_floor so every "
                "funded stratum gets at least one draw per replicate"
            )
        out = np.repeat((k // r)[:, None], r, axis=1)
        cols: np.ndarray = np.arange(r, dtype=np.int64)[None, :]
        out += (cols < (k % r)[:, None]).astype(np.int64)
        return out


def _largest_remainder(
    target: np.ndarray,
    total: int,
    eligible: np.ndarray,
    capacity: "np.ndarray | None" = None,
) -> np.ndarray:
    """Round *target* to integers summing to *total*, ties by stratum index.

    With *capacity* given, rounding up never pushes a stratum past what its
    stored cells can serve.
    """
    if not bool(np.isfinite(target).all()):
        raise ValueError("allocation target must be finite")
    out = np.floor(target).astype(np.int64)
    out[~eligible] = 0
    if capacity is not None:
        out = np.minimum(out, np.asarray(capacity, dtype=np.int64))
        out[~eligible] = 0
    short = total - int(out.sum())
    if short <= 0:
        return out
    frac = np.where(eligible, target - np.floor(target), -1.0)
    order = np.lexsort((np.arange(target.size), -frac))
    order = np.array([i for i in order if eligible[i]], dtype=np.int64)
    step = 0
    guard = 0
    while short > 0:
        idx = int(order[step % order.size])
        step += 1
        guard += 1
        if capacity is not None and out[idx] >= int(capacity[idx]):
            if guard > order.size * (total + 1):
                raise ValueError("capacity leaves the allocation short of k_eval")
            continue
        out[idx] += 1
        short -= 1
    return out
