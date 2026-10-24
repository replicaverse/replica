"""Named experimental factors and their distinct modeling and sampling roles."""

import json
import re
import warnings
from dataclasses import asdict, dataclass, replace

import numpy as np


@dataclass(frozen=True)
class Factor:
    """A categorical factor extracted at the analyzed observation's granularity.

    Args:
        name: Name used in random effects, categorical controls and sampling units.
        source: ``participant``, ``stimulus``, ``occurrence``, ``word``, or a metadata source
            ``stimulus:key``, ``datum:key`` or ``unit:key``. Unit metadata
            belongs to the analyzed annotation layer and must align with it.
            ``occurrence`` identifies a position in a stimulus and layer,
            shared by all participants who encounter that position.
        role: ``experimental`` permits generalization and resampling by this
            factor; ``model`` supplies model terms only, such as word type.
    """

    name: str
    source: str
    role: str = "experimental"

    def __post_init__(self):
        if not re.fullmatch(r"[A-Za-z][A-Za-z0-9_]*", self.name):
            raise ValueError(f"Invalid factor name: {self.name!r}")
        if self.role not in {"experimental", "model"}:
            raise ValueError("Factor role must be 'experimental' or 'model'")
        prefix, separator, key = self.source.partition(":")
        if self.source not in {
            "participant",
            "stimulus",
            "occurrence",
            "word",
        } and not (separator and key and prefix in {"stimulus", "datum", "unit"}):
            raise ValueError(f"Unknown factor source: {self.source!r}")

    def values(self, datum, layer: str, indices: np.ndarray) -> np.ndarray:
        """Extract raw labels; preparation validates them after row filtering.

        Args:
            datum: Response carrying participant and stimulus metadata.
            layer: Analyzed annotation layer, after any segmentation.
            indices: Selected positions within that layer.
        """
        return self.values_from(
            stimulus=datum.stimulus,
            metadata=datum.metadata,
            participant_id=datum.participant_id,
            layer=layer,
            indices=indices,
        )

    def values_from(self, *, stimulus, metadata, participant_id, layer, indices):
        """Extract a factor from observation coordinates and source metadata.

        Args:
            stimulus: Material owning the annotation units.
            metadata: Observation-level source metadata.
            participant_id: Observed participant identity, or None when absent.
            layer: Analyzed annotation layer.
            indices: Native unit positions selected in this block.
        """
        if self.source == "participant" and participant_id is None:
            raise ValueError("Participant factor is absent from these observations")
        if self.source == "participant":
            value = participant_id
        elif self.source == "stimulus":
            value = stimulus.stimulus_id
        elif self.source == "occurrence":
            value = np.asarray(
                [
                    json.dumps([str(stimulus.stimulus_id), layer, int(i)])
                    for i in indices
                ],
                dtype=object,
            )
        elif self.source == "word":
            if layer != "words":
                raise ValueError("A word-type factor requires the words analysis layer")
            value = np.asarray(stimulus.unit_strings(layer), dtype=object)[indices]
        else:
            prefix, key = self.source.split(":", 1)
            metadata = {
                "stimulus": stimulus.metadata,
                "datum": metadata,
                "unit": stimulus.layers[layer].metadata,
            }[prefix]
            if key not in metadata:
                raise ValueError(f"Missing {self.source!r} for factor {self.name!r}")
            value = metadata[key]
            if np.ma.is_masked(value):
                value = np.ma.asarray(value, dtype=object).filled(np.nan)
            if prefix == "unit":
                value = np.asarray(value, dtype=object)
                if value.shape != (len(stimulus.layers[layer]),):
                    raise ValueError(
                        f"Factor {self.name!r} is not aligned with {layer}"
                    )
                value = value[indices]
            elif np.ndim(value) != 0:
                raise ValueError(f"Factor source {self.source!r} must be scalar")
        labels = np.broadcast_to(np.asarray(value, dtype=object), (len(indices),))
        return labels


@dataclass(frozen=True)
class RandomEffect:
    """An intercept and named numeric slopes sharing one categorical factor.

    Args:
        factor: Declared factor name.
        slopes: Numeric predictor names. ``Design.comparison`` determines
            whether a slope remains when its fixed effect is absent.
        intercept: Include a random intercept.
        correlated: Estimate correlations between intercept and slopes.
    """

    factor: str
    slopes: tuple[str, ...] = ()
    intercept: bool = True
    correlated: bool = True

    def __post_init__(self):
        if isinstance(self.slopes, str):
            raise ValueError(
                "slopes must be a sequence of predictor names, not a string"
            )
        object.__setattr__(self, "slopes", tuple(self.slopes))
        if not self.intercept and not self.slopes:
            raise ValueError("A random effect requires an intercept or slope")
        if len(set(self.slopes)) != len(self.slopes):
            raise ValueError("Random slopes must be unique")


@dataclass(frozen=True)
class Design:
    """Declare factors once and select their modeling and sampling roles.

    Args:
        factors: Sources of named categorical labels. Defaults retain the
            participant and stimulus identities, plus occurrence IDs shared across
            readers at the same stimulus position.
        random_effects: Candidate random terms, resolved for each model using
            ``comparison`` and that model's fixed predictors. None selects
            participant intercepts and focal-predictor slopes, with correlations
            fixed to zero, plus occurrence intercepts. An explicit tuple replaces
            this structure; an empty tuple declares no random effects.
        categorical: Factor names included as fixed controls in both models.
        condition_on: Factors whose learned random effects enter prediction.
            Empty predicts with zero expected random effects.
            Evaluation warns when modeled readers are shared across training
            and test but their learned effects are omitted from prediction.
        unseen: ``zero`` gives an unseen random level zero effect; ``error``
            refuses unseen levels among the factors conditioned on.
        cv_unit: Experimental factor, or factors jointly held out by the default
            splitter. Defaults to unseen participants and stimuli. An explicit
            single factor retains the corresponding one-axis generalization.
        permutation_unit: Experimental factor whose paired loss sums are
            flipped together. None leaves the significance test undeclared.
        comparison: ``predictor`` removes absent fixed predictors from random
            slopes too; ``fixed_effect`` retains all declared random slopes.
            None defaults to ``predictor`` and warns on resolution, even when
            no slopes are removed. Choose either mode explicitly to acknowledge
            the comparison and silence that warning.
        scoring: ``marginal`` scores Gaussian predictions with residual variance
            plus variance from entirely unseen random-effect levels. ``residual``
            retains the earlier residual-only plug-in score, including for
            predictions conditioned on known readers.
    """

    factors: tuple[Factor, ...] = (
        Factor("participant", "participant"),
        Factor("stimulus", "stimulus"),
        Factor("item", "occurrence", role="model"),
    )
    random_effects: tuple[RandomEffect, ...] | None = None
    categorical: tuple[str, ...] = ()
    condition_on: tuple[str, ...] = ()
    unseen: str = "error"
    cv_unit: str | tuple[str, ...] = ("participant", "stimulus")
    permutation_unit: str | None = None
    comparison: str | None = None
    scoring: str = "marginal"

    def __post_init__(self):
        if not isinstance(self.cv_unit, str):
            object.__setattr__(self, "cv_unit", tuple(self.cv_unit))
        if not self.cv_units or len(set(self.cv_units)) != len(self.cv_units):
            raise ValueError("CV factors must be nonempty and unique")
        self._check_policy_choices()
        for key in ("factors", "random_effects", "categorical", "condition_on"):
            if key == "random_effects" and self.random_effects is None:
                continue
            if isinstance(getattr(self, key), str):
                raise ValueError(f"{key} must be a sequence, not a string")
            object.__setattr__(self, key, tuple(getattr(self, key)))
        for key in ("categorical", "condition_on"):
            if len(set(getattr(self, key))) != len(getattr(self, key)):
                raise ValueError(f"{key} entries must be unique")
        factors = {f.name: f for f in self.factors}
        if len(factors) != len(self.factors):
            raise ValueError("Factor names must be unique")
        random = {r.factor for r in self.random_effects or ()}
        if len(random) != len(self.random_effects or ()):
            raise ValueError(
                "Combine random terms for each factor into one RandomEffect"
            )
        requested = (
            random | set(self.categorical) | set(self.condition_on) | set(self.cv_units)
        )
        if self.permutation_unit is not None:
            requested.add(self.permutation_unit)
        if requested - factors.keys():
            raise ValueError(
                f"Undeclared factors: {sorted(requested - factors.keys())}"
            )
        if self.random_effects is not None and set(self.condition_on) - random:
            raise ValueError(
                "Prediction conditioning requires a declared random effect"
            )
        for unit in (*self.cv_units, self.permutation_unit):
            if unit is not None and factors[unit].role != "experimental":
                raise ValueError(
                    f"Model-only factor {unit!r} cannot be a sampling unit"
                )

    def _check_policy_choices(self):
        if self.comparison not in {None, "predictor", "fixed_effect"}:
            raise ValueError("comparison must be 'predictor', 'fixed_effect', or None")
        if self.scoring not in {"marginal", "residual"}:
            raise ValueError("scoring must be 'marginal' or 'residual'")
        if self.unseen not in {"zero", "error"}:
            raise ValueError("unseen must be 'zero' or 'error'")

    @property
    def cv_units(self) -> tuple[str, ...]:
        """Return held-out factor names without changing single-factor metadata."""
        return (self.cv_unit,) if isinstance(self.cv_unit, str) else self.cv_unit

    def resolve(self, predictors: tuple[str, ...]) -> "Design":
        """Resolve the default random structure from the focal predictor family.

        Args:
            predictors: Names of all focal columns, including declared lags.
                Baseline controls do not acquire default random slopes.
        """
        if self.random_effects is not None:
            return self
        if isinstance(predictors, str) or len(set(predictors)) != len(predictors):
            raise ValueError("predictors must be distinct focal column names")
        groups = {}
        for source in ("participant", "occurrence"):
            matches = [
                factor.name for factor in self.factors if factor.source == source
            ]
            if len(matches) != 1:
                raise ValueError(
                    f"Default random effects require exactly one {source!r} factor; "
                    "declare the factors or supply random_effects explicitly"
                )
            groups[source] = matches[0]
        return replace(
            self,
            random_effects=(
                RandomEffect(groups["participant"], predictors, correlated=False),
                RandomEffect(groups["occurrence"]),
            ),
        )

    def for_model(self, fixed: tuple[str, ...], *, warn: bool = True) -> "Design":
        """Resolve random terms for one model's fixed predictor set.

        Intercepts, factors and sampling units stay declared. A slope-only
        random term disappears if none of its slopes remain, and prediction
        conditioning follows the retained random terms.

        Args:
            fixed: Names of the numeric fixed predictors included in this model.
            warn: Warn about an unspecified comparison. A comparison driver
                may disable this after warning once during preflight.
        """
        if isinstance(fixed, str):
            raise ValueError("fixed must be a sequence of predictor names")
        if self.random_effects is None:
            raise ValueError(
                "Resolve the focal predictors before selecting model terms"
            )
        if self.comparison is None:
            if warn:
                warnings.warn(
                    "Design.comparison was not set; defaulting to 'predictor' "
                    "(complete predictor removal). Review baseline and target "
                    "fixed effects, random effects, and the inference method. "
                    "Set comparison='predictor' to confirm removal, or "
                    "comparison='fixed_effect' to retain shared random slopes.",
                    UserWarning,
                    stacklevel=2,
                )
            return replace(self, comparison="predictor").for_model(fixed)
        omitted = {s for r in self.random_effects for s in r.slopes} - set(fixed)
        if not omitted or self.comparison == "fixed_effect":
            return self
        effects = []
        for effect in self.random_effects:
            slopes = tuple(s for s in effect.slopes if s in fixed)
            if effect.intercept or slopes:
                effects.append(replace(effect, slopes=slopes))
        retained = {r.factor for r in effects}
        return replace(
            self,
            random_effects=tuple(effects),
            condition_on=tuple(f for f in self.condition_on if f in retained),
        )

    def metadata(self) -> dict:
        """Return a serializable declaration for recorded analysis provenance."""
        return {**asdict(self), "effective_comparison": self.comparison or "predictor"}

    def check_splits(self, splits, factors: dict[str, np.ndarray]) -> list[str]:
        """Warn when an explicit splitter divides the declared held-out units.

        Args:
            splits: Boolean training and test masks in analyzed row order.
            factors: Aligned factor labels.
        """
        messages = []
        for unit in self.cv_units:
            labels = factors[unit]
            for fold, (train, test) in enumerate(splits):
                if np.intersect1d(labels[train], labels[test]).size:
                    messages.append(f"Fold {fold} shares declared CV unit {unit!r}")
        if messages:
            warnings.warn("; ".join(messages), UserWarning, stacklevel=2)
        return messages

    def check_conditioning(self, splits, factors: dict[str, np.ndarray]) -> list[str]:
        """Warn when available reader effects are omitted from predictions.

        This checks actual reader overlap rather than treating every non-CV
        factor as known. It preserves deliberate zero-effect predictions.

        Args:
            splits: Boolean training and test masks in analyzed row order.
            factors: Aligned factor labels, after observation filtering.
        """
        random = {effect.factor for effect in self.random_effects or ()}
        messages = []
        for factor in self.factors:
            if (
                factor.source != "participant"
                or factor.name not in random
                or factor.name in self.condition_on
            ):
                continue
            labels = factors[factor.name]
            if any(
                np.intersect1d(labels[train], labels[test]).size
                for train, test in splits
            ):
                messages.append(
                    f"Test readers in factor {factor.name!r} also occur in training, "
                    "but their modeled random effects are not in condition_on. "
                    f"Add {factor.name!r} to condition_on to use learned reader "
                    "effects, or retain and report intentional zero-effect "
                    "predictions. Other random factors may still be unseen."
                )
        if messages:
            warnings.warn("; ".join(messages), UserWarning, stacklevel=2)
        return messages
