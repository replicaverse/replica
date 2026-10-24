"""CDR through evaluate(): the event channel, end to end.

A continuous-time model cannot be scored on the row-aligned design matrix, so
``evaluate()`` builds a second row space for models that declare
``needs_events``. These tests use a small generated corpus whose responses
carry a diffuse effect of one predictor, and check that the ordinary gym entry
point routes it to the native CDR and reports a held-out likelihood gain.
"""

from dataclasses import replace

import numpy as np
import pandas as pd
import pytest

from replica.analysis.cv import ItemSplitter, NoScaler
from replica.analysis.data.events import EventIndex, build_response_events, series_codes
from replica.analysis.evaluation.evaluate import evaluate
from replica.analysis.models.deconvolution.analysis import (
    CDRAnalysis,
    CDRFit,
    fold_group_codes,
)
from replica.analysis.specification.config import AnalysisConfig
from replica.analysis.specification.evaluation import EvaluationSpec
from replica.core import Datum
from replica.dataset.base import ReadingTimeDataset
from replica.feature.extractors import PrecomputedFeature


class _Corpus(ReadingTimeDataset):
    """Two participants reading two documents, with a word-onset timeline.

    Word onsets accumulate the previous word's reading time, which is how the
    self-paced-reading releases derive their timeline. The response to a word
    is a diffuse function of the focal predictor over the preceding words, so
    a convolutional model should beat a control-only baseline.
    """

    def __init__(self, n_words: int = 40, seed: int = 0):
        super().__init__()
        from replica.dataset.contracts import DatasetInfo

        rng = np.random.default_rng(seed)
        self.info = DatasetInfo(name="timed_corpus", language="en")
        self._predictors: dict[str, dict[str, np.ndarray]] = {}
        self._times: dict[tuple[str, str], np.ndarray] = {}

        for doc in ("d1", "d2"):
            words = [f"w{i}" for i in range(n_words)]
            stimulus = self._make_stimulus(words, f"trial_{doc}", {"docid": doc})
            focal = rng.standard_normal(n_words)
            control = rng.standard_normal(n_words)
            self._predictors[stimulus.stimulus_id] = {
                "focal": focal,
                "control": control,
            }
            for pid in ("p1", "p2"):
                decay = np.array([0.6**k for k in range(4)])
                diffuse = np.convolve(focal, decay)[:n_words]
                rt = 250.0 + 40.0 * diffuse + 12.0 * control
                rt += rng.normal(0, 4.0, n_words)
                self._data.append(
                    Datum(
                        stimulus=stimulus,
                        measure_name="spr_rt",
                        values=rt,
                        participant_id=pid,
                    )
                )
                onsets = np.concatenate([[0.0], np.cumsum(rt[:-1]) / 1000.0])
                self._times[(pid, stimulus.stimulus_id)] = onsets
        self.default_measure = "spr_rt"
        self._declared_measures = ("spr_rt",)

    def load(self, path: str) -> None:  # pragma: no cover - built in memory
        raise NotImplementedError

    def predictor_arrays(self) -> dict[str, dict[str, np.ndarray]]:
        return self._predictors


class _TimedCorpus(_Corpus):
    """The same corpus, exposing its word-onset timeline."""

    def event_index(self, measure: str, *, layer: str = "words") -> EventIndex:
        participants, stimuli, units, times = [], [], [], []
        for (pid, sid), onsets in sorted(self._times.items()):
            participants.extend([pid] * len(onsets))
            stimuli.extend([sid] * len(onsets))
            units.extend(range(len(onsets)))
            times.extend(onsets.tolist())
        participant_id = np.array(participants, dtype=object)
        stimulus_id = np.array(stimuli, dtype=object)
        series, _ = series_codes(participant_id, stimulus_id)
        return EventIndex(
            participant_id=participant_id,
            stimulus_id=stimulus_id,
            unit_index=np.array(units, dtype=int),
            time=np.array(times, dtype=float),
            series=series,
            layer=layer,
        )


def _spec(dataset: _Corpus) -> EvaluationSpec:
    from replica.feature import (
        FeatureModelOutput,
        ModelFeature,
        PrecomputedFeatureModel,
    )

    outputs = {}
    for sid, columns in dataset.predictor_arrays().items():
        n_units = len(next(iter(columns.values())))
        outputs[sid] = [
            FeatureModelOutput(
                values={name: float(values[i]) for name, values in columns.items()}
            )
            for i in range(n_units)
        ]
    model = PrecomputedFeatureModel("generated", outputs)
    return EvaluationSpec(
        predictors=[ModelFeature(model, PrecomputedFeature("focal", "focal"))],
        controls=[ModelFeature(model, PrecomputedFeature("control", "control"))],
        targets=[dataset.feature("spr_rt")],
    )


class TestEventChannel:
    def test_held_out_group_gets_fixed_unknown_code(self):
        """Test-only participants do not enlarge the trainable ranef table."""
        train, test = fold_group_codes(np.array([7, 3, 7, 9]), np.array([9, 11, 3, 11]))
        np.testing.assert_array_equal(train, [1, 0, 1, 2])
        np.testing.assert_array_equal(test, [2, 3, 0, 3])
        assert test.max() == np.unique(train).size

    def test_source_item_identity_can_differ_from_occurrence_index(self):
        """Repeated presentations remain separate events but share the item
        random-effect level named by the source formula."""

        class Timeline:
            def event_index(self, measure: str, *, layer: str = "words"):
                del measure, layer
                participant = np.array(["p1"] * 3 + ["p2"] * 3, dtype=object)
                stimulus = np.array(["d1"] * 6, dtype=object)
                series, _ = series_codes(participant, stimulus)
                item_id = np.empty(6, dtype=object)
                item_id[:] = [("d1", 1, 1), ("d1", 1, 1), ("d1", 1, 2)] * 2
                return EventIndex(
                    participant_id=participant,
                    stimulus_id=stimulus,
                    unit_index=np.tile(np.arange(3), 2),
                    time=np.tile(np.arange(3, dtype=float), 2),
                    series=series,
                    item_id=item_id,
                )

        events = build_response_events(
            dataset=Timeline(),
            dv_name="rt",
            rows={
                "participant_id": np.array(["p1"] * 3 + ["p2"] * 3),
                "stimulus_id": np.array(["d1"] * 6),
                "unit_index": np.tile(np.arange(3), 2),
            },
            feature_arrays={},
            control_names=(),
            focal_names=(),
        )
        np.testing.assert_array_equal(events.item, [0, 0, 1, 0, 0, 1])

    def test_the_analysis_layer_reaches_the_event_index(self):
        """A dataset is asked for the layer the rows are counted in, and an
        index counted in another layer is refused by name."""
        requested = []

        class Timeline:
            def event_index(self, measure: str, *, layer: str = "words"):
                del measure
                requested.append(layer)
                participant = np.array(["p1"] * 2, dtype=object)
                stimulus = np.array(["d1"] * 2, dtype=object)
                series, _ = series_codes(participant, stimulus)
                return EventIndex(
                    participant_id=participant,
                    stimulus_id=stimulus,
                    unit_index=np.arange(2),
                    time=np.arange(2, dtype=float),
                    series=series,
                )

        rows = {
            "participant_id": np.array(["p1"]),
            "stimulus_id": np.array(["d1"]),
            "unit_index": np.array([0]),
        }
        kwargs = dict(feature_arrays={}, control_names=(), focal_names=())
        build_response_events(
            dataset=Timeline(), dv_name="rt", rows=rows, layer="words", **kwargs
        )
        with pytest.raises(ValueError, match="'words'.*'clauses'"):
            build_response_events(
                dataset=Timeline(), dv_name="rt", rows=rows, layer="clauses", **kwargs
            )
        assert requested == ["words", "clauses"]

    def test_evaluate_asks_for_the_segmented_layer(self):
        from replica.analysis.models.fixed_effects.ols import OLS
        from replica.core import Span
        from replica.segmentation import Segmenter, WhitespaceSegmenter

        class _Pairs(Segmenter):
            def __init__(self):
                super().__init__("pairs")

            def segment(self, text):
                words = WhitespaceSegmenter()(text)
                return [
                    Span(words[i].start, words[i + 1].end)
                    for i in range(0, len(words), 2)
                ]

        requested = []

        class _Stop(Exception):
            pass

        class _SegmentedCorpus(_TimedCorpus):
            additive_measures = frozenset({"spr_rt"})

            def event_index(self, measure: str, *, layer: str = "words"):
                requested.append(layer)
                raise _Stop

        class _EventOLS(OLS):
            needs_events = True

        from replica.feature import UnitLength

        dataset = _SegmentedCorpus()
        spec = EvaluationSpec(
            predictors=[UnitLength()],
            targets=[dataset.feature("spr_rt")],
            segmentation=_Pairs(),
        )
        with pytest.raises(_Stop):
            evaluate(
                dataset,
                spec,
                AnalysisConfig(
                    design=None,
                    transform_dv=np.log2,
                    model=_EventOLS(),
                    cv=ItemSplitter(2),
                    scaler=NoScaler(),
                ),
            )
        assert requested == ["pairs"]

    def test_event_channel_is_only_built_on_request(self):
        """A model that does not ask for events never reaches the timeline."""
        from replica.analysis.models.fixed_effects.ols import OLS
        from replica.feature import named

        requested = []

        class _CountingCorpus(_TimedCorpus):
            def event_index(self, measure: str, *, layer: str = "words"):
                requested.append(measure)
                return super().event_index(measure, layer=layer)

        class _EventOLS(OLS):
            needs_events = True

        dataset = _CountingCorpus()
        config = AnalysisConfig(
            design=None,
            transform_dv=np.log2,
            model=OLS(),
            cv=ItemSplitter(2),
            scaler=NoScaler(),
        )
        results = evaluate(dataset, _spec(dataset), config)
        assert len(results) == 1
        assert np.isfinite(results[0].delta_ll)
        assert requested == []
        event_config = replace(config, model=_EventOLS())
        spec = replace(
            _spec(dataset), targets=[named(dataset.feature("spr_rt"), "duration")]
        )
        evaluate(dataset, spec, event_config)
        assert requested == ["spr_rt"]

        foreign = _CountingCorpus(seed=1)
        for source in (dataset, foreign):
            source.info = replace(
                source.info, material_namespace="timed", material_fields=["docid"]
            )
        assert not np.array_equal(
            next(iter(dataset._times.values())), next(iter(foreign._times.values()))
        )
        spec = replace(spec, targets=[named(foreign.feature("spr_rt"), "duration")])
        with pytest.raises(ValueError, match="different dataset event timeline"):
            evaluate(dataset, spec, event_config)
        assert requested == ["spr_rt"]

    def test_cdr_runs_through_evaluate_and_finds_the_diffuse_effect(self, tmp_path):
        dataset = _TimedCorpus()
        model = CDRAnalysis(
            kernel_family="Exp",
            history_length=8,
            artifact_dir=str(tmp_path),
            fit=CDRFit(
                n_iter=150,
                learning_rate=0.05,
                minibatch_size=64,
                convergence_n_iterates=50,
                seed=0,
            ),
        )
        results = evaluate(
            dataset,
            _spec(dataset),
            AnalysisConfig(
                design=None,
                model=model,
                cv=ItemSplitter(2),
                scaler=NoScaler(),
                transform_dv=lambda x: x,
                per_fold=True,
                retain_per_obs_delta_ll=True,
            ),
        )
        assert len(results) == 1
        result = results[0]
        assert np.isfinite(result.delta_ll)
        assert result.delta_ll > 0, (
            "convolved focal predictor should improve held-out likelihood"
        )
        # A model that decomposes its likelihood per observation keeps the
        # per-datapoint significance test available.
        assert result.fold_results
        per_obs = result.fold_results[0].per_obs_delta_ll
        assert per_obs is not None
        np.testing.assert_allclose(
            float(np.sum(per_obs)), result.fold_results[0].delta_ll, rtol=1e-6
        )
        artifact = pd.read_csv(tmp_path / "full_test.tsv", sep="\t")
        assert "series" in artifact
        assert not artifact[["series", "time"]].duplicated().any()
        for side in ("baseline", "full"):
            artifact = tmp_path / f"{side}_test.tsv"
            checkpoint = tmp_path / f"{side}_checkpoint.npz"
            assert artifact.read_text().splitlines()[0].split("\t") == [
                "participant",
                "stimulus",
                "series",
                "time",
                "prediction",
                "squared_error",
                "loglik",
            ]
            with np.load(checkpoint) as saved:
                assert int(saved["n_iterations"]) > 0
                assert np.isfinite(float(saved["final_mean_loss"]))
                assert "estimate_intercept" in saved
                assert bool(saved["standardize_response"])

    def test_dataset_without_a_timeline_is_refused(self):
        """A continuous-time model needs timestamps, and says so."""
        dataset = _Corpus()
        with pytest.raises(TypeError, match="event timeline"):
            evaluate(
                dataset,
                _spec(dataset),
                AnalysisConfig(
                    design=None,
                    model=CDRAnalysis(),
                    cv=ItemSplitter(2),
                    transform_dv=lambda x: x,
                ),
            )

    def test_a_response_matching_two_events_is_refused(self):
        """A unit presented twice gives its response no single timestamp;
        repeated units that no analyzed row needs stay allowed."""

        class Timeline:
            def event_index(self, measure: str, *, layer: str = "words"):
                del measure
                participant = np.array(["p1"] * 4, dtype=object)
                stimulus = np.array(["d1"] * 4, dtype=object)
                series, _ = series_codes(participant, stimulus)
                return EventIndex(
                    participant_id=participant,
                    stimulus_id=stimulus,
                    unit_index=np.array([0, 1, 2, 1]),
                    time=np.array([0.0, 1.0, 2.0, 9.0]),
                    series=series,
                    layer=layer,
                )

        def respond(units):
            return build_response_events(
                dataset=Timeline(),
                dv_name="rt",
                rows={
                    "participant_id": np.array(["p1"] * len(units)),
                    "stimulus_id": np.array(["d1"] * len(units)),
                    "unit_index": np.array(units),
                },
                feature_arrays={},
                control_names=(),
                focal_names=(),
            )

        np.testing.assert_array_equal(respond([0, 2]).time, [0.0, 2.0])
        with pytest.raises(ValueError, match="1 of 2 analyzed rows match more than"):
            respond([0, 1])
