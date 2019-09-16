from __future__ import absolute_import

from .meta import SnorkelBase, SnorkelSession, snorkel_engine, snorkel_postgres
from .context import Context, Document, Sentence, TemporarySpan, Span
from .context import construct_stable_id, split_stable_id
from .candidate import Candidate, candidate_subclass, Marginal
from .annotation import (
    Feature, FeatureKey, Label, LabelKey, GoldLabel, GoldLabelKey, StableLabel,
    Prediction, PredictionKey
)

# This call must be performed after all classes that extend SnorkelBase are
# declared to ensure the storage schema is initialized
SnorkelBase.metadata.create_all(snorkel_engine)
