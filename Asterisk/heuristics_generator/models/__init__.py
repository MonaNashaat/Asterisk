from __future__ import absolute_import

from .meta import AsteriskBase, AsteriskSession, Asterisk_engine, Asterisk_postgres
from .context import Context, Document, Sentence, TemporarySpan, Span
from .context import construct_stable_id, split_stable_id
from .candidate import Candidate, candidate_subclass, Marginal
from .annotation import (
    Feature, FeatureKey, Label, LabelKey, GoldLabel, GoldLabelKey, StableLabel,
    Prediction, PredictionKey
)

# This call must be performed after all classes that extend AsteriskBase are
# declared to ensure the storage schema is initialized
AsteriskBase.metadata.create_all(Asterisk_engine)
