# Fixture data

`fixture_math.jsonl` is a twelve-example, hand-audited fixture written for
offline smoke tests.  It is not a claim of training on GSM8K, MATH, or
OpenR1.  Each row contains a question, an exact gold answer, and a short
reference solution.  The smoke script creates four candidate completions per
question so that correctness and output-format rewards can be inspected.

For a real experiment, replace this file with a documented public split such
as GSM8K or OpenR1-Math, record the dataset revision and SHA-256, and perform a
train/validation/test contamination audit before reporting metrics.

`preference_pairs.jsonl` is a deterministic derived artifact produced by
`python scripts/build_preference_pairs.py`; it can be deleted and regenerated
from the fixture at any time.
