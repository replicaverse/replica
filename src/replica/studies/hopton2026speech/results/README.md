# Speech study inputs

[index.json](index.json) pins the input files, archive revisions and SHA-256 digests.
The study fetches and verifies the required tables when first used, then reads them
from the user cache. To fetch the analysis inputs in advance:

```bash
replica results fetch hopton2026speech --verification
```

The analysis bundle contains pupil observations, six predictor tables and the fixed
cross-validation folds. A second archive retains the probability inputs used to
build the predictor tables. No fitted result runs are registered here; the
[numerical fixtures](../../../../../tests/studies/hopton2026speech/fixtures) contain
the reference summaries used by the reproduction tests.
