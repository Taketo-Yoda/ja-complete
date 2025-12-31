# N-gram Model Data

This directory contains N-gram model files for Japanese text completion.

## License

**The model files (`*.pkl`) in this directory are licensed under CC BY-SA 3.0,
not the MIT License that applies to the source code.**

### default_ngram.pkl

- **License**: [CC BY-SA 3.0](https://creativecommons.org/licenses/by-sa/3.0/)
- **Source**: Japanese Wikipedia (https://ja.wikipedia.org/)
- **Copyright**: © Wikipedia contributors
- **Training Date**: 2025
- **Dataset**: Subset of approximately 100,000 articles from jawiki-latest-pages-articles.xml.bz2

This model contains statistical information (word frequencies and n-gram
probabilities) derived from Japanese Wikipedia articles. It does not contain
the original article text.

## Using the Model

When using this model in your projects:

1. **Attribution**: Give appropriate credit to Wikipedia
2. **License**: Any modifications must be distributed under CC BY-SA 3.0 or compatible
3. **Notice**: Indicate if you made changes to the model

Example attribution:
```
This software uses an N-gram model derived from Japanese Wikipedia
(https://ja.wikipedia.org/), licensed under CC BY-SA 3.0.
```

## Full License

See [LICENSE-CC-BY-SA.txt](../../../LICENSE-CC-BY-SA.txt) in the project root
for the complete license text and compliance requirements.
