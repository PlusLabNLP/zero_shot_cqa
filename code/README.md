### Syntactic Transformation

First, install the dependencies:
```
pip install allennlp==2.1.0 allennlp-models==2.1.0
```

Then, you can prepare the data in the same format as `data/dev.src` and run the command:

```
python rule_based_transform.py $INPUT > $OUTPUT
```
