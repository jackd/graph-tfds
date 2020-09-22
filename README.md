# graph-tfds

[tensorflow-datasets](https://github.com/tensorflow/datasets) implementations of various open source graph datasets.

Loading logic based on [tf2-gnn](https://github.com/microsoft/tf2-gnn).

## Comparison with tf2-gnn

`ppi` and `qm9` datasets are based on loading logic from [tf2-gnn](https://github.com/microsoft/tf2-gnn), though the interface is not identical. To see how to recover the original implementation, see [examples/benchmark.py](examples/benchmark.py).

## Pre-commit

This package uses [pre-commit](https://pre-commit.com/) to ensure commits meet minimum criteria. To Install, use

```bash
pip install pre-commit
pre-commit install
```

This will ensure git hooks are run before each commit. While it is not advised to do so, you can skip these hooks with

```bash
git commit --no-verify -m "commit message"
```
