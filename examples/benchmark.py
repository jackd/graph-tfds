"""
This script benchmarks the tfds / accumulator-batched implementation and tf2-gnn.

In addition to graph-tfds, this requires the following packages:

* [accumulator-batching](https://github.com/jackd/accumulator-batching)
* [tf2-gnn](https://github.com/microsoft/tf2-gnn)
* [absl-py](https://github.com/abseil/abseil-py)
* wget

```bash
git clone https://github.com/jackd/accumulator-batching.git
pip install -e accumulator-batching
pip install tf2-gnn absl-py wget
```
"""
import os
import zipfile
from typing import Callable, Optional

import tensorflow as tf
import tensorflow_datasets as tfds
import wget
from absl import app, flags
from dpu_utils.utils import RichPath

from accumulator_batching import RaggedAccumulator, TensorAccumulator, accumulated_batch
from graph_tfds import graphs  # pylint:ignore=unused-import
from tf2_gnn import data

flags.DEFINE_string("problem", default="ppi", help="one of ppi, qm9")
flags.DEFINE_bool("tfds", default=False, help="use tfds-based builder")
flags.DEFINE_integer("max_nodes", default=None, help="maximum number of nodes")
flags.DEFINE_integer("batch_size", default=None, help="maximum number of nodes")
flags.DEFINE_string("split", default="train", help="train, validation, test")
flags.DEFINE_integer("burn_iters", default=10, help="burn iterations")
flags.DEFINE_integer("min_iters", default=50, help="minimum run iters")

FLAGS = flags.FLAGS


def summarize(result, print_fn=print):
    """
    Args:
        result: output of a tf.test.Benchmark.run_op_benchmark call.
        print_fn: print-like function.
    """
    print_fn("Wall time (ms): {}".format(result["wall_time"] * 1000))
    gpu_mem = result["extras"].get("allocator_maximum_num_bytes_GPU_0_bfc", 0)
    print_fn("Memory (Mb):    {}".format(gpu_mem / 1024 ** 2))


def benchmark_dataset(
    dataset_fn: Callable[[], tf.data.Dataset], burn_iters: int, min_iters: int
):
    with tf.Graph().as_default() as graph:
        dataset = dataset_fn()
        element = tf.compat.v1.data.make_one_shot_iterator(dataset.repeat()).get_next()
        with tf.compat.v1.Session(graph=graph) as sess:
            bm = tf.test.Benchmark()
            print("Starting benchmarking...")
            result = bm.run_op_benchmark(
                sess, element, burn_iters=burn_iters, min_iters=min_iters
            )
            summarize(result)


def _unpack_dataset(*args):
    if len(args) == 1:
        (kwargs,) = args
        inputs = kwargs["inputs"]
        labels = kwargs["labels"]
    else:
        inputs, labels = args
    if isinstance(inputs, dict):
        node_features = inputs["node_features"]
        links = inputs["links"]
    else:
        node_features, links = inputs
    if not isinstance(links, tuple):
        links = (links,)
    return node_features, links, labels


def _block_diagonalize_batched(node_features, links, labels):
    offset = tf.expand_dims(node_features.row_splits, axis=-1)
    links = tuple(
        # tf.cast(link.values, tf.int32) + tf.gather(offset, link.value_rowids())
        link.values + tf.gather(offset, link.value_rowids())
        for link in links
    )
    if isinstance(labels, tf.RaggedTensor):
        labels = labels.values
    return (node_features, links), labels


def block_diagonal_batch_with_batch_size(dataset: tf.data.Dataset, batch_size: int):
    """
    Batch the input dataset block diagonally up to the given batch size.

    Args:
        dataset: tf.data.Dataset with spec ((nodes, (link*)), labels).
            nodes: [V?, ...] node features.
            link: [E?, 2] int edge/link indices.
            labels: [V?, ...] or [...] label data.
        batch_size: number of examples in the resulting batch.

    Returns:
        dataset with spec:
            nodes: [B, V?, ...] ragged node features.
            links: [E, 2] indices into flattened nodes.
            labels: [BV, ...] or [B, ...]
        B = batch_size
        BV = sum_b V_b
    """
    dataset = dataset.apply(
        tf.data.experimental.dense_to_ragged_batch(
            batch_size, row_splits_dtype=tf.int32
        )
    )
    return dataset.map(
        lambda *args: _block_diagonalize_batched(*_unpack_dataset(*args))
    )


def block_diagonal_batch_with_max_nodes(
    dataset: tf.data.Dataset, max_nodes: int,
):
    """
    Batch the input dataset block diagonally up to the speicified max nodes.

    All examples are assumed to have at least 1 node.

    Args:
        dataset: tf.data.Dataset with spec ((nodes, (link*)), labels).
            nodes: [V?, ...] node features.
            link: [E?, 2] int edge/link indices.
            labels: [V?, ...] or [...] label data.
        max_nodes: maximum number of nodes allowed in each batch.

    Returns:
        dataset with spec:
            nodes: [B, V?, ...] ragged node features.
            links: [E, 2] indices into flattened nodes.
            labels: [BV, ...] or [B, ...]
        BV <= max_nodes is the total number not nodes.
    """
    dataset.element_spec
    dataset = dataset.map(_unpack_dataset)
    node_features, links, labels = dataset.element_spec

    node_acc = RaggedAccumulator(flat_spec=node_features, max_flat_size=max_nodes,)
    link_accs = tuple(RaggedAccumulator(flat_spec=link) for link in links)
    label_acc = (
        RaggedAccumulator(labels)
        if labels.shape[0] is None
        else TensorAccumulator(labels)
    )
    accumulator = (node_acc, link_accs, label_acc)
    dataset = accumulated_batch(dataset, accumulator)
    dataset = dataset.map(_block_diagonalize_batched)
    return dataset


def get_tfds_dataset_fn(
    problem: str,
    split: str,
    max_nodes: Optional[int] = None,
    batch_size: Optional[int] = None,
):

    builder = tfds.builder(problem)
    builder.download_and_prepare()

    def add_back_edges(inputs, labels):
        nodes, links = inputs
        links = links + tuple(tf.reverse(l, axis=[1]) for l in links)
        return (nodes, links), labels

    def f():
        dataset = builder.as_dataset(split=split, as_supervised=True).shuffle(256)

        if batch_size is None:
            assert isinstance(max_nodes, int)
            dataset = block_diagonal_batch_with_max_nodes(dataset, max_nodes)
        else:
            dataset = block_diagonal_batch_with_batch_size(dataset, batch_size)
        dataset = dataset.map(add_back_edges)  # for comparison with base
        return dataset

    return f


def get_baseline_data_dir(problem):
    data_dir = f"data/{problem}"
    if not os.path.isdir(data_dir):
        if problem == "ppi":
            print("Downloading ppi data...")
            zipped = wget.download(
                "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip",
                out="/tmp/",
            )
            with zipfile.ZipFile(zipped, "r") as zipObj:
                zipObj.extractall(data_dir)
        elif problem == "qm9":
            os.makedirs(data_dir)
            paths = []
            try:
                for split in ("train", "valid", "test"):
                    path = os.path.join(data_dir, f"{split}.jsonl.gz")
                    print(f"Downloading qm9/{split} data...")
                    wget.download(
                        f"https://github.com/microsoft/tf-gnn-samples/raw/master/data/qm9/{split}.jsonl.gz",
                        out=path,
                    )
                    paths.append(path)
            except Exception:
                for path in paths:
                    os.remove(path)
                os.rmdir(data_dir)
                raise
        assert os.path.isdir(data_dir)
    return data_dir


def get_baseline_dataset_fn(problem: str, split: str, max_nodes: int):
    graph_dataset_cls = {"ppi": data.PPIDataset, "qm9": data.QM9Dataset}[problem]
    params = graph_dataset_cls.get_default_hyperparameters()
    params.update(
        dict(max_nodes=max_nodes, add_self_loop_edges=False, tie_fwd_bkwd_edges=False,)
    )
    graph_dataset = graph_dataset_cls(params)
    graph_dataset.load_data(RichPath.create(get_baseline_data_dir(problem)))
    fold = {
        "train": data.DataFold.TRAIN,
        "validation": data.DataFold.VALIDATION,
        "test": data.DataFold.TEST,
    }[split]

    def f():
        return graph_dataset.get_tensorflow_dataset(fold)

    return f


def main(_):
    FLAGS = flags.FLAGS
    kwargs = dict(problem=FLAGS.problem, split=FLAGS.split)

    max_nodes = FLAGS.max_nodes
    batch_size = FLAGS.batch_size
    if batch_size is None:
        if max_nodes is None:
            max_nodes = {"ppi": 10000, "qm9": 1000}[FLAGS.problem]
        kwargs["max_nodes"] = max_nodes
    else:
        if not FLAGS.tfds:
            raise NotImplementedError(
                "batch_size implementation only available for tfds " "implementations."
            )
        kwargs["batch_size"] = batch_size
    if FLAGS.tfds:
        dataset_fn = get_tfds_dataset_fn(**kwargs)
    else:
        dataset_fn = get_baseline_dataset_fn(**kwargs)
    benchmark_dataset(
        dataset_fn, burn_iters=FLAGS.burn_iters, min_iters=FLAGS.min_iters
    )
    if not FLAGS.tfds:
        print(
            "Benchmark complete. Please ignore possible GeneratorDataset "
            "iterator errors and kill program"
        )
    exit(0)


if __name__ == "__main__":
    app.run(main)
