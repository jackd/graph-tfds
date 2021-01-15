import json
import os

import numpy as np
import tensorflow as tf

from graph_tfds.graphs.ppi import NUM_CLASSES, NUM_FEATURES
from graph_tfds.graphs.ppi.ppi_test import PPITest

dummy_data_dir = os.path.realpath(os.path.join(os.path.dirname(__file__), "dummy_data"))


def create_fake_data(split: str, split_name: str, nodes_per_graph=10):
    def save_np(name, data):
        path = os.path.join(dummy_data_dir, f"{split_name}_{name}.npy")
        with tf.io.gfile.GFile(path, "wb") as fp:
            np.save(fp, data)

    def save_json(name, data):
        path = os.path.join(dummy_data_dir, f"{split_name}_{name}.json")
        with tf.io.gfile.GFile(path, "wb") as fp:
            json.dump(data, fp)

    num_graphs = PPITest.SPLITS[split]
    rng = np.random.default_rng(0)
    labels = rng.integers(
        2, size=(nodes_per_graph * num_graphs, NUM_CLASSES), dtype=np.int64,
    )
    feats = rng.uniform(size=(nodes_per_graph * num_graphs, NUM_FEATURES))

    graph_ids = np.concatenate(
        (
            rng.integers(num_graphs, size=(nodes_per_graph - 1) * num_graphs),
            np.arange(num_graphs),
        ),
        axis=0,
    )
    graph_ids.sort()

    graph = {"links": [{"source": i, "target": i} for i in range(graph_ids.shape[0])]}

    save_np("labels", labels)
    save_np("graph_id", graph_ids)
    save_np("feats", feats)
    save_json("graph", graph)


if __name__ == "__main__":
    create_fake_data("train", "train")
    create_fake_data("validation", "valid")
    create_fake_data("test", "test")
