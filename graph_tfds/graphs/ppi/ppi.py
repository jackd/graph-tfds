import os
from typing import List, Tuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.utils.file_io import load_json, load_np

NUM_FEATURES = 50
NUM_CLASSES = 121

CITATION = """\
@article{zitnik2017predicting,
  title={Predicting multicellular function through multi-layer tissue networks},
  author={Zitnik, Marinka and Leskovec, Jure},
  journal={Bioinformatics},
  volume={33},
  number={14},
  pages={i190--i198},
  year={2017},
  publisher={Oxford University Press}
}
"""


def raggedify_batched_graphs(
    node_graph_ids: np.ndarray,
    node_features: np.ndarray,
    node_labels: np.ndarray,
    links: np.ndarray,
) -> Tuple[List[np.ndarray], List[np.ndarray], List[np.ndarray]]:
    row_lengths = np.bincount(node_graph_ids)
    row_splits = np.cumsum(row_lengths[:-1])
    node_features = np.split(node_features, row_splits)
    node_labels = np.split(node_labels, row_splits)

    link_graph_ids = node_graph_ids[links[:, 0]]
    row_lengths = np.bincount(link_graph_ids)
    row_splits = np.cumsum(row_lengths[:-1])
    links = np.split(links, row_splits)
    for link, offset in zip(links, row_splits):
        link -= offset
    return node_features, node_labels, links


class PPI(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                dict(
                    graph=dict(
                        node_features=tfds.core.features.Tensor(
                            shape=(None, NUM_FEATURES), dtype=tf.float32
                        ),
                        links=tfds.core.features.Tensor(
                            shape=(None, 2), dtype=tf.int64
                        ),
                    ),
                    node_labels=tfds.core.features.Tensor(
                        shape=(None, NUM_CLASSES), dtype=tf.bool
                    ),
                )
            ),
            supervised_keys=("graph", "node_labels"),
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        data_dir: str = dl_manager.download_and_extract(
            "https://data.dgl.ai/dataset/ppi.zip"
            # "https://s3.us-east-2.amazonaws.com/dgl.ai/dataset/ppi.zip"
        )
        return {
            "train": self._generate_examples(data_dir, "train"),
            "validation": self._generate_examples(data_dir, "valid"),
            "test": self._generate_examples(data_dir, "test"),
        }

    def _generate_examples(self, data_dir: str, data_name: str):
        def full_path(name, ext="npy"):
            return os.path.join(data_dir, f"{data_name}_{name}.{ext}")

        graph_data = load_json(full_path("graph", "json"))
        node_features = load_np(full_path("feats")).astype(np.float32)
        node_labels = load_np(full_path("labels")).astype(np.bool)
        node_graph_ids = load_np(full_path("graph_id"))
        node_graph_ids -= np.min(node_graph_ids)

        links = graph_data["links"]
        links = np.array(
            tuple((link["source"], link["target"]) for link in links), dtype=np.int64
        )
        node_features, node_labels, links = raggedify_batched_graphs(
            node_graph_ids, node_features, node_labels, links
        )
        for i, nf in enumerate(node_features):
            yield i, dict(
                graph=dict(node_features=nf, links=links[i]),
                node_labels=node_labels[i],
            )


if __name__ == "__main__":
    config = tfds.core.download.DownloadConfig(register_checksums=True)
    ppi = PPI()
    ppi.download_and_prepare(download_config=config)

    dataset = ppi.as_dataset(split="train", as_supervised=True)
    node_sizes = []
    for inputs, labels in dataset:
        nodes = inputs["node_features"]
        node_sizes.append(nodes.shape[0])

    print("Node size stats:")
    print(f"  mean: {np.mean(node_sizes)}")
    print(f"  std : {np.std(node_sizes)}")
