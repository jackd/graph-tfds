"""graph_saint dataset."""

import json
import os
from typing import Dict

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

import gdown

_DESCRIPTION = """\
Datasets used in/provided by [GraphSAINT](https://github.com/GraphSAINT/GraphSAINT)."""

_CITATION = """\
@inproceedings{graphsaint-iclr20,
title={{GraphSAINT}: Graph Sampling Based Inductive Learning Method},
author={Hanqing Zeng and Hongkuan Zhou and Ajitesh Srivastava and Rajgopal Kannan and Viktor Prasanna},
booktitle={International Conference on Learning Representations},
year={2020},
url={https://openreview.net/forum?id=BJe8pkHFwS}
}
"""

_DRIVE_URL = "https://drive.google.com/uc?export=download&id={}"


class GraphSaintConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        *,
        num_classes: int,
        num_features: int,
        drive_ids: Dict[str, str],
        **kwargs,
    ):
        self.drive_ids = drive_ids
        self.num_classes = num_classes
        self.num_features = num_features
        super().__init__(
            version=tfds.core.Version("1.0.0"),
            release_notes={"1.0.0": "Initial release."},
            description=_DESCRIPTION,
            **kwargs,
        )


YELP = GraphSaintConfig(
    name="yelp",
    num_classes=100,
    num_features=300,
    drive_ids={
        "adj_full.npz": "1Juwx8HtDwSzmVIJ31ooVa1WljI4U5JnA",
        # "adj_train.npz": "1nnkeyMcaro-2_j20CLZ0P6nH4SdivEgJ",
        "feats.npy": "1Zy6BZH_zLEjKlEFSduKE5tV9qqA_8VtM",
        "role.json": "1NI5pa5Chpd-52eSmLW60OnB3WS5ikxq_",
        "class_map.json": "1VUcBGr0T0-klqerjAjxRmAqFuld_SMWU",
    },
)


class GraphSaint(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for graph_saint dataset."""

    BUILDER_CONFIGS = [YELP]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "graph": {
                        "links": tfds.core.features.Tensor(
                            shape=(None, 2), dtype=tf.int64
                        ),
                        "node_features": tfds.core.features.Tensor(
                            shape=(None, self.builder_config.num_features),
                            dtype=tf.float32,
                        ),
                    },
                    "node_labels": tfds.core.features.Tensor(
                        shape=(None, self.builder_config.num_classes), dtype=tf.int64,
                    ),
                    "train_ids": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.int64
                    ),
                    "validation_ids": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.int64
                    ),
                    "test_ids": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.int64
                    ),
                }
            ),
            supervised_keys=("graph", "node_labels"),
            homepage="http://snap.stanford.edu/graphsage/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        name = self.builder_config.name
        ids = self.builder_config.drive_ids
        dl_dir = dl_manager._download_dir  # pylint: disable=protected-access
        paths = {k: os.path.join(dl_dir, f"{name}-{k}") for k in ids}
        missing_paths = {k: v for k, v in paths.items() if not tf.io.gfile.exists(v)}

        for k, path in missing_paths.items():
            url = _DRIVE_URL.format(ids[k])
            gdown.download(url, output=path)

        return {
            "train": self._generate_examples(paths),
        }

    def _generate_examples(self, paths):
        """Yields examples."""
        with tf.io.gfile.GFile(paths["class_map.json"], "rb") as fp:
            class_map = json.load(fp)
        labels = np.empty(
            (len(class_map), self.builder_config.num_classes), dtype=np.int64
        )
        for k, v in class_map.items():
            labels[int(k)] = v
        del class_map

        with tf.io.gfile.GFile(paths["adj_full.npz"], "rb") as fp:
            adj = sp.load_npz(fp).tocoo()
        links = np.stack((adj.row, adj.col), axis=-1).astype(np.int64)
        del adj

        with tf.io.gfile.GFile(paths["feats.npy"], "rb") as fp:
            feats = np.load(fp).astype(np.float32)

        with tf.io.gfile.GFile(paths["role.json"], "rb") as fp:
            roles = json.load(fp)

        train_ids, validation_ids, test_ids = (
            np.array(roles[k], dtype=np.int64) for k in ("tr", "va", "te")
        )
        del roles

        data = dict(
            graph=dict(links=links, node_features=feats),
            node_labels=labels,
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )
        yield 0, data


if __name__ == "__main__":

    config = tfds.core.download.DownloadConfig(register_checksums=True)
    for builder_config in GraphSaint.BUILDER_CONFIGS:
        builder = GraphSaint(config=builder_config)
        builder.download_and_prepare(download_config=config)
