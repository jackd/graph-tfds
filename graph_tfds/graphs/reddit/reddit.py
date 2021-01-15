"""reddit dataset."""
import os

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.utils.dgl.core import DGL_URL

_DESCRIPTION = """
Reddit dataset for community detection (node classification).

It should also contain any processing which has been applied (if any),
(e.g. corrupted example skipped, images cropped,...):
"""

_CITATION = """\
@inproceedings{hamilton2017inductive,
  author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title = {Inductive Representation Learning on Large Graphs},
  booktitle = {NIPS},
  year = {2017}
}
"""


class RedditCommunities(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for reddit dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

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
                            shape=(None, 602), dtype=tf.float32
                        ),
                    },
                    "node_labels": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.int64,
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
        path = dl_manager.download_and_extract(f"{DGL_URL}reddit.zip")

        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        coo_adj = sp.load_npz(os.path.join(path, "reddit_graph.npz"))
        # features and labels
        reddit_data = np.load(os.path.join(path, "reddit_data.npz"))
        features = reddit_data["feature"].astype(np.float32)

        labels = reddit_data["label"].astype(np.int64)
        # tarin/val/test indices
        node_types = reddit_data["node_types"]
        del reddit_data
        links = np.stack((coo_adj.row, coo_adj.col), axis=-1).astype(np.int64)
        del coo_adj
        data = dict(
            graph=dict(links=links, node_features=features),
            node_labels=labels,
            train_ids=np.where(node_types == 1)[0].astype(np.int64),
            validation_ids=np.where(node_types == 2)[0].astype(np.int64),
            test_ids=np.where(node_types == 3)[0].astype(np.int64),
        )
        del node_types, labels, links, features
        yield 0, data


if __name__ == "__main__":

    config = tfds.core.download.DownloadConfig(register_checksums=True)
    reddit = RedditCommunities()
    reddit.download_and_prepare(download_config=config)

    dataset = reddit.as_dataset(split="full", as_supervised=True)
