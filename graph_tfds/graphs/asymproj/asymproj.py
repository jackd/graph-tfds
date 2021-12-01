"""asymproj dataset."""
import pickle as pkl

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

_DESCRIPTION = """
Graph edge data provided in Learning Edge Representation via Low-Rank Asymmetric Projections.
"""

_CITATION = """
@INPROCEEDINGS{asymproj,
  authors = {Sami Abu-El-Haija AND Bryan Perozzi AND Rami Al-Rfou},
  title = {Learning Edge Representations via Low-Rank Asymmetric Projections},
  booktitle = {ACM International Conference on Information and Knowledge Management (CIKM)},
  year = {2017},
}
"""

CA_ASTRO_PH = tfds.core.BuilderConfig(name="ca-AstroPh")
CA_HEP_TH = tfds.core.BuilderConfig(name="ca-HepTh")
SOC_EPINION = tfds.core.BuilderConfig(name="soc-epinions")
SOC_FACEBOOK = tfds.core.BuilderConfig(name="soc-facebook")
WIKI_VOTE = tfds.core.BuilderConfig(name="wiki-vote")
PPI = tfds.core.BuilderConfig(name="ppi")


class Asymproj(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for asymproj dataset."""

    VERSION = tfds.core.Version("1.0.0")
    RELEASE_NOTES = {
        "1.0.0": "Initial release.",
    }

    BUILDER_CONFIGS = [
        CA_ASTRO_PH,
        CA_HEP_TH,
        SOC_EPINION,
        SOC_FACEBOOK,
        WIKI_VOTE,
        PPI,
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "test_pos": tfds.features.Tensor(shape=(None, 2), dtype=tf.int64),
                    "test_neg": tfds.features.Tensor(shape=(None, 2), dtype=tf.int64),
                    "train_pos": tfds.features.Tensor(shape=(None, 2), dtype=tf.int64),
                    "train_neg": tfds.features.Tensor(shape=(None, 2), dtype=tf.int64),
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            homepage="https://github.com/google/asymproj_edge_dnn",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            {"data": "http://sami.haija.org/graph/datasets.tgz"}
        )["data"]

        return {
            "full": self._generate_examples(
                path / "datasets" / self.builder_config.name
            ),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        with tf.io.gfile.GFile(path / "index.pkl", "rb") as fp:
            index = pkl.load(fp)
        yield 0, {
            "test_pos": np.load(tf.io.gfile.GFile(path / "test.txt.npy", "rb")).astype(
                np.int64
            ),
            "test_neg": np.load(
                tf.io.gfile.GFile(path / "test.neg.txt.npy", "rb")
            ).astype(np.int64),
            "train_pos": np.load(tf.io.gfile.GFile(path / "test.txt.npy", "rb")).astype(
                np.int64
            ),
            "train_neg": np.load(
                tf.io.gfile.GFile(path / "test.neg.txt.npy", "rb")
            ).astype(np.int64),
            "num_nodes": index["largest_cc_num_nodes"],
        }
