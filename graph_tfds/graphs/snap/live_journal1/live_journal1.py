"""live_journal1 dataset."""

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_datasets as tfds
from graph_tfds.graphs.snap.core import CITATION, HOMEPAGE

_DESCRIPTION = "LiveJournal online social network."


class LiveJournal1(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for live_journal1 dataset."""

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
                    "edges": tfds.features.Tensor(shape=(68993772, 2), dtype=tf.int64),
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=None,
            homepage=f"{HOMEPAGE}soc-LiveJournal1.html",
            citation=CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://snap.stanford.edu/data/soc-LiveJournal1.txt.gz"
        )

        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        with open(path, "rb") as fp:
            edges = pd.read_csv(fp, comment="#", delimiter="\t").values
        yield 0, {"edges": edges.astype(np.int64), "num_nodes": edges.max() + 1}
