"""web_google dataset."""

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_datasets as tfds
from graph_tfds.graphs.snap.core import CITATION, HOMEPAGE

_DESCRIPTION = """
Nodes represent web pages and directed edges represent hyperlinks between them. The
data was released in 2002 by Google as a part of Google Programming Contest.
"""


class WebGoogle(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for web_google dataset."""

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
                    "edges": tfds.features.Tensor(shape=(5105038, 2), dtype=tf.int64),
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=None,
            homepage=f"{HOMEPAGE}web-Google.html",
            citation=CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://snap.stanford.edu/data/web-Google.txt.gz"
        )

        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        with open(path, "rb") as fp:
            edges = pd.read_csv(fp, comment="#", delimiter="\t").values
        yield 0, {"edges": edges.astype(np.int64), "num_nodes": edges.max() + 1}
