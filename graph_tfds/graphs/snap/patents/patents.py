"""web_google dataset."""

import numpy as np
import pandas as pd
import tensorflow as tf

import tensorflow_datasets as tfds
from graph_tfds.graphs.snap.core import CITATION, HOMEPAGE

_DESCRIPTION = """
U.S. patent dataset is maintained by the National Bureau of Economic Research. The
dataset spans 37 years (January 1, 1963 to December 30, 1999), and includes all the
utility patents granted during that period, totaling 3,923,922 patents. The citation
graph includes all citations made by patents granted between 1975 and 1999, totaling
16,522,438 citations. For the patents dataset there are 1,803,511 nodes for which we
have no information about their citations (we only have the in-links).

The data was originally released by NBER.
"""


class Patents(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for patents dataset."""

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
                    "edges": tfds.features.Tensor(shape=(16518947, 2), dtype=tf.int64),
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int64),
                }
            ),
            supervised_keys=None,
            homepage=f"{HOMEPAGE}cit-Patents.html",
            citation=CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            "https://snap.stanford.edu/data/cit-Patents.txt.gz"
        )

        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        with open(path, "rb") as fp:
            edges = pd.read_csv(fp, comment="#", delimiter="\t").values
        yield 0, {"edges": edges.astype(np.int64), "num_nodes": edges.max() + 1}
