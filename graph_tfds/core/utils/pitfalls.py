"""
DatasetBuilder base class for datasets from

[Pitfalls of Graph Neural Network Evaluation](https://arxiv.org/pdf/1811.05868.pdf).

See `graphs.amazon` and `graphs.coauthors` for implementations.
"""
import pathlib
from typing import Mapping

import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.features.sparse import SparseComponents, SparseTensor

_CITATION = """\
@article{shchur2018pitfalls,
  title={Pitfalls of graph neural network evaluation},
  author={Shchur, Oleksandr and Mumme, Maximilian and Bojchevski, Aleksandar and G{\"u}nnemann, Stephan},
  journal={arXiv preprint arXiv:1811.05868},
  year={2018}
}
"""


def _load_csr_data(data: Mapping[str, np.ndarray], prefix: str):
    return sp.csr_matrix(
        (data[f"{prefix}_data"], data[f"{prefix}_indices"], data[f"{prefix}_indptr"]),
        shape=data[f"{prefix}_shape"],
    )


def _load_pifalls_npz_data(path: pathlib.Path):
    data = np.load(path)
    adj = _load_csr_data(data, "adj").tocoo()
    attr = _load_csr_data(data, "attr").tocoo()
    return dict(
        graph=dict(
            links=np.stack((adj.row, adj.col), axis=1).astype(np.int64),
            node_features=SparseComponents(
                indices=np.stack((attr.row, attr.col), axis=1).astype(np.int64),
                values=attr.data.astype(np.float32),
                dense_shape=attr.shape,
            ),
        ),
        node_labels=data["labels"].astype(np.int64),
    )


class PitfallsConfig(tfds.core.BuilderConfig):
    def __init__(self, name: str, url_name: str, description: str):
        super().__init__(name=name, description=description)
        self.url_name = url_name


with tfds.core.registered.skip_registration():

    class Pitfalls(tfds.core.GeneratorBasedBuilder):
        """DatasetBuilder for amazon dataset."""

        VERSION = tfds.core.Version("0.0.1")
        RELEASE_NOTES = {
            "0.0.1": "Initial release.",
        }

        def _info(self) -> tfds.core.DatasetInfo:
            """Returns the dataset metadata."""
            return tfds.core.DatasetInfo(
                builder=self,
                features=tfds.features.FeaturesDict(
                    {
                        "graph": {
                            "links": tfds.core.features.Tensor(
                                shape=(None, 2), dtype=tf.int64
                            ),
                            "node_features": SparseTensor(ndims=2, dtype=tf.float32),
                        },
                        "node_labels": tfds.core.features.Tensor(
                            shape=(None,), dtype=tf.int64,
                        ),
                    }
                ),
                supervised_keys=("graph", "node_labels"),
                homepage="https://github.com/shchur/gnn-benchmark",
                citation=_CITATION,
            )

        def _split_generators(self, dl_manager: tfds.download.DownloadManager):
            path = dl_manager.download_and_extract(
                "https://github.com/shchur/gnn-benchmark/raw/master/data/npz/"
                f"{self.builder_config.url_name}.npz"
            )

            return {
                "full": self._generate_examples(path),
            }

        def _generate_examples(self, path):
            return [(0, _load_pifalls_npz_data(path))]
