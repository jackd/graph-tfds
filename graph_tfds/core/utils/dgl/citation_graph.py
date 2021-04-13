import os

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.features.sparse import SparseComponents, SparseTensor
from graph_tfds.core.utils.dgl.core import DGL_URL
from graph_tfds.core.utils.file_io import load_pickle, loadtxt


def load_citation_graph(name, data_dir):
    def full_path(attr_name):
        return os.path.join(data_dir, f"ind.{name}.{attr_name}")

    x, y, tx, ty, allx, ally, graph = (
        load_pickle(full_path(n), encoding="latin1")
        for n in ("x", "y", "tx", "ty", "allx", "ally", "graph")
    )

    test_idx_reorder = loadtxt(full_path("test.index"), dtype=np.int64)
    test_idx_range = np.sort(test_idx_reorder)
    a = nx.adjacency_matrix(nx.from_dict_of_lists(graph))  # CSR
    a.setdiag(0)
    a.eliminate_zeros()
    a = a.tocoo()
    links = np.stack((a.row, a.col), axis=1).astype(np.int64)
    if name == "citeseer":
        # Fix citeseer dataset (there are some isolated nodes in the graph)
        # Find isolated nodes, add them as zero-vecs into the right position
        test_idx_range_full = range(min(test_idx_reorder), max(test_idx_reorder) + 1)
        tx_extended = sp.lil_matrix((len(test_idx_range_full), x.shape[1]))
        tx_extended[test_idx_range - min(test_idx_range), :] = tx
        tx = tx_extended
        ty_extended = np.zeros((len(test_idx_range_full), y.shape[1]))
        ty_extended[test_idx_range - min(test_idx_range), :] = ty
        ty = ty_extended

    node_features = sp.vstack((allx, tx)).tolil()
    node_features[test_idx_reorder, :] = node_features[test_idx_range, :]

    onehot_labels = np.vstack((ally, ty))
    onehot_labels[test_idx_reorder, :] = onehot_labels[test_idx_range, :]
    node_labels = np.argmax(onehot_labels, 1)

    test_ids = test_idx_range
    train_ids = np.arange(len(y), dtype=np.int64)
    validation_ids = np.arange(len(y), len(y) + 500, dtype=np.int64)
    node_features = node_features.tocoo()

    return {
        "graph": {
            "links": links,
            "node_features": SparseComponents(
                indices=np.stack(
                    (node_features.row, node_features.col), axis=-1
                ).astype(np.int64),
                values=node_features.data.astype(np.float32),
                dense_shape=node_features.shape,
            ),
        },
        "node_labels": node_labels,
        "train_ids": train_ids,
        "validation_ids": validation_ids,
        "test_ids": test_ids,
    }


_CITATION = """\
@article{sen2008collective,
  title={Collective classification in network data},
  author={Sen, Prithviraj and Namata, Galileo and Bilgic, Mustafa and Getoor, Lise and Galligher, Brian and Eliassi-Rad, Tina},
  journal={AI magazine},
  volume={29},
  number={3},
  pages={93--93},
  year={2008}
}
"""

_DESCRIPTION = "{} citation network, with preprocessing as per DGL."


class CitationGraph(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for CitationGraph dataset."""

    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION.format(self._NAME),
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
            homepage="http://www.cs.umd.edu/~sen/lbc-proj/LBC.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        data_dir = dl_manager.download_and_extract(f"{DGL_URL}{self._URL_NAME}.zip")

        return {
            "full": self._generate_examples(data_dir),
        }

    def _generate_examples(self, data_dir: str):
        """Yields examples."""
        data = load_citation_graph(self._URL_NAME, data_dir)
        yield 0, data


def _vis(builder):
    # pylint: disable=import-outside-toplevel
    import matplotlib.pyplot as plt

    # pylint: enable=import-outside-toplevel

    name = builder.name
    builder.download_and_prepare(
        download_config=tfds.core.download.DownloadConfig(register_checksums=True)
    )
    print(f"Prepared {name}")

    inputs, labels = tf.data.experimental.get_single_element(
        builder.as_dataset(as_supervised=True, split="full")
    )
    row_ids, col_ids = inputs["links"].numpy().T
    labels = labels.numpy()
    G = nx.Graph()
    for i in range(len(labels)):
        G.add_node(i)
    for ri, ci in zip(row_ids, col_ids):
        G.add_edge(ri, ci)
    print(f"{name} has {len(labels)} nodes and {len(row_ids)} edges")
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, alpha=0.4, node_size=2)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.1)
    plt.show()
