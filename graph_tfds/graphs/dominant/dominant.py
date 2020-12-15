"""dominant dataset."""

import numpy as np
import scipy.io
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.features.ragged import RaggedComponents, pack_ragged_components

_DESCRIPTION = (
    "Single-example {name} dataset provided in "
    "_Deep Anomlay Detection on Attributed Networks_"
)

_CITATION = """\
@inproceedings{ding2019deep,
  title={Deep Anomaly Detection on Attributed Networks},
  author={Ding, Kaize and Li, Jundong and Bhanushali, Rohit and Liu, Huan},
  booktitle={SIAM International Conference on Data Mining (SDM)},
  year={2019}
}
"""

CONFIG_NAMES = (
    "Amazon",
    "BlogCatalog",
    "Disney",
    "Enron",
)


def csr_to_ragged(csr_matrix, dtype=tf.int64, row_splits_dtype=tf.int64):
    assert csr_matrix.has_sorted_indices
    return tf.RaggedTensor.from_row_splits(
        tf.convert_to_tensor(csr_matrix.indices, dtype=dtype),
        tf.convert_to_tensor(csr_matrix.indptr, dtype=row_splits_dtype),
    )


def load_data(path: str, name: str):
    data = scipy.io.loadmat(path)
    if name == "BlogCatalog":
        labels = data["Label"]
        attributes = sp.csr_matrix(data["Attributes"])
        adjacency = sp.lil_matrix(data["Network"])
    else:
        labels = data["gnd"]
        attributes = sp.csr_matrix(data["X"])
        adjacency = sp.lil_matrix(data["A"])

    adjacency = adjacency.tocsr()
    assert np.all(adjacency.data == 1)
    return dict(
        graph=dict(
            adjacency=csr_to_ragged(adjacency),
            node_features=dict(
                indices=csr_to_ragged(attributes),
                values=tf.convert_to_tensor(attributes.data, tf.float32),
            ),
        ),
        labels=tf.squeeze(tf.convert_to_tensor(labels, tf.bool), axis=-1),
    )


class Dominant(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for dominant dataset."""

    BUILDER_CONFIGS = [
        tfds.core.BuilderConfig(
            name,
            version=tfds.core.Version("0.0.1"),
            description=_DESCRIPTION.format(name=name),
        )
        for name in CONFIG_NAMES
    ]
    RELEASE_NOTES = {
        "0.0.1": "Initial release.",
    }

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "graph": {
                        "adjacency": RaggedComponents(
                            flat_shape=(None,), dtype=tf.int64
                        ),
                        "node_features": {
                            "indices": RaggedComponents(
                                flat_shape=(None,), dtype=tf.int64
                            ),
                            "values": tfds.core.features.Tensor(
                                shape=(None,), dtype=tf.float32
                            ),
                        },
                    },
                    "labels": tfds.core.features.Tensor(shape=(None,), dtype=tf.bool),
                }
            ),
            supervised_keys=("graph", "labels"),
            homepage="https://github.com/kaize0409/GCN_AnomalyDetection",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        name = self.builder_config.name

        path = dl_manager.download_and_extract(
            "https://github.com/kaize0409/GCN_AnomalyDetection/raw/master/gae/data/"
            f"{name}.mat"
        )
        return {"train": self._generate_examples(path)}

    def _generate_examples(self, path: str):
        """Yields examples."""
        yield 0, load_data(path, name=self.builder_config.name)


if __name__ == "__main__":
    import networkx as nx
    import matplotlib.pyplot as plt

    name_index = 0
    builder = Dominant(config=Dominant.BUILDER_CONFIGS[name_index])
    name = builder.builder_config.name
    builder.download_and_prepare(
        download_config=tfds.core.download.DownloadConfig(register_checksums=True)
    )
    print(f"Prepared {name}")

    inputs, labels = tf.data.experimental.get_single_element(
        builder.as_dataset(as_supervised=True, split="train")
    )

    adj = pack_ragged_components(inputs["adjacency"])
    row_ids = adj.value_rowids().numpy()
    col_ids = adj.values.numpy()
    labels = labels.numpy()
    G = nx.Graph()
    for i in range(len(labels)):
        G.add_node(i)
    for ri, ci in zip(row_ids, col_ids):
        G.add_edge(ri, ci)
    (normal,) = np.where(np.logical_not(labels))
    (different,) = np.where(labels)
    print(f"{name} has {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(f"{len(normal)} labels are False, {len(different)} labels are True")

    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, nodelist=normal, alpha=0.4, node_size=2)
    nx.draw_networkx_nodes(
        G, pos, nodelist=different, node_color="r", alpha=0.4, node_size=4
    )
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.1)
    plt.show()
