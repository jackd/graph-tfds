"""cora dataset."""
import os

import tensorflow as tf
import tensorflow_datasets as tfds

from graph_tfds.core.features.ragged import RaggedTensor
from graph_tfds.core.utils.file_io import load_content_and_cites

_DESCRIPTION = """Cora citation network."""

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

CLASS_NAMES = (
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
)


class Cora(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for cora dataset."""

    VERSION = tfds.core.Version("0.0.1")
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
                        "adjacency": RaggedTensor(flat_shape=(None,), dtype=tf.int64),
                        "node_features": RaggedTensor(
                            flat_shape=(None,), dtype=tf.int64
                        ),
                    },
                    "node_labels": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.int64,
                    ),
                    "node_ids": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.string
                    ),
                }
            ),
            supervised_keys=("graph", "node_labels"),
            homepage="http://www.cs.umd.edu/~sen/lbc-proj/LBC.html",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        cora_dir = dl_manager.download_and_extract(
            "http://www.cs.umd.edu/~sen/lbc-proj/data/cora.tgz"
        )

        return {
            "train": self._generate_examples(os.path.join(cora_dir, "cora")),
        }

    def _generate_examples(self, cora_dir: str):
        """Yields examples."""
        adjacency, node_features, labels, node_ids = load_content_and_cites(
            os.path.join(cora_dir, "cora"), {k: i for i, k in enumerate(CLASS_NAMES)}
        )
        yield 0, {
            "graph": {"adjacency": adjacency, "node_features": node_features},
            "node_labels": labels,
            "node_ids": node_ids,
        }


if __name__ == "__main__":
    import matplotlib.pyplot as plt
    import networkx as nx
    import numpy as np

    name_index = 0
    builder = Cora()
    builder.download_and_prepare(
        download_config=tfds.core.download.DownloadConfig(register_checksums=True)
    )
    print("Prepared cora")

    inputs, labels = tf.data.experimental.get_single_element(
        builder.as_dataset(as_supervised=True, split="train")
    )
    adj = inputs["adjacency"]
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
    print(f"cora has {len(G.nodes())} nodes and {len(G.edges())} edges")
    print(len(row_ids))
    pos = nx.spring_layout(G)
    nx.draw_networkx_nodes(G, pos, alpha=0.4, node_size=2)
    nx.draw_networkx_edges(G, pos, width=1.0, alpha=0.1)
    plt.show()
