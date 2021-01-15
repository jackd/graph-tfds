"""graph_sage dataset."""
import json
import os
from typing import Dict, List, Optional

import networkx as nx
import numpy as np
import scipy.sparse as sp
import tensorflow as tf
import tensorflow_datasets as tfds

_REDDIT_DESCRIPTION = """
Reddit dataset for community detection (node classification) as used in GraphSage.

This is a graph dataset from Reddit posts made in the month of September, 2014.
The node label in this case is the community, or “subreddit”, that a post belongs to.
The authors sampled 50 large communities and built a post-to-post graph, connecting
posts if the same user comments on both. In total this dataset contains 232,965
posts with an average degree of 492.

Statistics

- Nodes: 232,965
- Edges: 114,615,892
- Node feature size: 602
- Number of training samples: 153,431
- Number of validation samples: 23,831
- Number of test samples: 55,703
"""

_PPI_DESCRIPTION = """

"""

_CITATION = """\
@inproceedings{hamilton2017inductive,
  author = {Hamilton, William L. and Ying, Rex and Leskovec, Jure},
  title = {Inductive Representation Learning on Large Graphs},
  booktitle = {NIPS},
  year = {2017}
}
"""


class GraphSageConfig(tfds.core.BuilderConfig):
    def __init__(
        self,
        name: str,
        num_features: int,
        num_classes: int,
        description: str,
        multiclass: bool,
    ):
        self.num_features = num_features
        self.num_classes = num_classes
        self.multiclass = multiclass
        super().__init__(
            name=name,
            version=tfds.core.Version("1.0.0"),
            release_notes={"1.0.0": "Initial release."},
            description=description,
        )


REDDIT = GraphSageConfig(
    name="reddit",
    description=_REDDIT_DESCRIPTION,
    num_features=602,
    num_classes=41,
    multiclass=False,
)
PPI = GraphSageConfig(
    name="ppi",
    description=_PPI_DESCRIPTION,
    num_features=50,
    num_classes=121,
    multiclass=True,
)


def node_link_graph(
    nodes: List[Dict],
    links: List[Dict],
    directed=False,
    graph: Optional[Dict] = None,
    multigraph: bool = False,
) -> nx.Graph:
    if graph is None:
        graph = {}
    if multigraph:
        raise NotImplementedError()
    G = nx.DiGraph(**graph) if directed else nx.Graph(**graph)
    G.add_nodes_from(enumerate(nodes))
    for edge in links:
        G.add_edge(edge["source"], edge["target"])
    return G


class GraphSage(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for graph_sage dataset."""

    BUILDER_CONFIGS = [
        REDDIT,
        PPI,
    ]

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        if self.builder_config.multiclass:
            node_labels = tfds.core.features.Tensor(
                shape=(None, self.builder_config.num_classes), dtype=tf.bool
            )
        else:
            node_labels = tfds.core.features.Tensor(shape=(None,), dtype=tf.int64)
        return tfds.core.DatasetInfo(
            builder=self,
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
                    "node_labels": node_labels,
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
        url = f"http://snap.stanford.edu/graphsage/{self.builder_config.name}.zip"
        path = dl_manager.download_and_extract(url)

        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""

        name = self.builder_config.name
        prefix = os.path.join(path, name, name)

        with tf.io.gfile.GFile(prefix + "-G.json", "rb") as fp:
            # G_data = json.load(fp)
            G = node_link_graph(**json.load(fp))
        num_nodes = G.number_of_nodes()
        # Get train/val/test indices

        validation_ids = np.array(
            [node for node in G.nodes() if G.nodes[node]["val"]], dtype=np.int64
        )
        test_ids = np.array(
            [node for node in G.nodes() if G.nodes[node]["test"]], dtype=np.int64
        )
        train_mask = np.ones((num_nodes,), dtype=np.bool)
        train_mask[validation_ids] = False
        train_mask[test_ids] = False
        train_ids = np.where(train_mask)[0].astype(np.int64)

        # get edges
        edges = np.array(G.edges(), dtype=np.int64)

        with tf.io.gfile.GFile(prefix + "-id_map.json", "rb") as fp:
            id_map = json.load(fp)
        num_labels = len(id_map)
        gather_indices = [
            id_map[str(G.nodes[node]["id"])] for node in G.nodes
        ]  # for labels and feats
        del G
        assert len(edges.shape) == 2
        adj = sp.csr_matrix(
            (np.ones((edges.shape[0]), dtype=np.float32), (edges[:, 0], edges[:, 1])),
            shape=(num_nodes, num_nodes),
        )
        adj = adj.maximum(adj.transpose())
        adj = adj.tocoo()
        edges = np.stack((adj.row, adj.col), axis=-1).astype(np.int64)
        del adj

        assert len(set(gather_indices)) == len(gather_indices)

        # labels
        with tf.io.gfile.GFile(prefix + "-class_map.json", "rb") as fp:
            labels_json = json.load(fp)
        changed = np.zeros((num_labels,), dtype=np.bool)

        if self.builder_config.multiclass:
            labels = np.empty(
                (num_labels, self.builder_config.num_classes), dtype=np.bool
            )
        else:
            labels = np.empty((num_labels,), dtype=np.int64)

        if name == "reddit":
            for k, v in labels_json.items():
                k = id_map[k]
                labels[k] = v
                changed[k] = True
        elif name == "ppi":
            for k, v in labels_json.items():
                k = int(k)
                labels[k] = v
                changed[k] = v
        else:
            raise ValueError(
                f"Invalid builder_config.name {name} - must be one of 'reddit' or 'ppi'"
            )
        assert np.all(changed)
        del labels_json, changed
        labels = labels[gather_indices]

        with tf.io.gfile.GFile(prefix + "-feats.npy", "rb") as fp:
            node_features = np.load(fp).astype(np.float32)
        node_features = node_features[gather_indices]

        data = dict(
            graph=dict(links=edges, node_features=node_features),
            node_labels=labels,
            train_ids=train_ids,
            validation_ids=validation_ids,
            test_ids=test_ids,
        )

        yield 0, data


if __name__ == "__main__":

    config = tfds.core.download.DownloadConfig(register_checksums=True)
    for builder_config in GraphSage.BUILDER_CONFIGS:
        builder = GraphSage(config=builder_config)
        builder.download_and_prepare(download_config=config)
