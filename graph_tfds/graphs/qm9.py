import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds
from dpu_utils.utils import RichPath

from graph_tfds.core.features.feature_tuple import FeatureTuple

CITATION = """\
@article{ramakrishnan2014quantum,
  title={Quantum chemistry structures and properties of 134 kilo molecules},
  author={Ramakrishnan, Raghunathan and Dral, Pavlo O and Rupp, Matthias and Von Lilienfeld, O Anatole},
  journal={Scientific data},
  volume={1},
  pages={140022},
  year={2014},
  publisher={Nature Publishing Group}
}
"""

NUM_TARGETS = 13
NUM_FEATURES = 15
NUM_EDGE_TYPES = 4


class Qm9(tfds.core.GeneratorBasedBuilder):
    VERSION = tfds.core.Version("0.0.1")

    def _info(self):
        link_features = tfds.core.features.Tensor(shape=(None, 2), dtype=tf.int64)
        return tfds.core.DatasetInfo(
            builder=self,
            features=tfds.core.features.FeaturesDict(
                dict(
                    graph=dict(
                        node_features=tfds.core.features.Tensor(
                            shape=(None, NUM_FEATURES), dtype=tf.float32
                        ),
                        links=FeatureTuple((link_features,) * NUM_EDGE_TYPES),
                    ),
                    targets=tfds.core.features.Tensor(
                        shape=(NUM_TARGETS,), dtype=tf.float32
                    ),
                    example_id=tfds.core.features.Tensor(shape=(), dtype=tf.int64),
                )
            ),
            supervised_keys=("graph", "targets"),
            citation=CITATION,
        )

    def _split_generators(self, dl_manager):
        # xyz_url = "https://ndownloader.figshare.com/files/3195389"
        url = "https://github.com/microsoft/tf-gnn-samples/raw/master/data/qm9/{split}.jsonl.gz"
        urls = dl_manager.download(
            {k: url.format(split=k) for k in ("train", "valid", "test")}
        )
        return [
            tfds.core.SplitGenerator(
                name=tfds.Split.TRAIN, gen_kwargs=dict(path=urls["train"])
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.VALIDATION, gen_kwargs=dict(path=urls["valid"]),
            ),
            tfds.core.SplitGenerator(
                name=tfds.Split.TEST, gen_kwargs=dict(path=urls["test"])
            ),
        ]

    def _generate_examples(self, path: str):
        # tfds mangles path, obscures extension
        data = RichPath.create(path).read_as_jsonl()
        for d in data:
            src, edge_id, dst = np.array(  # pylint:disable=unpacking-non-sequence
                d["graph"], dtype=np.int64
            ).T
            edge_id -= 1
            links = tf.dynamic_partition(
                np.stack((src, dst), axis=-1), edge_id, num_partitions=NUM_EDGE_TYPES
            )
            node_features = np.array(d["node_features"], dtype=np.float32)
            graph = dict(node_features=node_features, links=links)
            targets = np.squeeze(np.array(d["targets"], dtype=np.float32), axis=-1)
            example_id = int(d["id"][4:])
            yield example_id, dict(graph=graph, targets=targets, example_id=example_id)


QM9 = Qm9  # hack naming issues

if __name__ == "__main__":
    import graph_tfds  # register checksums dir
    import numpy as np

    config = None
    ppi = QM9()
    ppi.download_and_prepare(download_config=config)

    dataset = ppi.as_dataset(split="train", as_supervised=True)
    node_sizes = []
    for inputs, labels in dataset:
        nodes = inputs["node_features"]
        node_sizes.append(nodes.shape[0])

    print("Node size stats:")
    print(f"  mean: {np.mean(node_sizes)}")
    print(f"  std : {np.std(node_sizes)}")
