"""
Relational-GCN datasets.

Selected datasets from
"""

import os

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds

_DESCRIPTION = """
The dataset contains a graph depicting the connectivity of a knowledge base.

Currently, the knowledge bases from the
`RGCN paper <https://arxiv.org/pdf/1703.06103.pdf>`_ supported are

- FB15k
- FB15k-237
- wn18

Data is sourced from [DGL](https://github.com/dmlc/dgl).
"""

_CITATION = """@inproceedings{schlichtkrull2018modeling,
  title={Modeling relational data with graph convolutional networks},
  author={Schlichtkrull, Michael and Kipf, Thomas N and Bloem, Peter and Van Den Berg, Rianne and Titov, Ivan and Welling, Max},
  booktitle={European semantic web conference},
  pages={593--607},
  year={2018},
  organization={Springer}
}
"""

FB15K = tfds.core.BuilderConfig("FB15k")
FB15K_237 = tfds.core.BuilderConfig("FB15k-237")
WN18 = tfds.core.BuilderConfig("wn18")


def _read_dictionary(filename):
    d = {}
    with open(filename, "r+") as f:  # pylint: disable=unspecified-encoding
        for line in f:
            line = line.strip().split("\t")
            d[line[1]] = int(line[0])
    return d


def _read_triplets(filename):
    with open(filename, "r+") as f:  # pylint: disable=unspecified-encoding
        for line in f:
            processed_line = line.strip().split("\t")
            yield processed_line


def _read_triplets_as_list(filename, entity_dict, relation_dict):
    l = []
    for triplet in _read_triplets(filename):
        s = entity_dict[triplet[0]]
        r = relation_dict[triplet[1]]
        o = entity_dict[triplet[2]]
        l.append([s, r, o])
    return l


class Rgcn(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for rgcn dataset."""

    BUILDER_CONFIGS = [FB15K, FB15K_237, WN18]

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
                    "num_nodes": tfds.features.Tensor(shape=(), dtype=tf.int64),
                    "num_relations": tfds.features.Tensor(shape=(), dtype=tf.int64),
                    "relations": {
                        # src, rel, dst
                        "train": tfds.features.Tensor(shape=(None, 3), dtype=tf.int64),
                        "validation": tfds.features.Tensor(
                            shape=(None, 3), dtype=tf.int64
                        ),
                        "test": tfds.features.Tensor(shape=(None, 3), dtype=tf.int64),
                    },
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/tkipf/relational-gcn",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        path = dl_manager.download_and_extract(
            f"https://data.dgl.ai/dataset/{self.builder_config.name}.tgz"
        )
        return {
            "full": self._generate_examples(path),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        entity_path = os.path.join(path, "entities.dict")
        relation_path = os.path.join(path, "relations.dict")
        train_path = os.path.join(path, "train.txt")
        valid_path = os.path.join(path, "valid.txt")
        test_path = os.path.join(path, "test.txt")
        entity_dict = _read_dictionary(entity_path)
        relation_dict = _read_dictionary(relation_path)
        train = np.asarray(
            _read_triplets_as_list(train_path, entity_dict, relation_dict)
        )
        valid = np.asarray(
            _read_triplets_as_list(valid_path, entity_dict, relation_dict)
        )
        test = np.asarray(_read_triplets_as_list(test_path, entity_dict, relation_dict))

        yield 0, {
            "num_nodes": len(entity_dict),
            "num_relations": len(relation_dict),
            "relations": {"train": train, "validation": valid, "test": test},
        }
