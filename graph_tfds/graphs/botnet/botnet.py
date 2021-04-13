"""botnet dataset."""

import pickle

import h5py
import tensorflow as tf
import tensorflow_datasets as tfds

_DESCRIPTION = """\
A collection of different botnet topologyies overlaid onto normal background network
traffic, containing featureless graphs of relatively large scale for inductive
learning."""

_CITATION = """\
@article{zhou2020auto,
  title={Automating Botnet Detection with Graph Neural Networks},
  author={Jiawei Zhou*, Zhiying Xu*, Alexander M. Rush, and Minlan Yu},
  journal={AutoML for Networking and Systems Workshop of MLSys 2020 Conference},
  year={2020}
}
"""


class BotnetConfig(tfds.core.BuilderConfig):
    def __init__(self, name: str):
        super().__init__(
            name=name, description=f"{name} dataset",
        )


CHORD = tfds.core.BuilderConfig(name="chord")
DEBRU = tfds.core.BuilderConfig(name="debru")
KADEM = tfds.core.BuilderConfig(name="kadem")
LEET = tfds.core.BuilderConfig(name="leet")
C2 = tfds.core.BuilderConfig(name="c2")
P2P = tfds.core.BuilderConfig(name="p2p")


class Botnet(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for botnet dataset."""

    BUILDER_CONFIGS = [CHORD, DEBRU, KADEM, LEET, C2, P2P]
    VERSION = tfds.core.Version("0.0.1")
    RELEASE_NOTES = {"0.0.1": "Initial release."}

    def _info(self) -> tfds.core.DatasetInfo:
        """Returns the dataset metadata."""
        return tfds.core.DatasetInfo(
            builder=self,
            description=_DESCRIPTION,
            features=tfds.features.FeaturesDict(
                {
                    "num_evil_edges": tfds.core.features.Tensor(
                        shape=(), dtype=tf.int64
                    ),
                    "num_evils": tfds.core.features.Tensor(shape=(), dtype=tf.int64),
                    "idx": tfds.core.features.Tensor(shape=(), dtype=tf.int64),
                    "links": tfds.core.features.Tensor(shape=(None, 2), dtype=tf.int64),
                    "edge_labels": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.bool
                    ),
                    "node_labels": tfds.core.features.Tensor(
                        shape=(None,), dtype=tf.bool
                    ),
                }
            ),
            supervised_keys=None,
            homepage="https://github.com/harvardnlp/botnet-detection",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        name = self.builder_config.name
        path = dl_manager.download_and_extract(
            f"https://zenodo.org/record/3689089/files/botnet_{name}.tar.gz"
        )
        paths = {
            "split_path": path / f"{name}_split_idx.pkl",
            "data_path": path / f"{name}_raw.hdf5",
        }
        return {
            tfds.core.Split.TRAIN: self._generate_examples("train", **paths),
            tfds.core.Split.VALIDATION: self._generate_examples("val", **paths),
            tfds.core.Split.TEST: self._generate_examples("test", **paths),
        }

    def _generate_examples(self, split_id: str, split_path: str, data_path: str):
        """Yields examples."""
        with open(split_path, "rb") as fp:
            ids = pickle.load(fp)[split_id]
        raw_data = h5py.File(data_path, mode="r")
        for idx in ids:
            data = raw_data[str(idx)]
            example = {
                "num_evil_edges": data.attrs["num_evil_edges"],
                "num_evils": data.attrs["num_evils"],
                "idx": idx,
                "links": data["edge_index"][:].T,
                "edge_labels": data["edge_y"][:].astype(bool),
                "node_labels": data["y"][:].astype(bool),
            }
            yield idx, example


if __name__ == "__main__":
    dl_config = tfds.core.download.DownloadConfig(register_checksums=True)
    for i, config in enumerate(Botnet.BUILDER_CONFIGS):
        builder = Botnet(config=config)
        print("---------------")
        print(f"Preparing {config.name}, {i+1} / {len(Botnet.BUILDER_CONFIGS)}")
        builder.download_and_prepare(download_config=dl_config)
