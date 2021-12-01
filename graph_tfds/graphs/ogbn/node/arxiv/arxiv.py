"""arxiv dataset."""

import tensorflow_datasets as tfds

_DESCRIPTION = """\
Graph: The ogbn-arxiv dataset is a directed graph, representing the citation network
between all Computer Science (CS) arXiv papers indexed by MAG [1]. Each node is an
arXiv paper and each directed edge indicates that one paper cites another one. Each
paper comes with a 128-dimensional feature vector obtained by averaging the embeddings
of words in its title and abstract. The embeddings of individual words are computed by
running the skip-gram model [2] over the MAG corpus. We also provide the mapping from
MAG paper IDs into the raw texts of titles and abstracts here. In addition, all papers
are also associated with the year that the corresponding paper was published.

Prediction task: The task is to predict the 40 subject areas of arXiv CS papers, e.g.,
cs.AI, cs.LG, and cs.OS, which are manually determined (i.e., labeled) by the paper's
authors and arXiv moderators. With the volume of scientific publications doubling every
12 years over the past century, it is practically important to automatically classify
each publication’s areas and topics. Formally, the task is to predict the primary
categories of the arXiv papers, which is formulated as a 40-class classification
problem.

Dataset splitting: We consider a realistic data split based on the publication dates of
the papers. The general setting is that the ML models are trained on existing papers
and then used to predict the subject areas of newly-published papers, which supports
the direct application of them into real-world scenarios, such as helping the arXiv
moderators. Specifically, we propose to train on papers published until 2017, validate
on those published in 2018, and test on those published since 2019.
"""

_CITATION = """\
@article{hu2020open,
  title={Open graph benchmark: Datasets for machine learning on graphs},
  author={Hu, Weihua and Fey, Matthias and Zitnik, Marinka and Dong, Yuxiao and Ren, Hongyu and Liu, Bowen and Catasta, Michele and Leskovec, Jure},
  journal={arXiv preprint arXiv:2005.00687},
  year={2020}
}
"""


class Arxiv(tfds.core.GeneratorBasedBuilder):
    """DatasetBuilder for arxiv dataset."""

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
                    # These are the features of your dataset like images, labels ...
                    "image": tfds.features.Image(shape=(None, None, 3)),
                    "label": tfds.features.ClassLabel(names=["no", "yes"]),
                }
            ),
            # If there's a common (input, target) tuple from the
            # features, specify them here. They'll be used if
            # `as_supervised=True` in `builder.as_dataset`.
            supervised_keys=("image", "label"),  # Set to `None` to disable
            homepage="https://ogb.stanford.edu/",
            citation=_CITATION,
        )

    def _split_generators(self, dl_manager: tfds.download.DownloadManager):
        """Returns SplitGenerators."""
        # TODO(arxiv): Downloads the data and defines the splits
        path = dl_manager.download_and_extract("https://todo-data-url")

        # TODO(arxiv): Returns the Dict[split names, Iterator[Key, Example]]
        return {
            "train": self._generate_examples(path / "train_imgs"),
        }

    def _generate_examples(self, path):
        """Yields examples."""
        # TODO(arxiv): Yields (key, example) tuples from the dataset
        for f in path.glob("*.jpeg"):
            yield "key", {
                "image": f,
                "label": "yes",
            }
