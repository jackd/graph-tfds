"""coauthor dataset."""

from graph_tfds.core.utils import pitfalls

_DESCRIPTION = """\
The Amazon Computers and Amazon Photo networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent authors that are connected by an edge if they co-authored a
paper.
Given paper keywords for each author's papers, the task is to map authors
to their respective field of study."""


CS = pitfalls.PitfallsConfig(
    name="cs", url_name="ms_academic_cs", description=_DESCRIPTION,
)

PHYSICS = pitfalls.PitfallsConfig(
    name="physics", url_name="ms_academic_phy", description=_DESCRIPTION,
)


class Coauthor(pitfalls.Pitfalls):
    """DatasetBuilder for Coauthor dataset."""

    BUILDER_CONFIGS = [CS, PHYSICS]


if __name__ == "__main__":
    import tensorflow_datasets as tfds

    dl_config = tfds.core.download.DownloadConfig(register_checksums=True)
    for config in Coauthor.BUILDER_CONFIGS:
        builder = Coauthor(config=config)
        builder.download_and_prepare(download_config=dl_config)
