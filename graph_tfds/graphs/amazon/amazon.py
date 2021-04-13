"""amazon dataset."""

from graph_tfds.core.utils import pitfalls

_DESCRIPTION = """\
The Amazon Computers and Amazon Photo networks from the
`"Pitfalls of Graph Neural Network Evaluation"
<https://arxiv.org/abs/1811.05868>`_ paper.
Nodes represent goods and edges represent that two goods are frequently
bought together.
Given product reviews as bag-of-words node features, the task is to
map goods to their respective product category.
"""

COMPUTERS = pitfalls.PitfallsConfig(
    name="computers", url_name="amazon_electronics_computers", description=_DESCRIPTION,
)

PHOTO = pitfalls.PitfallsConfig(
    name="photo", url_name="amazon_electronics_photo", description=_DESCRIPTION,
)


class Amazon(pitfalls.Pitfalls):
    """DatasetBuilder for amazon dataset."""

    BUILDER_CONFIGS = [COMPUTERS, PHOTO]


if __name__ == "__main__":
    import tensorflow_datasets as tfds

    dl_config = tfds.core.download.DownloadConfig(register_checksums=True)
    for config in Amazon.BUILDER_CONFIGS:
        builder = Amazon(config=config)
        builder.download_and_prepare(download_config=dl_config)
