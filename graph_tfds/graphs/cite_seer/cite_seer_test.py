"""cite_seer dataset."""

import tensorflow_datasets as tfds

from . import cite_seer


class CiteSeerTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for cite_seer dataset."""

    DATASET_CLASS = cite_seer.CiteSeer
    SPLITS = {"train": 1}
    DL_EXTRACT_RESULT = ""


if __name__ == "__main__":
    tfds.testing.test_main()
