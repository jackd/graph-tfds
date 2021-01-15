"""ppi dataset."""

import tensorflow_datasets as tfds

from graph_tfds.graphs.ppi import PPI


class PPITest(tfds.testing.DatasetBuilderTestCase):
    """Tests for ppi dataset."""

    DATASET_CLASS = PPI
    SPLITS = {
        "train": 5,
        "validation": 2,
        "test": 3,
    }

    DL_EXTRACT_RESULT = ""


if __name__ == "__main__":
    tfds.testing.test_main()
