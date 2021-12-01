"""asymproj dataset."""

import tensorflow_datasets as tfds
from graph_tfds.graphs.asymproj import asymproj


class AsymprojTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for asymproj dataset."""

    # TODO(asymproj):
    DATASET_CLASS = asymproj.Asymproj
    SPLITS = {
        "full": 1,
    }
    # DL_EXTRACT_RESULT = {"http://sami.haija.org/graph/datasets.tgz": "datasets"}
    DL_EXTRACT_RESULT = {"data": "."}

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
