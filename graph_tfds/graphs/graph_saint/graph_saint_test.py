"""graph_saint dataset."""

import tensorflow_datasets as tfds

from . import graph_saint


class GraphSaintTest(tfds.testing.DatasetBuilderTestCase):
    """Tests for graph_saint dataset."""

    # TODO(graph_saint):
    DATASET_CLASS = graph_saint.GraphSaint
    SPLITS = {
        "train": 3,  # Number of fake train example
        "test": 1,  # Number of fake test example
    }

    # If you are calling `download/download_and_extract` with a dict, like:
    #   dl_manager.download({'some_key': 'http://a.org/out.txt', ...})
    # then the tests needs to provide the fake output paths relative to the
    # fake data directory
    # DL_EXTRACT_RESULT = {'some_key': 'output_file1.txt', ...}


if __name__ == "__main__":
    tfds.testing.test_main()
