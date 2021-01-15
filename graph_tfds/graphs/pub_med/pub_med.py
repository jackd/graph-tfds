"""cora dataset."""
from graph_tfds.core.utils.dgl import citation_graph

NUM_CLASSES = 3
NUM_FEATURES = 500


class PubMed(citation_graph.CitationGraph):
    """DatasetBuilder for cora dataset."""

    _NAME = "pubmed"
    _URL_NAME = "pubmed"


if __name__ == "__main__":
    citation_graph._vis(PubMed())  # pylint: disable=protected-access
