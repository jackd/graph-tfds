"""cora dataset."""
from graph_tfds.core.utils.dgl import citation_graph

CLASS_NAMES = (
    "Agents",
    "AI",
    "DB",
    "IR",
    "ML",
    "HCI",
)


NUM_FEATURES = 3703


class CiteSeer(citation_graph.CitationGraph):
    """DatasetBuilder for cora dataset."""

    _NAME = "citeseer"
    _URL_NAME = "citeseer"


if __name__ == "__main__":
    citation_graph._vis(CiteSeer())  # pylint: disable=protected-access
