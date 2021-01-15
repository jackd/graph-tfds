"""cora dataset."""
from graph_tfds.core.utils.dgl import citation_graph

CLASS_NAMES = (
    "Case_Based",
    "Genetic_Algorithms",
    "Neural_Networks",
    "Probabilistic_Methods",
    "Reinforcement_Learning",
    "Rule_Learning",
    "Theory",
)

NUM_FEATURES = 1433


class Cora(citation_graph.CitationGraph):
    """DatasetBuilder for cora dataset."""

    _NAME = "cora"
    _URL_NAME = "cora_v2"


if __name__ == "__main__":
    citation_graph._vis(Cora())  # pylint: disable=protected-access
