from typing import List, Mapping, Sequence, Tuple

import numpy as np
import scipy.sparse as sp
import tensorflow as tf

from graph_tfds.core.utils.ragged import csr_to_ragged


def load_content(path: str, class_indices: Mapping[str, int]):
    """
    Load .content files.

    .content files (like used in cora) contain node features on separate lines. Each
    line is of the form
    ```
    node_id feature_0 feature_1 ... class_label
    ```
    where values are separated by tabs ("\t") and feature_i is 0 or 1.

    See `graphs/cora/cora.py` for example usage.

    Args:
        path: path to .content file.

    Returns:
        node_ids: list of node ids (str).
        features: list of lists of nonzero feature indices.
        labels: list[int] class indices corresponding to class names.
    """
    node_ids = []
    rows = []
    labels = []
    with tf.io.gfile.GFile(path, "r") as fp:
        for line in fp.readlines():
            node_id, *features, label = line.rstrip().split("\t")
            node_ids.append(node_id)
            labels.append(class_indices[label])
            (row,) = np.where([int(f) for f in features])
            rows.append(row)
    return node_ids, rows, labels


def load_cites(path: str, node_ids: Sequence[str]) -> Tuple[List[int], List[int]]:
    node_indices = {k: i for i, k in enumerate(node_ids)}
    src = []
    dst = []
    with tf.io.gfile.GFile(path, "r") as fp:
        for line in fp.readlines():
            dst_, src_ = line.rstrip().split("\t")
            if dst_ in node_indices and src_ in node_indices:
                dst.append(node_indices[dst_])
                src.append(node_indices[src_])

    return dst, src  # src -> dst


def load_content_and_cites(base_path: str, class_indices: Mapping[str, int]):
    node_ids, features, labels = load_content(f"{base_path}.content", class_indices)
    i, j = load_cites(f"{base_path}.cites", node_ids)
    data = np.ones((len(i),), dtype=np.int64)
    adjacency = sp.coo_matrix((data, (i, j))).tocsr()

    features = tf.ragged.constant(features, dtype=tf.int64)
    adjacency = csr_to_ragged(adjacency)
    labels = tf.convert_to_tensor(labels, dtype=tf.int64)
    return adjacency, features, labels, node_ids
