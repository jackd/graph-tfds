import scipy.sparse as sp
import tensorflow as tf


def csr_to_ragged(csr_matrix: sp.csr_matrix, dtype=tf.int64, row_splits_dtype=tf.int64):
    assert csr_matrix.has_sorted_indices
    return tf.RaggedTensor.from_row_splits(
        tf.convert_to_tensor(csr_matrix.indices, dtype=dtype),
        tf.convert_to_tensor(csr_matrix.indptr, dtype=row_splits_dtype),
    )
