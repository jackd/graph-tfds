from typing import NamedTuple, Tuple

import numpy as np
import tensorflow as tf

import tensorflow_datasets as tfds


class SparseComponents(NamedTuple):
    indices: np.ndarray
    values: np.ndarray
    dense_shape: Tuple[int, ...]


class SparseTensor(tfds.core.features.FeatureConnector):
    def __init__(
        self, ndims: int, dtype: tf.DType = tf.float32,
    ):
        self._ndims = ndims
        self._dtype = dtype
        self._base = tfds.core.features.FeaturesDict(
            {
                "indices": tfds.core.features.Tensor(
                    shape=(None, ndims), dtype=tf.int64
                ),
                "values": tfds.core.features.Tensor(shape=(None,), dtype=dtype),
                "dense_shape": tfds.core.features.Tensor(
                    shape=(ndims,), dtype=tf.int64
                ),
            }
        )

    def encode_example(self, example: tf.SparseTensor):
        components = {
            "indices": example.indices,
            "values": example.values,
            "dense_shape": example.dense_shape,
        }
        return self._base.encode_example(components)

    def decode_example(self, tfexample_data):
        components = self._base.decode_example(tfexample_data)
        return tf.SparseTensor(
            components["indices"], components["values"], components["dense_shape"]
        )

    def get_serialized_info(self):
        return self._base.get_serialized_info()

    def get_tensor_info(self):
        return tfds.core.features.TensorInfo(
            shape=(None,) * self._ndims, dtype=self._dtype,
        )
