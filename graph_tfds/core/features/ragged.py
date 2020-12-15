from typing import Dict

import tensorflow as tf
import tensorflow_datasets as tfds


class RaggedComponents(tfds.core.features.FeaturesDict):
    def __init__(
        self,
        flat_shape,
        dtype: tf.DType = tf.float32,
        ragged_rank: int = 1,
        row_splits_dtype: tf.DType = tf.int32,
    ):
        self._ragged_rank = ragged_rank
        self._values_dtype = dtype
        self._row_splits_dtype = row_splits_dtype
        rs = tfds.core.features.Tensor(shape=(None,), dtype=row_splits_dtype)
        super().__init__(
            {
                "flat_values": tfds.core.features.Tensor(shape=flat_shape, dtype=dtype),
                **{f"nested_row_splits_{i}": rs for i in range(ragged_rank)},
            }
        )

    def encode_example(self, example: tf.RaggedTensor):
        if example.ragged_rank != self._ragged_rank:
            raise ValueError(
                f"Expected ragged_rank {self._ragged_rank} but example has "
                f"{example.ragged_rank}"
            )
        components = {
            f"nested_row_splits_{i}": rs
            for i, rs in enumerate(example.nested_row_splits)
        }
        components["flat_values"] = example.flat_values
        return super().encode_example(components)


def pack_ragged_components(components_dict: Dict[str, tf.Tensor]) -> tf.RaggedTensor:
    ragged_rank = len(components_dict) - 1
    return tf.RaggedTensor.from_nested_row_splits(
        components_dict["flat_values"],
        [components_dict[f"nested_row_splits_{i}"] for i in range(ragged_rank)],
    )
