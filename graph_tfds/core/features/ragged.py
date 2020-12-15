import tensorflow as tf
import tensorflow_datasets as tfds


class RaggedTensor(tfds.core.features.FeatureConnector):
    def __init__(
        self,
        flat_shape,
        dtype: tf.DType = tf.float32,
        ragged_rank: int = 1,
        row_splits_dtype: tf.DType = tf.int64,
    ):
        self._ragged_rank = ragged_rank
        self._values_dtype = dtype
        self._row_splits_dtype = row_splits_dtype
        self._flat_shape = tuple(flat_shape)
        rs = tfds.core.features.Tensor(shape=(None,), dtype=row_splits_dtype)
        self._base = tfds.core.features.FeaturesDict(
            {
                "flat_values": tfds.core.features.Tensor(
                    shape=self._flat_shape, dtype=dtype
                ),
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
        return self._base.encode_example(components)

    def decode_example(self, tfexample_data):
        components_dict = self._base.decode_example(tfexample_data)
        return tf.RaggedTensor.from_nested_row_splits(
            components_dict["flat_values"],
            [
                components_dict[f"nested_row_splits_{i}"]
                for i in range(self._ragged_rank)
            ],
        )

    def get_serialized_info(self):
        return self._base.get_serialized_info()

    def get_tensor_info(self):
        return tfds.core.features.TensorInfo(
            shape=(None,) * self._ragged_rank + self._flat_shape,
            dtype=self._values_dtype,
        )
