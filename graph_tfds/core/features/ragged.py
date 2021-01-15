import itertools
from collections import namedtuple

import numpy as np
import tensorflow as tf
import tensorflow_datasets as tfds

RaggedComponents = namedtuple("RaggedComponents", ["flat_values", "nested_row_splits"])


def _lengths_to_splits(row_lengths):
    out = np.empty((len(row_lengths) + 1,), dtype=row_lengths.dtype)
    np.cumsum(row_lengths, out=out[1:])
    out[0] = 0
    return out


def _ids_to_lengths(value_rowids, nrows=None):
    return np.bincount(value_rowids, minlength=nrows or 0)


def _ids_to_splits(value_rowids, nrows=None):
    return _lengths_to_splits(_ids_to_lengths(value_rowids, nrows))


def _dict_to_components(example, ragged_rank, dtype, row_splits_dtype):
    """
  Convert a dict to RaggedComponents.

  dict must have keys:
    - "values" and one of ("row_splits", "row_lengths", "value_rowids"); or
    - "flat_values" and onf of ("nested_row_splits", "nested_row_lengths",
        "nested_value_rowids")
  """
    if "values" in example:
        if "flat_values" in example:
            raise ValueError("Only one of 'values', 'flat_values' can be provided.")
        values = example["values"]
        if ragged_rank == 1:
            out = RaggedComponents(np.array(values, dtype=dtype), [])
        else:
            out = _to_components(values, ragged_rank - 1, dtype, row_splits_dtype)
        if "row_splits" in example:
            row_splits = np.array(example["row_splits"], dtype=row_splits_dtype)
        elif "row_lengths" in example:
            row_splits = _lengths_to_splits(
                np.array(example["row_lengths"], dtype=row_splits_dtype)
            )
        elif "value_rowids" in example:
            row_splits = _ids_to_splits(example["value_rowids"], example.get("nrows"))
        else:
            raise ValueError(
                "Missing key: if 'values' is provided, one of "
                "'row_splits', 'row_lengths', 'value_rowids' "
                " also required, got {}".format(tuple(example.keys()))
            )
        out.nested_row_splits.insert(0, row_splits)
        return out
    if "flat_values" in example:
        flat_values = example["flat_values"]
        if "nested_row_splits" in example:
            nested_row_splits = example["nested_row_splits"]
        elif "nested_row_lengths" in example:
            nested_row_splits = [
                _lengths_to_splits(rl) for rl in example["nested_row_lengths"]
            ]
        elif "nested_value_rowids" in example:
            nested_value_rowids = example["nested_value_rowids"]
            if "nested_nrows" in example:
                nested_nrows = example["nested_nrows"]
            else:
                nested_nrows = [None for _ in nested_value_rowids]
            nested_row_splits = [
                _ids_to_splits(*args) for args in zip(nested_value_rowids, nested_nrows)
            ]
        else:
            raise ValueError(
                "Missing key: if 'flat_values' is provided, one of "
                "'nested_row_splits', 'nested_row_lengths', 'nested_value_rowids' "
                " also required, got {}".format(tuple(example.keys()))
            )
        assert len(nested_row_splits) == ragged_rank
        return RaggedComponents(flat_values, nested_row_splits)
    raise ValueError("Missing key: one of 'values', 'flat_values' is required.")


def _lists_to_components(lists, ragged_rank, dtype, row_splits_dtype):
    assert ragged_rank >= 0
    if ragged_rank == 0:
        return RaggedComponents(np.array(lists, dtype=dtype), [])

    components = _lists_to_components(
        list(itertools.chain(*lists)), ragged_rank - 1, dtype, row_splits_dtype
    )
    row_lengths = np.array([len(li) for li in lists], dtype=row_splits_dtype)
    components.nested_row_splits.insert(0, _lengths_to_splits(row_lengths))
    return components


def _rt_to_components(example, ragged_rank, dtype, row_splits_dtype):
    assert example.ragged_rank == ragged_rank
    assert example.flat_values.dtype == dtype
    assert all(rs.dtype == row_splits_dtype for rs in example.nested_row_splits)
    return RaggedComponents(example.flat_values, example.nested_row_splits)


def _to_components(example, ragged_rank, dtype, row_splits_dtype):
    if hasattr(dtype, "as_numpy_dtype"):
        dtype = dtype.as_numpy_dtype
    if hasattr(row_splits_dtype, "as_numpy_dtype"):
        row_splits_dtype = row_splits_dtype.as_numpy_dtype

    if isinstance(example, RaggedComponents):
        assert example.flat_values.dtype == dtype
        assert all(rs.dtype == row_splits_dtype for rs in example.nested_row_splits)
        return example
    if isinstance(example, dict):
        return _dict_to_components(example, ragged_rank, dtype, row_splits_dtype)
    if isinstance(example, tf.RaggedTensor):
        return _rt_to_components(example, ragged_rank, dtype, row_splits_dtype)
    if isinstance(example, (list, tuple, np.ndarray)):
        return _lists_to_components(example, ragged_rank, dtype, row_splits_dtype)

    raise ValueError(
        "Invalid example type - must be a "
        "dict, tf.RaggedTensor, list or tuple, "
        "got {}".format(example)
    )


class RaggedTensor(tfds.core.features.FeatureConnector):
    def __init__(
        self, flat_shape, ragged_rank=1, dtype=tf.float32, row_splits_dtype=tf.int64,
    ):
        self._ragged_rank = ragged_rank
        self._dtype = tf.dtypes.as_dtype(dtype)
        self._row_splits_dtype = tf.dtypes.as_dtype(row_splits_dtype)
        self._flat_shape = tuple(flat_shape)
        rs = tfds.core.features.Tensor(shape=(None,), dtype=self._row_splits_dtype)
        self._features = {
            "flat_values": tfds.core.features.Tensor(
                shape=self._flat_shape, dtype=self._dtype
            ),
            **{"nested_row_splits_{}".format(i): rs for i in range(ragged_rank)},
        }

    def to_json_content(self):
        return dict(
            flat_shape=self._flat_shape,
            ragged_rank=self._ragged_rank,
            row_splits_dtype=self._row_splits_dtype.name,
            dtype=self._dtype.name,
        )

    def encode_example(self, example):
        components = _to_components(
            example, self._ragged_rank, self._dtype, self._row_splits_dtype
        )
        components_dict = {
            "nested_row_splits_{}".format(i): rs
            for i, rs in enumerate(components.nested_row_splits)
        }
        components_dict["flat_values"] = components.flat_values
        return {
            k: v.encode_example(components_dict[k]) for k, v in self._features.items()
        }

    def decode_example(self, tfexample_data):
        components_dict = {
            k: v.decode_example(tfexample_data[k]) for k, v in self._features.items()
        }
        return tf.RaggedTensor.from_nested_row_splits(
            components_dict["flat_values"],
            [
                components_dict["nested_row_splits_{}".format(i)]
                for i in range(self._ragged_rank)
            ],
        )

    def get_serialized_info(self):
        return {k: v.get_serialized_info() for k, v in self._features.items()}

    def get_tensor_info(self):
        return tfds.core.features.TensorInfo(
            shape=(None,) * (self._ragged_rank + 1) + self._flat_shape[1:],
            dtype=self._dtype,
            sequence_rank=self._ragged_rank,
        )
