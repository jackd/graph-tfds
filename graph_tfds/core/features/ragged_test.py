# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# Lint as: python3
"""Tests for tensorflow_graphics.datasets.features.voxel_feature."""

import numpy as np
import tensorflow as tf
from tensorflow.python.ops.ragged import (  # pylint: disable=no-name-in-module
    ragged_tensor,
)

import tensorflow_datasets as tfds
from graph_tfds.core.features import ragged


class RaggedTensorTest(tfds.testing.FeatureExpectationsTestCase):
    """Test Cases for RaggedTensor FeatureConnector."""

    def _assertRaggedEqual(self, a, b, msg):
        """Asserts that two ragged tensors are equal."""
        assert ragged_tensor.is_ragged(a)
        assert ragged_tensor.is_ragged(b)
        self.assertEqual(a.ragged_rank, b.ragged_rank)
        self.assertAllEqual(a.flat_values, b.flat_values, msg=msg)
        for a_rs, b_rs in zip(a.nested_row_splits, b.nested_row_splits):
            self.assertAllEqual(a_rs, b_rs, msg=msg)

    def test_ragged_tensor(self):
        """Tests ragged tensor encoding/decoding to DatasetFeature."""
        dtype = tf.float32
        row_splits_dtype = tf.int64

        def add_ragged_rank(values, nrows):
            size = (
                values.nrows()
                if isinstance(values, tf.RaggedTensor)
                else tf.shape(values)[0]
            )
            row_splits = rng.uniform((nrows + 1,))
            row_splits = tf.math.cumsum(row_splits, exclusive=True, axis=0)
            row_splits = tf.cast(
                row_splits / row_splits[-1] * tf.cast(size, tf.float32),
                row_splits_dtype,
            )
            return tf.RaggedTensor.from_row_splits(values, row_splits)

        def as_dict(rt, keys):
            def maybe_call(x):
                return x() if callable(x) else x

            return tf.nest.map_structure(
                lambda x: x.numpy(), {k: maybe_call(getattr(rt, k)) for k in keys}
            )

        def as_lists(rt):
            if rt.ragged_rank == 1:
                values = rt.values.numpy()
                splits = rt.row_splits.numpy()
            else:
                values = as_lists(rt.values)
                splits = rt.row_splits.numpy()
            return list(np.split(values, splits[1:-1]))

        def get_tests(rt):
            tests = [
                tfds.testing.FeatureExpectationItem(value=as_lists(rt), expected=rt)
            ]

            for keys in (
                ("values", "row_splits"),
                ("values", "row_lengths"),
                ("values", "value_rowids"),
                ("flat_values", "nested_row_splits"),
                ("flat_values", "nested_row_lengths"),
                ("flat_values", "nested_value_rowids"),
            ):
                tests.append(
                    tfds.testing.FeatureExpectationItem(
                        value=as_dict(rt, keys), expected=rt
                    )
                )
            return tests

        rng = tf.random.Generator.from_seed(0)
        flat_shape = (100, 3)
        values = rng.normal(flat_shape, dtype=dtype)
        r1 = add_ragged_rank(values, 10)
        r1_tests = get_tests(r1)
        self.assertFeature(
            feature=ragged.RaggedTensor(
                flat_shape=flat_shape,
                ragged_rank=1,
                dtype=dtype,
                row_splits_dtype=row_splits_dtype,
            ),
            shape=(None, None, *flat_shape[1:]),
            dtype=dtype,
            tests=r1_tests,
        )

        r2 = add_ragged_rank(r1, 3)
        r2_tests = get_tests(r2)
        self.assertFeature(
            feature=ragged.RaggedTensor(
                flat_shape=flat_shape,
                ragged_rank=2,
                dtype=dtype,
                row_splits_dtype=row_splits_dtype,
            ),
            shape=(None, None, None, *flat_shape[1:]),
            dtype=dtype,
            tests=r2_tests,
        )


if __name__ == "__main__":
    tfds.testing.test_main()
