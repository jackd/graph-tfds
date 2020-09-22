from typing import Sequence

import tensorflow_datasets as tfds


class FeatureTuple(tfds.core.features.FeatureConnector, Sequence):
    """Tuple equivalent of `FeatureDict`."""

    def __init__(self, features: tfds.core.features.FeatureConnector):
        self._features = features

    def __getitem__(self, s):
        return self._features[s]

    def __len__(self):
        return len(self._features)

    def _assert_same_length(self, values):
        if len(values) != len(self):
            raise ValueError(f"values must have {len(self)} entries, got {len(values)}")

    def __flatten(self, x):
        return tuple(x[i] for i in range(len(self)))

    def __nest(self, x):
        return {i: x for i, x in enumerate(x)}

    def get_tensor_info(self):
        return self.__flatten(self.get_serialized_info())

    def get_serialized_info(self):
        return self.__nest(f.get_serialized_info() for f in self._features)

    def encode_example(self, example_data):
        self._assert_same_length(example_data)
        return self.__nest(
            f.encode_example(x) for f, x in zip(self._features, example_data)
        )

    def decode_example(self, tfexample_data):
        self._assert_same_length(tfexample_data)
        return tuple(
            f.decode_example(e)
            for f, e in zip(self._features, self.__flatten(tfexample_data))
        )
