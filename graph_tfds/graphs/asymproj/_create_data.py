import os
import pickle as pkl
from pathlib import Path

import numpy as np
import tensorflow as tf


def main():
    path = Path(os.path.dirname(__file__)) / "dummy_data/datasets"
    for name in (
        "ca-AstroPh",
        "ca-HepTh",
        "soc-epinions",
        "soc-facebook",
        "wiki-vote",
        "ppi",
    ):
        dirname = path / name
        tf.io.gfile.makedirs(dirname)

        for i, split in enumerate(["train", "test"]):
            np.save(dirname / f"{split}.txt.npy", [[0, i * 2]])
            np.save(dirname / f"{split}.neg.txt.npy", [[0, i * 2 + 1]])
        with tf.io.gfile.GFile(dirname / "index.pkl", "wb") as fp:
            pkl.dump({"largest_cc_num_nodes": 4}, fp)


if __name__ == "__main__":
    main()
