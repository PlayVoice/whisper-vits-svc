import os
import numpy as np

# average -> ave -> eva :haha

eva_conf = {
    './configs/singers/singer0022.npy': 0,
    './configs/singers/singer0030.npy': 0,
    './configs/singers/singer0047.npy': 0.5,
    './configs/singers/singer0051.npy': 0.5,
}

if __name__ == "__main__":

    eva = np.zeros(256)
    for k, v in eva_conf.items():
        assert os.path.isfile(k), k
        spk = np.load(k)
        eva = eva + spk * v
    np.save("eva.spk.npy", eva, allow_pickle=False)
