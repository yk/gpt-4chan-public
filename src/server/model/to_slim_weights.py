import argparse
import json
import time

import jax
import numpy as np
import optax

from mesh_transformer import util
from mesh_transformer.checkpoint import read_ckpt, write_ckpt, read_ckpt_lowmem
from mesh_transformer.transformer_shard import CausalTransformer
from smart_open import open

from mesh_transformer.util import clip_by_global_norm, to_bf16, to_f16
from model.constants import ModelParams


if __name__ == "__main__":
    params = ModelParams().__dict__
    convert_fn = to_bf16

    cores_per_replica = params["cores_per_replica"]

    assert cores_per_replica <= 8

    start = time.time()
    print(f"jax devices: {jax.device_count()}")
    print(f"jax runtime initialized in {time.time() - start:.06}s")

    mesh_shape = (jax.device_count() // cores_per_replica, cores_per_replica)
    devices = np.array(jax.devices()).reshape(mesh_shape)

    with jax.experimental.maps.mesh(devices, ("dp", "mp")):
        network = CausalTransformer(params)

        start = time.time()
        network.state = read_ckpt(
            network.state, f"checkpoint/", devices.shape[1], load_opt=False
        )
        print(f"network loaded in {time.time() - start:.06}s")

        start = time.time()
        del network.state["opt_state"]

        network.state["params"] = convert_fn(network.state["params"])
        print(f"network converted in {time.time() - start:.06}s")

        suffix = "_slim"

        for i in range(cores_per_replica):
            write_ckpt(network.state, f"checkpoint_slim/", i)
            print(f"written shard {i}")
