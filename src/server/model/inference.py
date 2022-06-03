import random
from typing import Any, Optional
from loguru import logger
import time

import jax
import numpy as np
import transformers
from jax import numpy as jnp
from jax.experimental import maps
from mesh_transformer.checkpoint import read_ckpt_lowmem
from mesh_transformer.sampling import nucleaus_sample
from mesh_transformer.transformer_shard import CausalTransformer

from .constants import ModelParams, InferConfig


def default(value: Any, fallback: Any) -> Any:
    # luke prefers making a function that chooses between `value` and `feedback` so i am gonna keep it
    if value is None:
        return fallback

    return value


_cores_per_replica = ModelParams.cores_per_replica
_mesh_shape = (jax.device_count() // _cores_per_replica, _cores_per_replica)
_devices = np.array(jax.devices()).reshape(_mesh_shape)

#maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(_devices, ("dp", "mp")), ())
maps.thread_resources.env = maps.ResourceEnv(maps.Mesh(_devices, ("dp", "mp")))


class Inference:
    _NP_ONE = np.ones((1,))

    def __init__(
            self,
            path: Optional[str] = None,
            parameters: Optional[ModelParams] = None,
            config: Optional[InferConfig] = None,
    ):
        path = "checkpoint_slim/" if path is None else path

        self.params = ModelParams() if parameters is None else parameters
        self.params.sampler = nucleaus_sample
        self.config = InferConfig() if config is None else config

        self.model = CausalTransformer(self.params.__dict__)
        self.tokenizer = transformers.GPT2TokenizerFast.from_pretrained("gpt2")
        self.model.state = read_ckpt_lowmem(
            self.model.state, path, self.params.cores_per_replica, load_opt=False
        )

    def generate_tokens(
            self,
            prompt: np.ndarray,
            length: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
    ) -> np.ndarray:
        length = default(length, self.config.token_length)
        top_p = default(top_p, self.config.top_p)
        new_temp = random.random() * (self.config.max_temperature - self.config.min_temperature)
        new_temp += self.config.min_temperature
        temperature = default(temperature, new_temp)
        #prompt = prompt[:, -2048:]
        #prompt = prompt[:, -length:]

        start_time = time.time()
        source = jnp.array(
            np.pad(
                prompt,
                (
                    (0, 0),
                    (self.params.seq - prompt.shape[1], 0),
                ),
            )
        )
        logger.info(f"creating source took {time.time() - start_time}")
        sampler_options = {
            "top_p": self._NP_ONE * top_p,
            "temp": self._NP_ONE * temperature,
        }

        start_time = time.time()
        #with jax.experimental.maps.mesh(_devices, ("dp", "mp")):
        logger.info(f"creating mesh took {time.time() - start_time}")
        start_time = time.time()
        out = self.model.generate(
            source, self._NP_ONE * prompt.shape[1], length, sampler_options
        )
        logger.info(f"generate took {time.time() - start_time}")

        #import IPython; IPython.embed()
        return out[1][0][0, :, 0]

    def generate(
            self,
            prompt: str,
            length: Optional[int] = None,
            top_p: Optional[float] = None,
            temperature: Optional[float] = None,
    ) -> str:
        inp_tokens = self.tokenizer([prompt], verbose=False, return_tensors="np")
        inp_tokens = inp_tokens["input_ids"][0]
        out_tokens = self.generate_tokens(
            inp_tokens.reshape(1, -1), length, top_p, temperature
        )

        return self.tokenizer.decode(out_tokens)
