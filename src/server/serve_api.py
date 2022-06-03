#!/usr/bin/env python3

from typing import Optional
import threading
import queue
import time
from loguru import logger
from pathlib import Path
import contextlib

import pydantic
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI()

origins = ["*"]

app.add_middleware(
        CORSMiddleware,
        allow_origins=origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
        )


class Settings(pydantic.BaseSettings):
    queue_size: int = 1024
    log_file: str = "logs/serve_api.log"
    api_keys_file: str = 'valid_api_keys.txt'
    hf_model: str = ''
    hf_cuda: bool = False
    pre_prompt_length: int = 512


settings = Settings()

def _check_api_key(key):
    key = key.strip()
    for line in Path(settings.api_keys_file).open():
        if not line:
            continue
        valid_key = line.split()[0]
        if key == valid_key:
            break
    else:
        return False
    return True

request_queue = queue.Queue(maxsize=settings.queue_size)

@contextlib.contextmanager
def jax_generation():
    from model import inference
    import jax
    model = inference.Inference(path="../model_slim/step_88001/")

    def _generate(request):
        response = model.generate(
                prompt=request.prompt,
                length=request.length,
                top_p=request.top_p,
                temperature=request.temperature,
                )
        return response
    with jax.experimental.maps.mesh(inference._devices, ("dp", "mp")):
        yield _generate

@contextlib.contextmanager
def hf_generation():
    from transformers import GPTJForCausalLM, AutoTokenizer
    import torch

    if settings.hf_cuda:
        model = GPTJForCausalLM.from_pretrained(
            settings.hf_model, revision="float16", torch_dtype=torch.float16, low_cpu_mem_usage=True
        )
        model.cuda()
    else:
        model = GPTJForCausalLM.from_pretrained( settings.hf_model, torch_dtype=torch.float32)
    model.eval()

    tokenizer = AutoTokenizer.from_pretrained("EleutherAI/gpt-j-6B")

    def _generate(request: CompleteRequest):
        input_ids = tokenizer(request.prompt, return_tensors="pt").input_ids

        max_prompt_length = 2048 - request.length
        input_ids = input_ids[:, -max_prompt_length:]

        if request.pre_prompt:
            pp_input_ids = tokenizer(request.pre_prompt, return_tensors="pt").input_ids
            pp_input_ids = pp_input_ids[:, :settings.pre_prompt_length]
            input_ids = input_ids[:, -(max_prompt_length-len(pp_input_ids)):]
            full_prompt = tokenizer.batch_decode(pp_input_ids)[0] + tokenizer.batch_decode(input_ids)[0]
            input_ids = tokenizer(full_prompt, return_tensors="pt").input_ids
            input_ids = input_ids[:, -max_prompt_length:]


        if settings.hf_cuda:
            input_ids = input_ids.cuda()

        with torch.no_grad():
            gen_tokens = model.generate(
                input_ids,
                do_sample=True,
                temperature=request.temperature,
                top_p=request.top_p,
                typical_p=request.typical_p,
                max_new_tokens=request.length,
            ).detach().cpu()
            gen_text = tokenizer.batch_decode(gen_tokens)[0]
            prompt_decoded = tokenizer.batch_decode(input_ids.detach().cpu())[0]
        if not gen_text.startswith(prompt_decoded):
            raise Exception(f"Generated text does not start with prompt: {gen_text}\n(prompt was {prompt_decoded})")
        gen_text = gen_text[len(prompt_decoded):]
        return gen_text
    yield _generate

def worker():
    if settings.hf_model:
        generation = hf_generation
    else:
        generation = jax_generation
    with generation() as generate_fn:
        with open(settings.log_file, "a") as logf:
            while True:
                response_queue = None
                try:
                    start_time = time.time()
                    (request, response_queue) = request_queue.get()
                    logger.info(f"getting request took {time.time() - start_time}")
                    start_time = time.time()
                    response = generate_fn(request)
                    logger.info(f"generate took {time.time() - start_time}, response length: {len(response)}")
                    start_time = time.time()

                    logf.write(f"##### {request.api_key} ##### {time.time()} #####\n")
                    logf.write(f"{request.pre_prompt}\n")
                    logf.write("###\n")
                    logf.write(f"{request.prompt}\n")
                    logf.write("#####\n")
                    logf.write(f"{response}\n\n")
                    logf.flush()

                    logger.info(f"writing log took {time.time() - start_time}")
                    start_time = time.time()
                    response_queue.put(response)
                    logger.info(f"putting response took {time.time() - start_time}")
                except KeyboardInterrupt:
                    logger.info(f"Got KeyboardInterrupt... quitting!")
                    raise
                except Exception:
                    logger.exception(f"Got exception, will continue")
                    if response_queue is not None:
                        response_queue.put("")



@app.get("/")
async def main():
    return {"response": "Hello, world!"}

class CompleteRequest(pydantic.BaseModel):
    prompt: pydantic.constr(min_length=0, max_length=2**14)
    pre_prompt: pydantic.constr(min_length=0, max_length=2**14) = ''
    api_key: pydantic.constr(min_length=1, max_length=128) = "x"*9
    length: pydantic.conint(ge=1, le=1024) = 128
    top_p: pydantic.confloat(ge=0.0, le=1.0) = 1.0
    temperature: pydantic.confloat(ge=0.0) = 1.0
    typical_p: pydantic.confloat(ge=0.0, le=1.0) = 1.0

def _enqueue(request: CompleteRequest):
    response_queue = queue.Queue()
    request_queue.put((request, response_queue))
    response = response_queue.get()
    return response


@app.on_event("startup")
def startup():
    threading.Thread(
            target=worker,
            daemon=True,
            ).start()
    _enqueue(CompleteRequest(prompt="hello"))

@app.post("/complete")
def complete(request: CompleteRequest):
    logger.info(f"Received request from key {request.api_key}. Queue size is {request_queue.qsize()}")
    if request_queue.full():
        logger.warning("Request queue full.")
        raise ValueError("Request queue full.")
    if not _check_api_key(request.api_key):
        logger.warning(f"api key not valid: {request.api_key}, discarding...")
        raise ValueError("Invalid API key")
    response = _enqueue(request)
    return {"response": response}
