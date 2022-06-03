#!/usr/bin/env python3

from absl import flags, app

from loguru import logger

from pathlib import Path
import shutil
import functools

import tokenizers
import tensorflow as tf
import tqdm

flags.DEFINE_string('txt_fn', '../tmp/kek.txt', 'input txt')
flags.DEFINE_string('out_dir', '../tmp/tfrecords/', 'output directory (will be cleared)')
flags.DEFINE_integer('chunk_size', 2**24, 'how many tokens go into one tfrecords file')
flags.DEFINE_integer('read_buffer_size', 2**10, 'input file read buffer size')

FLAGS = flags.FLAGS


@functools.lru_cache(maxsize=1)
def get_tokenizer():
    return tokenizers.Tokenizer.from_pretrained('gpt2')


def make_record_file(record_ids, out_dir, file_no):
    out_fn = str(out_dir / f'tokens-{file_no:05d}.tfrecord')
    with tf.io.TFRecordWriter(out_fn) as writer:
        feature = {'text': tf.train.Feature(int64_list=tf.train.Int64List(value=record_ids))}
        tf_example = tf.train.Example(features=tf.train.Features(feature=feature))
        writer.write(tf_example.SerializeToString())


def read_in_blocks(f):
    while True:
        block = f.read(FLAGS.read_buffer_size)
        if not block:
            break
        yield block

def main(_):
    out_dir = Path(FLAGS.out_dir)
    if out_dir.exists():
        logger.warning(f'clearing {out_dir}')
        shutil.rmtree(out_dir)
    out_dir.mkdir(exist_ok=True)
    tokenizer = get_tokenizer()
    with open(FLAGS.txt_fn) as in_f:
        current_ids = []
        out_file_no = 0
        for block in tqdm.tqdm(read_in_blocks(in_f)):
            current_ids.extend(tokenizer.encode(block).ids)
            while len(current_ids) >= FLAGS.chunk_size:
                record_ids, current_ids = current_ids[:FLAGS.chunk_size], current_ids[FLAGS.chunk_size:]
                make_record_file(record_ids, out_dir, out_file_no)
                out_file_no += 1

        if current_ids:
            make_record_file(current_ids, out_dir, out_file_no)


if __name__ == "__main__":
    app.run(main)
