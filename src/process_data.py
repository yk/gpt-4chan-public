#!/usr/bin/env python3

import json
import bs4
from loguru import logger
import multiprocessing as mp
import tqdm
from absl import app, flags

import warnings
warnings.filterwarnings("ignore", category=bs4.MarkupResemblesLocatorWarning, module='bs4')

DATA_FN = '../tmp/pol_062016-112019_labeled.ndjson'
OUT_FN = '../tmp/kek.txt'

flags.DEFINE_string('data_fn', DATA_FN, 'data file')
flags.DEFINE_string('out_fn', OUT_FN, 'output file')

FLAGS = flags.FLAGS

# from here: https://gist.github.com/zmwangx/ad0830ba94b1fd98f428
def text_with_newlines(elem):
    text = ''
    for e in elem.descendants:
        if isinstance(e, str):
            # text += e.strip()
            text += e
        elif e.name == 'br' or e.name == 'p':
            text += '\n'
    return text
 

def parse_line(line):
    data = json.loads(line)
    posts_text = []
    for post in data.get('posts', []):
        try:
            if 'com' in post:
                soup = bs4.BeautifulSoup(post['com'], 'lxml')
                post_text = text_with_newlines(soup).strip()
            else:
                post_text = ''
            post_text = f'--- {post["no"]}\n{post_text}'
            posts_text.append(post_text)
        except Exception:
            logger.exception(f'failed to parse post {post}')
    return '\n'.join(posts_text)


def main(_):
    with open(FLAGS.out_fn, 'w') as out_f:
        with open(FLAGS.data_fn) as in_f:
            with mp.Pool() as pool:
                for parsed_line in pool.imap(parse_line, tqdm.tqdm(in_f)):
                    out_f.write(parsed_line + '\n-----\n')


if __name__ == '__main__':
    app.run(main)
