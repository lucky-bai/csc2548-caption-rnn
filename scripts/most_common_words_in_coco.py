# Script to get the N most common words in COCO
import os
import ujson
import spacy
from collections import Counter

N_WORDS = 10000

COCO_JSON_FILE = '../../annotations/captions_train2014.json'

nlp = spacy.load('en_core_web_md')

with open(COCO_JSON_FILE) as f:
  data = ujson.load(f)

captions = [x['caption'] for x in data['annotations']]

words = []
for caption in captions:
  tokens = nlp(caption, disable = ['tagger', 'ner', 'parser'])
  for token in tokens:
    text = token.text.lower()
    words.append(text)

    if len(words) % 1000 == 0:
      print(len(words))

ctr = Counter(words)

with open('../coco_words.txt', 'w') as f:
  for x, _ in ctr.most_common(N_WORDS):
    f.write(x + '\n')
