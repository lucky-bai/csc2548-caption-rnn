# Compute KL divergence between word frequency distributions
import spacy
import json
import math
from collections import defaultdict


VALID_JSONS = [
  'checkpoints/valid_00_00.json',
  'checkpoints/valid_00_20.json',
  'checkpoints/valid_00_40.json',
  'checkpoints/valid_00_60.json',
  'checkpoints/valid_00_80.json',
  'checkpoints/valid_20_00.json',
  'checkpoints/valid_20_20.json',
  'checkpoints/valid_20_40.json',
  'checkpoints/valid_20_60.json',
  'checkpoints/valid_20_80.json',
]


def get_word_distribution(json_path, json_format):
  """For a json file, tally up the word distributions considering only words on the list"""
  with open(json_path) as f:
    data = json.load(f)

  words = defaultdict(int)
  total_words = 0

  # Tally up across all
  if json_format == 'truth':
    captions = data['annotations']
    for caption in captions:
      for w in caption['caption'].lower().split():
        if w in vocab_words:
          words[w] += 1
          total_words += 1
  elif json_format == 'test':
    for caption in data:
      for w in caption['caption'].lower().split():
        if w in vocab_words:
          words[w] += 1
          total_words += 1

  # Normalize by number of words
  for k, v in words.items():
    words[k] = v / total_words
  
  return words


def exceed_length_limit(json_path):
  with open(json_path) as f:
    data = json.load(f)

  c = 0
  for caption in data:
    if len(caption['caption'].split()) == 20:
      c += 1
  return c / len(data)


def KL(dist_p, dist_q):
  """Compute discrete KL divergence"""
  kl = 0
  for w in dist_q.keys():
    if dist_p[w] > 0:
      kl += dist_p[w] * (math.log(dist_p[w] / dist_q[w]))
  return kl


def main():
  global vocab_words
  nlp = spacy.load('en_core_web_md', disable = ['tagger', 'parser', 'ner'])

  with open('coco_words.txt') as f:
    vocab_words = f.read().split()

  # Only allow real words
  vocab_words = set(list(filter(lambda w: w.islower() and w in nlp.vocab, vocab_words)))


  ground_truth_distribution = get_word_distribution('../annotations/captions_val2014.json', json_format = 'truth')
  for validation_file in VALID_JSONS:
    print(validation_file)
    validation_distribution = get_word_distribution(validation_file, json_format = 'test')
    print('Unique:', len(validation_distribution))
    print('KL:', KL(validation_distribution, ground_truth_distribution))
    print('Exceed:', exceed_length_limit(validation_file))
    print()


main()
