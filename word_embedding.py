import torch
import spacy
import string
import numpy as np
import pdb


class WordEmbedding:
  """Manager for vocabulary and word embeddings"""

  def __init__(self):
    self.nlp = spacy.load('en_core_web_md', disable = ['tagger', 'parser', 'ner'])
    with open('coco_words.txt') as f:
      self.vocab_words = f.read().split()

    # Only allow real words
    self.vocab_words = list(filter(lambda w: w.islower() and w in self.nlp.vocab, self.vocab_words))

    # Special vectors
    self.END_MARKER = np.zeros(301)
    self.END_MARKER[-1] = 1
    self.vocab_words.append('.')

  def get_word_embedding(self, word):
    if word == '.':
      return self.END_MARKER
    v = self.nlp.vocab.get_vector(word)
    v = np.append(v, [0])
    return v

  def get_word_from_index(self, ix):
    return self.vocab_words[ix]

  def get_index_from_word(self, word):
    return self.vocab_words.index(word)


  def sentence_to_embedding(self, text, pad):
    """Process natural language sentence into sequence of word vectors"""

    # Convert to lowercase and remove punctuation
    text = text.lower()
    text = ''.join([c for c in text if c not in string.punctuation])
    
    embeddings = []
    words = []
    for word in text.split():
      if word in self.vocab_words:
        words.append(word)
        embeddings.append(self.get_word_embedding(word))

    # Pad with periods if it's too short
    words = words[:pad]
    embeddings = embeddings[:pad]
    while len(words) < pad:
      words.append('.')
      embeddings.append(self.END_MARKER)

    return words, embeddings



def test():
  manager = WordEmbedding()
  print(manager.sentence_to_embedding('I have a Large Gray wabbit, she is cute'))


#test()
