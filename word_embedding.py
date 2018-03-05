import torch
import spacy

class WordEmbedding:
  """Manager for vocabulary and word embeddings"""

  def __init__(self):
    self.nlp = spacy.load('en_core_web_md')
    with open('coco_words.txt') as f:
      self.vocab_words = f.read().split()

  def get_word_embedding(self, word):
    t = self.nlp(word)
    return torch.Tensor(t[0].vector)

  def get_word_from_index(self, ix):
    return self.vocab_words[ix]



def test():
  manager = WordEmbedding()
  v = manager.get_word_embedding('car')
  print(v)
  print(manager.get_word_from_index(0))
  print(manager.get_word_from_index(1))
  print(manager.get_word_from_index(2))


#test()
