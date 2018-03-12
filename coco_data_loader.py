import torch
import os
import numpy as np
import torch.utils.data
import torchvision
import word_embedding
from PIL import Image
import json


IMAGE_DIR = '../train2014'
CAPTION_JSON = '../annotations/captions_train2014.json'


def resize_and_pad(img):
  img.thumbnail((224, 224))
  w, h = img.size
  new_img = Image.new('RGB', (224, 224), 'black')
  new_img.paste(img, ((224 - w)//2, (224 - h)//2))
  return new_img


class CocoData(torch.utils.data.Dataset):
  """Utility for loading COCO data in batches"""

  def __init__(self):
    self.word_embeddings = word_embedding.WordEmbedding()
    with open(CAPTION_JSON) as jsonf:
      data = json.load(jsonf)
      self.captions = data['annotations']


  def __len__(self):
    return len(self.captions)


  def __getitem__(self, idx):
    caption = self.captions[idx]
    image_id = caption['image_id']
    image_file = '%s/COCO_train2014_%012d.jpg' % (IMAGE_DIR, image_id)
    text = caption['caption']

    # Process text
    words, wordvecs = self.word_embeddings.sentence_to_embedding(text, pad = 20)
    
    # Load image
    img = Image.open(image_file)
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.Lambda(resize_and_pad),
      torchvision.transforms.ToTensor(),
    ])
    img = transforms(img)

    return img, words, wordvecs
