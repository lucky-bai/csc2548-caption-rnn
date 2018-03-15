import torch
import os
import numpy as np
import torch.utils.data
import torchvision
import word_embedding
from PIL import Image
import json


TRAIN_DIR = '../train2014'
VALID_DIR = '../val2014'

TRAIN_JSON = '../annotations/captions_train2014.json'
VALID_JSON = '../annotations/captions_val2014.json'


def resize_and_pad(img):
  img.thumbnail((224, 224))
  w, h = img.size
  new_img = Image.new('RGB', (224, 224), 'black')
  new_img.paste(img, ((224 - w)//2, (224 - h)//2))
  return new_img


class CocoData(torch.utils.data.Dataset):
  """Utility for loading COCO data in batches"""

  def __init__(self, mode):
    self.word_embeddings = word_embedding.WordEmbedding()
    self.mode = mode

    if mode == 'train':
      self.json_file = TRAIN_JSON
    else:
      self.json_file = VALID_JSON

    with open(self.json_file) as jsonf:
      data = json.load(jsonf)
      self.captions = data['annotations']


  def __len__(self):
    return len(self.captions)


  def __getitem__(self, idx):
    caption = self.captions[idx]
    image_id = caption['image_id']
    if self.mode == 'train':
      image_file = '%s/COCO_train2014_%012d.jpg' % (TRAIN_DIR, image_id)
    else:
      image_file = '%s/COCO_val2014_%012d.jpg' % (VALID_DIR, image_id)
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


class CocoDataValid(torch.utils.data.Dataset):
  def __init__(self):
    self.images = os.listdir(VALID_DIR)

  def __len__(self):
    return len(self.images)

  def __getitem__(self, idx):
    """Return (ID, image tensor)"""
    image_file = VALID_DIR + '/' + self.images[idx]

    img = Image.open(image_file)
    transforms = torchvision.transforms.Compose([
      torchvision.transforms.Lambda(resize_and_pad),
      torchvision.transforms.ToTensor(),
    ])
    img = transforms(img)

    img_id = int(image_file.split('.')[-2][-12:])
    return img_id, img
