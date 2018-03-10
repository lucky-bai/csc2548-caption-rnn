import torch
import torchvision
from torch.autograd import Variable
import torch.optim
import pdb
from PIL import Image
import json
import caption_net
import numpy as np
import os
import random

RNG_SEED = 236346

EPOCHS = 10
BATCH_SIZE = 20
SAVE_MODEL_EVERY = 200

IMAGE_DIR = '../train2014'
CAPTION_JSON = '../annotations/captions_train2014.json'



def resize_and_pad(img):
  img.thumbnail((224, 224))
  w, h = img.size
  new_img = Image.new('RGB', (224, 224), 'black')
  new_img.paste(img, ((224 - w)//2, (224 - h)//2))
  return new_img


# Image of a bed
TEST_IMAGE = '../train2014/COCO_train2014_000000436508.jpg'


def test_vgg_on_image():
  model = caption_net.VGG(caption_net.make_layers(caption_net.VGG_MODEL_CFG)).cuda()
  model.load_state_dict(torch.load(caption_net.VGG_MODEL_FILE))

  img = Image.open(TEST_IMAGE)
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resize_and_pad),
    torchvision.transforms.ToTensor(),
  ])
  img = transforms(img).unsqueeze(0)
  img = Variable(img).cuda()
  out = model(img)
  _, result = torch.topk(out, k = 5)
  result = result[0].data.tolist()

  with open('imagenet_to_human.json') as jsonf:
    imagenet_to_human = json.load(jsonf)

  for r in result:
    r = str(r)
    print(r, imagenet_to_human[r])


def training_loop():

  # Load JSON
  with open(CAPTION_JSON) as jsonf:
    data = json.load(jsonf)
    captions = data['annotations']

  # Shuffle captions
  random.shuffle(captions)
  captions_batches = [captions[i:i+BATCH_SIZE] for i in range(0, len(captions), BATCH_SIZE)]
  num_batches = len(captions_batches)

  # Initialize model
  model = caption_net.CaptionNet().cuda()
  model.train()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

  for epoch in range(EPOCHS):
    for batch_ix, captions_batch in enumerate(captions_batches):
      batch_loss = 0
      optimizer.zero_grad()

      for caption in captions_batch:
        image_id = caption['image_id']
        image_file = '%s/COCO_train2014_%012d.jpg' % (IMAGE_DIR, image_id)
        text = caption['caption']

        # Load image
        img = Image.open(image_file)
        transforms = torchvision.transforms.Compose([
          torchvision.transforms.Lambda(resize_and_pad),
          torchvision.transforms.ToTensor(),
        ])
        img = transforms(img).unsqueeze(0)
        img = Variable(img).cuda()

        # Compute loss
        loss = model.forward_perplexity(img, text)
        batch_loss += loss

      # Update parameters
      batch_loss.backward()
      optimizer.step()
      print('Epoch %d, batch %d/%d, loss %0.9f' % (epoch, batch_ix, num_batches, batch_loss))
      batch_loss = 0

      if (batch_ix+1) % SAVE_MODEL_EVERY == 0:
        print('Saving...')
        torch.save(model.state_dict(), 'caption_net.t7')

    # Save at end of each epoch
    print('Saving...')
    torch.save(model.state_dict(), 'caption_net.t7')



def inference_mode():
  """Generate a caption for a new image"""
  model = caption_net.CaptionNet().cuda()
  model.load_state_dict(torch.load('caption_net.t7'))
  model.eval()

  img = Image.open(TEST_IMAGE)
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resize_and_pad),
    torchvision.transforms.ToTensor(),
  ])
  img = transforms(img).unsqueeze(0)
  img = Variable(img).cuda()
  out = model(img)

  print(out)



def main():
  np.random.seed(RNG_SEED)
  torch.manual_seed(RNG_SEED)
  random.seed(RNG_SEED)

  #test_vgg_on_image()
  training_loop()
  #inference_mode()


main()
