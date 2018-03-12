import torch
import torchvision
from torch.autograd import Variable
import torch.optim
import pdb
from PIL import Image
import coco_data_loader
import json
import caption_net
import numpy as np
import os
import random
import sys

RNG_SEED = 236347

EPOCHS = 10
BATCH_SIZE = 100
SAVE_MODEL_EVERY = 200

IMAGE_DIR = '../train2014'



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
  dataloader = torch.utils.data.DataLoader(
    coco_data_loader.CocoData(),
    batch_size = BATCH_SIZE,
    num_workers = 16,
    shuffle = True,
  )

  # Initialize model
  model = caption_net.CaptionNet().cuda()
  model.train()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))

  for epoch in range(EPOCHS):
    batch_loss = 0
    for batch_ix, (images, sentences, wordvecs) in enumerate(dataloader):
      images = Variable(images).cuda()
      batch_loss = model.forward_perplexity(images, sentences, wordvecs)

      # Update parameters
      batch_loss.backward()
      optimizer.step()
      print('Epoch %d, batch %d/%d, loss %0.9f' % (epoch, batch_ix, len(dataloader), batch_loss))
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

  if len(sys.argv) >= 2:
    img_path = sys.argv[1]
  else:
    img_path = TEST_IMAGE

  img = Image.open(img_path)
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
