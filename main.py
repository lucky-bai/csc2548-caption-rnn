import torch
import torchvision
from torch.autograd import Variable
import torch.optim
import pdb
from PIL import Image
import coco_data_loader
import caption_net
import numpy as np
import random
import sys
import argparse
import json

RNG_SEED = 236347

EPOCHS = 20
BATCH_SIZE = 100
SAVE_MODEL_EVERY = 600

IMAGE_DIR = '../train2014'


# Image of a bed
TEST_IMAGE = '../train2014/COCO_train2014_000000436508.jpg'


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
    for batch_ix, (images, sentences, wordvecs) in enumerate(dataloader):
      optimizer.zero_grad()
      images = Variable(images).cuda()
      batch_loss = model.forward_perplexity(images, sentences, wordvecs)

      # Update parameters
      batch_loss.backward()
      optimizer.step()
      print('Epoch %d, batch %d/%d, loss %0.9f' % (epoch, batch_ix, len(dataloader), batch_loss))

      if (batch_ix+1) % SAVE_MODEL_EVERY == 0:
        print('Saving...')
        torch.save(model.state_dict(), 'caption_net.t7')

    # Save at end of each epoch
    print('Saving...')
    torch.save(model.state_dict(), 'caption_net.t7')


def validation_loop():
  dataloader = torch.utils.data.DataLoader(
    coco_data_loader.CocoDataValid(),
    batch_size = BATCH_SIZE,
    num_workers = 16,
    shuffle = True,
  )

  model = caption_net.CaptionNet().cuda()
  model.load_state_dict(torch.load('caption_net.t7'))
  model.eval()

  valid_out = []
  for batch_ix, (image_ids, images) in enumerate(dataloader):
    print('Validation %d/%d' % (batch_ix, len(dataloader)))

    images = Variable(images).cuda()
    captions = model(images)

    for image_id, caption in zip(image_ids, captions):
      caption = ' '.join(caption)
      valid_out.append({
        'image_id': image_id,
        'caption': caption,
      })

  with open('valid.json', 'w') as json_out_file:
    json.dump(valid_out, json_out_file, indent = 2)


def inference_mode(imgfile):
  """Generate a caption for a new image"""
  model = caption_net.CaptionNet().cuda()
  model.load_state_dict(torch.load('caption_net.t7'))
  model.eval()

  img = Image.open(imgfile)
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(coco_data_loader.resize_and_pad),
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

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode')
  parser.add_argument('--file')
  args = parser.parse_args()

  if args.mode == 'train':
    training_loop()
  elif args.mode == 'valid':
    validation_loop()
  else:
    inference_mode(args.file)


main()
