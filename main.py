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

  training_loop()
  #inference_mode()


main()
