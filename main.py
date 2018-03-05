import torch
import torchvision
from torch.autograd import Variable
import pdb
from PIL import Image
import json
import caption_net
import numpy as np

RNG_SEED = 236346



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


def main():
  np.random.seed(RNG_SEED)
  torch.manual_seed(RNG_SEED)

  #test_vgg_on_image()
  model = caption_net.CaptionNet().cuda()

  img = Image.open(TEST_IMAGE)
  transforms = torchvision.transforms.Compose([
    torchvision.transforms.Lambda(resize_and_pad),
    torchvision.transforms.ToTensor(),
  ])
  img = transforms(img).unsqueeze(0)
  img = Variable(img).cuda()
  #out = model(img)
  perp = model.forward_perplexity(img, ['the', 'man', 'is', 'biting', 'the', 'dog'])

  print(perp)



main()
