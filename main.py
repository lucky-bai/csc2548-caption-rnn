import torch
import torchvision
import torch.nn as nn
from torch.autograd import Variable
import pdb
from PIL import Image
import json


VGG_MODEL_FILE = 'vgg16-397923af.pth'
MODEL_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']
VGG_IMG_DIM = 224


class VGG(nn.Module):

  def __init__(self, features, num_classes=1000, init_weights=True):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, 4096),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(4096, num_classes),
    )
    if init_weights:
      self._initialize_weights()

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

  def _initialize_weights(self):
    for m in self.modules():
      if isinstance(m, nn.Conv2d):
        n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
        m.weight.data.normal_(0, math.sqrt(2. / n))
        if m.bias is not None:
          m.bias.data.zero_()
      elif isinstance(m, nn.BatchNorm2d):
        m.weight.data.fill_(1)
        m.bias.data.zero_()
      elif isinstance(m, nn.Linear):
        m.weight.data.normal_(0, 0.01)
        m.bias.data.zero_()


def make_layers(cfg, batch_norm=False):
  layers = []
  in_channels = 3
  for v in cfg:
    if v == 'M':
      layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
    else:
      conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
      if batch_norm:
        layers += [conv2d, nn.BatchNorm2d(v), nn.ReLU(inplace=True)]
      else:
        layers += [conv2d, nn.ReLU(inplace=True)]
      in_channels = v
  return nn.Sequential(*layers)



def load_vgg16(**kwargs):
  kwargs['init_weights'] = False
  model = VGG(make_layers(MODEL_CFG), **kwargs)
  model.load_state_dict(torch.load(VGG_MODEL_FILE), strict = False)
  return model


def resize_and_pad(img):
  img.thumbnail((224, 224))
  w, h = img.size
  new_img = Image.new('RGB', (224, 224), 'black')
  new_img.paste(img, ((224 - w)//2, (224 - h)//2))
  return new_img


# Image of a bed
TEST_IMAGE = '../train2014/COCO_train2014_000000436508.jpg'


def main():
  model = load_vgg16().cuda()
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


main()
