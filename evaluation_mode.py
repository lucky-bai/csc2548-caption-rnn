import torch
import torchvision
from torch.autograd import Variable
import pdb
from PIL import Image
import coco_data_loader
import caption_net
import json

BATCH_SIZE = 150


def evaluation_loop():
  """Generate a set of captions for evaluation"""
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
    print('Evaluation %d/%d' % (batch_ix, len(dataloader)))

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


def caption_single_image(imgfile):
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
