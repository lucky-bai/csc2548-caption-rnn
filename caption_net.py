import torch
import torch.nn as nn
from torch.autograd import Variable
import word_embedding
import pdb
import numpy as np

# Pretrained weights for VGG16
VGG_MODEL_FILE = 'vgg16-397923af.pth'
VGG_MODEL_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# Input dimensions of VGG16 input image
VGG_IMG_DIM = 224

# Recurrent size must be same as last hidden layer off VGG16
RNN_HIDDEN_SIZE = 4096

# Dimension of word embeddings
WORDVEC_SIZE = 300

# Assume a limited language model consisting of this many words
VOCABULARY_SIZE = 6000


class VGG(nn.Module):

  def __init__(self, features, num_classes=1000):
    super(VGG, self).__init__()
    self.features = features
    self.classifier = nn.Sequential(
      nn.Linear(512 * 7 * 7, RNN_HIDDEN_SIZE),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(RNN_HIDDEN_SIZE, RNN_HIDDEN_SIZE),
      nn.ReLU(True),
      nn.Dropout(),
      nn.Linear(RNN_HIDDEN_SIZE, num_classes),
    )

  def forward(self, x):
    x = self.features(x)
    x = x.view(x.size(0), -1)
    x = self.classifier(x)
    return x

  def forward_until_hidden_layer(self, x):
    """Stop at the hidden layer before the final classification layer"""
    x = self.features(x)
    x = x.view(x.size(0), -1)
    # Do 4 of the layers in classifier
    x = self.classifier._modules['0'](x)
    x = self.classifier._modules['1'](x)
    x = self.classifier._modules['2'](x)
    x = self.classifier._modules['3'](x)
    return x


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



class CaptionNet(nn.Module):

  def __init__(self):
    super(CaptionNet, self).__init__()

    # Make VGG net
    self.vgg = VGG(make_layers(VGG_MODEL_CFG))
    self.vgg.load_state_dict(torch.load(VGG_MODEL_FILE))

    # Recurrent layer
    self.rnn_cell = nn.RNNCell(
      input_size = WORDVEC_SIZE,
      hidden_size = RNN_HIDDEN_SIZE,
      nonlinearity = 'relu',
    )

    # Linear layer to convert hidden layer to word in vocab
    self.hidden_to_vocab = nn.Linear(RNN_HIDDEN_SIZE, VOCABULARY_SIZE)


  def forward(self, img):
    """Forward pass through network
    Input: image tensor
    Output: sequence of words
    """
    hidden = self.vgg.forward_until_hidden_layer(img)

    # First input is zero vector
    next_input = Variable(torch.zeros(WORDVEC_SIZE)).cuda()
    
    # For now, let's just generate 10 words (should actually generate until end token)
    words = []
    for _ in range(10):
      hidden = self.rnn_cell(next_input, hidden)
      word_class = self.hidden_to_vocab(hidden)
      _, word_ix = torch.max(word_class, 1)
      word_ix = int(word_ix)

      cur_word = word_embedding.get_word_from_index(word_ix)
      words.append(cur_word)

      # Update input to next layer
      next_input = Variable(word_embedding.get_word_embedding(cur_word)).cuda()

    return words


  def forward_perplexity(self, img, words):
    """Given image and ground-truth caption, compute negative log likelihood perplexity"""
    # Todo
    return 0
