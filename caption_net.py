import torch
import torch.nn as nn
import torch.nn.functional as F
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
# Add one to handle END_MARKER
WORDVEC_SIZE = 300 + 1

# Length of vocab_words handled by WordEmbedding
# Todo: stop hardcoding this
VOCABULARY_SIZE = 9870 + 1

# Pad with periods if too short
SENTENCE_LENGTH = 20


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
    
    # Freeze all VGG layers
    for param in self.vgg.parameters():
      param.requires_grad = False

    # Recurrent layer
    self.lstm_cell = nn.LSTMCell(
      input_size = WORDVEC_SIZE,
      hidden_size = RNN_HIDDEN_SIZE,
    )

    # Linear layer to convert hidden layer to word in vocab
    self.hidden_to_vocab = nn.Linear(RNN_HIDDEN_SIZE, VOCABULARY_SIZE)

    self.word_embeddings = word_embedding.WordEmbedding()


  def forward(self, imgs):
    """Forward pass through network
    Input: image tensor
    Output: sequence of words
    """
    batch_size = imgs.shape[0]
    hidden = self.vgg.forward_until_hidden_layer(imgs)
    cell = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()

    # First word vector is zero
    next_input = Variable(torch.zeros(batch_size, WORDVEC_SIZE)).cuda()
    
    captions = [[] for _ in range(batch_size)]
    for ix in range(SENTENCE_LENGTH):
      hidden, cell = self.lstm_cell(next_input, (hidden, cell))
      word_class = self.hidden_to_vocab(hidden)
      _, word_ix = torch.max(word_class, 1)

      for batch_ix in range(batch_size):

        # Append only if we're not already done (hit a period)
        if len(captions[batch_ix]) == 0 or captions[batch_ix][-1] != '.':
          cur_word = self.word_embeddings.get_word_from_index(int(word_ix[batch_ix]))
          captions[batch_ix].append(cur_word)

        # Update input to next layer
        next_input[batch_ix, :] = torch.Tensor(self.word_embeddings.get_word_embedding(cur_word))

    return captions


  def forward_perplexity(self, imgs, sentences, wordvecs):
    """Given image and ground-truth caption, compute negative log likelihood perplexity"""
    batch_size = imgs.shape[0]

    # (batch_ix, position in sentence, 301)
    wordvecs = torch.stack(wordvecs).permute(1,0,2)

    hidden = self.vgg.forward_until_hidden_layer(imgs)
    cell = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()

    # Train it to predict the next word given previous word vector
    # Remove the last word vector and prepend zero vector as first input
    wordvecs_shift_one = torch.cat([
        torch.zeros(batch_size, 1, WORDVEC_SIZE).double(),
        wordvecs[:, :-1, :]
      ],
      dim = 1,
    )

    sum_nll = 0
    for ix in range(SENTENCE_LENGTH):
      next_input = Variable(wordvecs_shift_one[:, ix, :]).cuda().float()
      hidden, cell = self.lstm_cell(next_input, (hidden, cell))

      word_class = self.hidden_to_vocab(hidden)

      word_ix_list = []
      for w in sentences[ix]:
        word_ix_list.append(self.word_embeddings.get_index_from_word(w))
      word_ix = Variable(torch.LongTensor(word_ix_list)).cuda()

      nll = F.cross_entropy(word_class, word_ix)
      sum_nll += nll

    return sum_nll
