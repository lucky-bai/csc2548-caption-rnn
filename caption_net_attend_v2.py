import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
import word_embedding
import pdb
import numpy as np
import torchvision.models as vmodels

# Pretrained weights for VGG16
VGG_MODEL_FILE = 'vgg16-397923af.pth'
VGG_MODEL_CFG = [64, 64, 'M', 128, 128, 'M', 256, 256, 256, 'M', 512, 512, 512, 'M', 512, 512, 512, 'M']

# Input dimensions of VGG16 input image
VGG_IMG_DIM = 224

VGG_FEATURE_DIM = 14*14
VGG_FILTER_NUM = 512
#VGG_FEATURE_DIM = 7*7*512

ATTN_REG_FACTOR = 0.0007
# 0326 used 0.0005

# Recurrent size must be same as last hidden layer off VGG16
RNN_HIDDEN_SIZE = 512

# Dimension of word embeddings
# Add one to handle END_MARKER
WORDVEC_SIZE = 300 + 1

# Length of vocab_words handled by WordEmbedding
# Todo: stop hardcoding this
VOCABULARY_SIZE = 9870 + 1

# Pad with periods if too short
SENTENCE_LENGTH = 20

class myVGG(nn.Module):
  def __init__(self):
    super(myVGG, self).__init__()

    # Make VGG net
    myvgg16 = vmodels.vgg16(pretrained=True)
    #modules = list(myvgg16.children())[:-1]
    modules = list(myvgg16.features)[:-1]
    self.vggnet = nn.Sequential(*modules)
    #self.classifier = nn.Sequential(*(myvgg16.classifier[i] for i in range(3)))
    #self.classifier = nn.Sequential(*(myvgg16.classifier[i] for i in range(6)))
    #self.linear = nn.Linear(4096, RNN_HIDDEN_SIZE)
    #self.init_weights()

  def init_weights(self):
    self.linear.weight.data.normal_(0.0, 0.02)
    self.linear.bias.data.fill_(0)
  
  def forward(self, x):
    features = self.vggnet(x)
    features = Variable(features.data)
    #features = features.view(features.size(0), -1)
    #x = self.classifier(features)
    #x = self.linear(x)
    return features
#100,512,17,17
#4096


class CaptionNet(nn.Module):

  def __init__(self):
    super(CaptionNet, self).__init__()

    # Encode
    self.vgg = myVGG()
    
    # Freeze all VGG layers
    for param in self.vgg.parameters():
      param.requires_grad = False

    #enabling fine tuning
    for param in self.vgg.vggnet[28].parameters():
      param.requires_grad = True

    #Soft Attention
    self.attn = nn.Linear(RNN_HIDDEN_SIZE + WORDVEC_SIZE, VGG_FEATURE_DIM)
    self.attn_combine = nn.Linear(WORDVEC_SIZE+VGG_FILTER_NUM, WORDVEC_SIZE)
    self.gating_coeff = nn.Linear(RNN_HIDDEN_SIZE, VGG_FILTER_NUM)

    # Recurrent layer
    self.lstm_cell = nn.LSTMCell(
      input_size = WORDVEC_SIZE,
      hidden_size = RNN_HIDDEN_SIZE,
    )

    # Linear layer to convert hidden layer to word in vocab
    self.hidden_to_vocab = nn.Linear(RNN_HIDDEN_SIZE, VOCABULARY_SIZE)

    self.word_embeddings = word_embedding.WordEmbedding()

    self.init_weights()

  def init_weights(self):
    self.attn.weight.data.normal_(0.0, 0.02)
    self.attn.bias.data.fill_(0)
    self.attn_combine.weight.data.normal_(0.0, 0.02)
    self.attn_combine.bias.data.fill_(0)

  def init_hidden(self, batch_size):
    weight = next(self.parameters()).data
    return (Variable(weight.new(batch_size, RNN_HIDDEN_SIZE).zero_()), Variable(weight.new(batch_size, RNN_HIDDEN_SIZE).zero_()))


  def forward(self, imgs):
    """Forward pass through network
    Input: image tensor
    Output: sequence of words
    """
    batch_size = imgs.shape[0]
    encoding = self.vgg(imgs)
    #cell = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()
    #hidden = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()
    hidden, cell = self.init_hidden(batch_size)
    hidden, cell = Variable(hidden.data).cuda(), Variable(cell.data).cuda()


    # First word vector is zero
    next_input = Variable(torch.zeros(batch_size, WORDVEC_SIZE)).cuda()
    
    # Keep generating until end token
    captions = [[] for _ in range(batch_size)]
    for ix in range(SENTENCE_LENGTH):
      attn_weights = F.softmax(self.attn(torch.cat((next_input, hidden), 1)), dim=1)
      attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoding.view(batch_size,VGG_FILTER_NUM,-1).transpose(1,2)).view(batch_size,-1)

      gamma = F.sigmoid(self.gating_coeff(hidden))
      attn_applied = gamma * attn_applied

      next_input = torch.cat((next_input, attn_applied), 1)

      #next_input = self.attn_combine(next_input).unsqueeze(0)
      next_input = self.attn_combine(next_input)

      next_input = F.relu(next_input)

      hidden, cell = self.lstm_cell(next_input, (hidden, cell))
      word_class = self.hidden_to_vocab(hidden)
      _, word_ix = torch.max(word_class, 1)

      #print("verify shape\n", word_class.shape)
      #print("word_ix\n", word_ix.shape)
      #print("next input shape", next_input.shape)

      for batch_ix in range(batch_size):
          # Append only if we're not already done (hit a period)
          if len(captions[batch_ix]) == 0 or captions[batch_ix][-1] != '.':
              cur_word = self.word_embeddings.get_word_from_index(int(word_ix[batch_ix]))
              #print("cur_word", cur_word)
              captions[batch_ix].append(cur_word)
              #print("wordvec shape", self.word_embeddings.get_word_embedding(cur_word).shape)
              #print("next_input shape", next_input[batch_ix,:].shape)

          # Update input to next layer
          next_input[batch_ix,:] = torch.Tensor(self.word_embeddings.get_word_embedding(cur_word))

    return captions


  def forward_perplexity(self, imgs, sentences, wordvecs):
    """Given image and ground-truth caption, compute negative log likelihood perplexity"""
    batch_size = imgs.shape[0]

    # (batch_ix, position in sentence, 301)
    wordvecs = torch.stack(wordvecs).permute(1,0,2)

    encoding = self.vgg(imgs)
    #print("check encoding dimension", encoding.shape)
    #cell = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()
    #hidden = Variable(torch.zeros(batch_size, RNN_HIDDEN_SIZE)).cuda()
    hidden, cell = self.init_hidden(batch_size)
    hidden, cell = Variable(hidden.data).cuda(), Variable(cell.data).cuda()

    # Train it to predict the next word given previous word vector
    # Remove the last word vector and prepend zero vector as first input
    wordvecs_shift_one = torch.cat([
        torch.zeros(batch_size, 1, WORDVEC_SIZE).double(),
        wordvecs[:, :-1, :]
      ],
      dim = 1,
    )

    sum_nll = 0
    sum_attend_weight = Variable(torch.FloatTensor(batch_size,VGG_FEATURE_DIM).zero_()).cuda() 
    for ix in range(SENTENCE_LENGTH):
      next_input = Variable(wordvecs_shift_one[:, ix, :]).cuda().float()
      #print("Debug atten weight shape\n", next_input.shape, hidden.shape)

      #attention
      attn_weights = F.softmax(self.attn(torch.cat((next_input, hidden), 1)), dim=1)

      attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoding.view(batch_size,VGG_FILTER_NUM,-1).transpose(1,2)).view(batch_size,-1)

      gamma = F.sigmoid(self.gating_coeff(hidden))
      attn_applied = gamma * attn_applied

      next_input = torch.cat((next_input, attn_applied), 1)
      #next_input = self.attn_combine(next_input).unsqueeze(0)
      next_input = self.attn_combine(next_input)
      next_input = F.relu(next_input)

      #decode
      hidden, cell = self.lstm_cell(next_input, (hidden, cell))

      word_class = self.hidden_to_vocab(hidden)

      word_ix_list = []
      for w in sentences[ix]:
        word_ix_list.append(self.word_embeddings.get_index_from_word(w))
      word_ix = Variable(torch.LongTensor(word_ix_list)).cuda()

      nll = F.cross_entropy(word_class, word_ix)
      sum_attend_weight.data += attn_weights.data
      sum_nll += nll

    soft_attn_reg = torch.bmm((1 - sum_attend_weight).unsqueeze(1), (1 - sum_attend_weight).unsqueeze(1).transpose(1,2))
    soft_attn_reg = soft_attn_reg.data.sum()
    print("loss debug\n", soft_attn_reg*ATTN_REG_FACTOR)
    return sum_nll + ATTN_REG_FACTOR*soft_attn_reg
