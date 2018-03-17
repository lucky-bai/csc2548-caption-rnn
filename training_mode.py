import torch
from torch.autograd import Variable
import torch.optim
import pdb
import coco_data_loader
import caption_net

EPOCHS = 50
BATCH_SIZE = 150
SAVE_MODEL_EVERY = 999999


def get_validation_loss(model):
  model.eval()
  valid_dataloader = torch.utils.data.DataLoader(
    coco_data_loader.CocoData(mode = 'valid'),
    batch_size = BATCH_SIZE,
    num_workers = 16,
    shuffle = True,
  )

  validation_loss = 0
  for batch_ix, (images, sentences, wordvecs) in enumerate(valid_dataloader):
    images = Variable(images).cuda()
    batch_loss = model.forward_perplexity(images, sentences, wordvecs)
    print('Validation %d/%d' % (batch_ix, len(valid_dataloader)))
    validation_loss += float(batch_loss)
  
  # Reset evaluation mode when we're done
  model.train()

  # Return average validation loss
  return validation_loss / len(valid_dataloader)



def training_loop():
  train_dataloader = torch.utils.data.DataLoader(
    coco_data_loader.CocoData(mode = 'train'),
    batch_size = BATCH_SIZE,
    num_workers = 16,
    shuffle = True,
  )

  # Initialize model
  model = caption_net.CaptionNet().cuda()
  model.train()
  optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()))
  best_validation_loss = 1e8

  for epoch in range(EPOCHS):
    for batch_ix, (images, sentences, wordvecs) in enumerate(train_dataloader):
      optimizer.zero_grad()
      images = Variable(images).cuda()
      batch_loss = model.forward_perplexity(images, sentences, wordvecs)

      # Update parameters
      batch_loss.backward()
      optimizer.step()
      print('Epoch %d, batch %d/%d, loss %0.9f' % (epoch, batch_ix, len(train_dataloader), batch_loss))

      if (batch_ix+1) % SAVE_MODEL_EVERY == 0:
        print('Saving...')
        torch.save(model.state_dict(), 'caption_net.t7')

    # Calculate validation loss at end of epoch
    validation_loss = get_validation_loss(model)
    print('Epoch %d, validation Loss %0.9f' % (epoch, validation_loss))

    # Save if validation loss improved, otherwise stop early
    if validation_loss < best_validation_loss:
      best_validation_loss = validation_loss
      print('Saving...')
      torch.save(model.state_dict(), 'caption_net.t7')
    else:
      break
