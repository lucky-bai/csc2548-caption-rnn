import torch
import pdb
import numpy as np
import random
import argparse
import training_mode
import evaluation_mode

RNG_SEED = 236348


def main():
  np.random.seed(RNG_SEED)
  torch.manual_seed(RNG_SEED)
  random.seed(RNG_SEED)

  parser = argparse.ArgumentParser()
  parser.add_argument('--mode')
  parser.add_argument('--image')
  parser.add_argument('--eval-dropout', type=float)
  parser.add_argument('--output-json')
  parser.add_argument('--model-weights')
  args = parser.parse_args()
  print(args)

  if args.mode == 'train':
    training_mode.training_loop()
  elif args.mode == 'eval':
    evaluation_mode.evaluation_loop(args)
  else:
    evaluation_mode.caption_single_image(args.image)


main()
