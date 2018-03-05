Experiments with RNNs for image captioning

# Setup Project

Download COCO2014 training and validation set

Download VGG pretrained model:

```
wget https://download.pytorch.org/models/vgg16-397923af.pth
````

Install Spacy and download 'en_core_web_md' word vectors

Run script `most_common_words_in_coco.py` to generate index <-> word mapping
