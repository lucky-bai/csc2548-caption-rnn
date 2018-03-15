# Simple HTML visualization of some captions
import json
import os

NUM_IMAGES_SHOW = 100
VALID_JSON_FILE = '../valid.json'
VALID_FOLDER = '../../val2014'

HTML_TEMPLATE = """
<p>
<img src="%s" style="max-width:400px; max-height:400px" /> <br>
<b>%s</b>
</p>
<hr>
"""
HTML_FILE = 'captions.html'


def main():
  with open(VALID_JSON_FILE) as f:
    data = json.load(f)

  with open(HTML_FILE, 'w') as html_file:
    data = data[:NUM_IMAGES_SHOW]
    for caption in data:
      image_id = int(caption['image_id'])
      caption = caption['caption']

      img_path = '%s/COCO_val2014_%012d.jpg' % (VALID_FOLDER, image_id)
      img_path2 = '%s/COCO_val2014_%012d.jpg' % ('img', image_id)
      os.system('cp %s img' % img_path)

      html_file.write(HTML_TEMPLATE % (img_path2, caption))

  # Zip it up for easy transfer
  os.system('zip -r captions.zip captions.html img')


main()
