#this is in python2
import sys
sys.path.insert(0, './coco-caption-master')
#import os 
#print os.environ['PATH']
from pycocotools.coco import COCO
from pycocoevalcap.eval import COCOEvalCap
import matplotlib.pyplot as plt
import skimage.io as io
import pylab
pylab.rcParams['figure.figsize'] = (10.0, 8.0)

import json
from json import encoder
encoder.FLOAT_REPR = lambda o: format(o, '.3f')

#file setup
mainDir='.'
dataType='val2014'
annFile='%s/annotations/captions_%s.json'%(mainDir,dataType)
#subtypes=['results','evalImgs','eval']
resFile ='%s/captions_myresults.json'%(mainDir)
evalImgsFile = '%s/captions_evalImgs.json'%(mainDir)
evalFile = '%s/captions_eval.json'%(mainDir)
print annFile

IMG_ID=[203564,322141]

#create coco object and cocoRes object
coco = COCO(annFile)
cocoRes = coco.loadRes(resFile)

#create cocoEval object by taking coco and cocoRes
cocoEval = COCOEvalCap(coco, cocoRes)

#evaluate on a subset of images by setting
#cocoEval.params['image_id'] = cocoRes.getImgIds()
#remove when evaluating full validation set
cocoEval.params['image_id'] = cocoRes.getImgIds(imgIds=IMG_ID)
#print cocoEval.params['image_id']
#evaluate
cocoEval.evaluate()

#print output evaluation scores
for metric, score in cocoEval.eval.items():
    print '%s: %.3f'%(metric, score)

#use evalImgs to retrieve low score result
evals = [eva for eva in cocoEval.evalImgs if eva['Bleu_2'] < 20]
print evals
print 'ground truth captions'
imgId = evals[0]['image_id']
annIds = coco.getAnnIds(imgIds=imgId)
anns = coco.loadAnns(annIds)
coco.showAnns(anns)

print '\n'
print 'generated caption (Bleu_2 score %0.1f)'%(evals[0]['Bleu_2'])
annIds = cocoRes.getAnnIds(imgIds=imgId)
anns= cocoRes.loadAnns(annIds)
coco.showAnns(anns)

#img=coco.loadImgs(imgId)[0]
#I = io.imread('%s/%s/%s'%(mainDir,dataType,img['file_name']))
#plt.imshow(I)
#plt.axis('off')
#plt.show()

#save result
json.dump(cocoEval.evalImgs, open(evalImgsFile, 'w'))
json.dump(cocoEval.eval, open(evalFile, 'w'))


