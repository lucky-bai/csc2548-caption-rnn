for file in checkpoints/*.json ; do
  echo $file
  python2 coco_eval_kit/eval.py $file
done
