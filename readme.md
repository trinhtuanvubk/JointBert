### JointBERT

-To train:
```
python3 main.py --do_train  --model_dir=./ckpt_test
```
-To infer:
```
CUDA_VISIBLE_DEVICES=1 python3 inference.py --sentence "customer service" --model_dir ./ckpt --use_crf
```