## JointBERT

### Data

- Put your data in `data` folder. It shoule consist of 2 files: `train_slots.jsonl` and `test_slots.jsonl`
- A line sample in each file: 
'{"text": "to make a payment", "tokens": ["to", "make", "a", "payment"], "intention": "payment", "slots": ["O", "O", "O", "B-PAYMENT"]}'

### Training

- To train:
```
python3 main.py --do_train  --model_dir=./ckpt
```

### Inference

- To infer:
```
CUDA_VISIBLE_DEVICES=1 python3 inference.py --sentence "customer service" --model_dir ./ckpt
```