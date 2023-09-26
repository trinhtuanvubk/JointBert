## JointBERT

### Environment
```
conda create -n jointbert python=3.8
conda activate jointbert
pip3 install -r requirements.txt
```

### Data

- Put your data in `data` folder. It should consist of 2 files: `train_slots.jsonl` and `test_slots.jsonl`
- A line sample in each file: 
'{"text": "to make a payment", "tokens": ["to", "make", "a", "payment"], "intention": "payment", "slots": ["O", "O", "O", "B-PAYMENT"]}'

- Preprocessing: Because using roberta's tokenizer will split the string into subwords differently than using space, we need to convert from token splitted by space to token splitted by Roberta's tokenizer. For example: ["customer",  "service"] (tokenized by space) -> ["custom", "er", "Ġservice"] (tokenized by roberta's tokenizer). So, this example would have the tag sequence as ["B-CUSTOMER", I-CUSTOMER", "B-SERVICE"] instead of ["B-CUSTOMER", "B-SERVICE"]. After training, at inference time, we map from tag sequence ["B-CUSTOMER", I-CUSTOMER", "B-SERVICE"] to ["B-CUSTOMER", "B-SERVICE"]. The preprocessing code is the `transform` function at `trainer.py`

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