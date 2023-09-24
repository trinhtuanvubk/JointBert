
'''
NOTE
- inputs: {"intput_ids": {batch, max_length},
            "attention_mask": {batch, max_length},
            "token_type_ids":
            "intent_label_ids":
            "slot_labels_ids": 
            }
- ouputs:{"outputs": {batch}} - outputs[0]: {batch, num_intents
                              - outputs[1]: {batch, max_length, num_entities}}

'''

from model import JointRoberta
from transformers import RobertaTokenizer
import torch
from utils import get_intent_labels, get_slot_labels 
import argparse
import numpy as np


parser = argparse.ArgumentParser()
parser.add_argument("--sentence", default="customer serice", required=False, type=str, help="Enter an input sentence")
parser.add_argument("--model_dir", default="/home/sangdt/research/JointBert/ckpt", required=False, type=str, help="Path to save, load model")
parser.add_argument("--intent_label_file", default="/home/sangdt/research/JointBert/processed_data/intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="/home/sangdt/research/JointBert/processed_data/slot_label.txt", type=str, help="Slot Label file")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
parser.add_argument("--ignore_index", default=1, type=int,
                    help='Specifies a target value that is ignored and does not contribute to the input gradient')

parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

args = parser.parse_args()



device = torch.device("cuda")
ckpt = "./checkpoints/"
onnx_path = "./checkpoints/model.onnx"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dummy_model_input = tokenizer(["customer service", "I want to book this hotel"], padding='max_length', truncation=True, return_tensors="pt")

input_ids = dummy_model_input['input_ids'].to(device)
attention_mask = dummy_model_input['attention_mask'].to(device)
intent_label_lst = get_intent_labels(args)
dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}
slot_label_lst = get_slot_labels(args)
dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}
model = JointRoberta.from_pretrained(args.model_dir, args, intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst).to(device)
model.eval()
# export
torch.onnx.export(
    model, 
    tuple([input_ids, attention_mask]),
    f="./ckpt/model_test.onnx",  
    verbose=True,
    input_names=['input_ids', 'attention_mask'], 
    output_names=['outputs'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'max_length'}, 
                  'attention_mask': {0: 'batch_size', 1: 'max_length'}, 
                  'outputs': {0: 'batch_size'}}, 
    do_constant_folding=True, 
)
