import torch
import onnxruntime
import numpy as np

from transformers import RobertaTokenizer
import torch
from utils import get_intent_labels, get_slot_labels 
import argparse
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--sentence", default="customer serice", required=False, type=str, help="Enter an input sentence")
parser.add_argument("--model_dir", default="/home/sangdt/research/JointBert/ckpt_no_crf", required=False, type=str, help="Path to save, load model")
parser.add_argument("--intent_label_file", default="/home/sangdt/research/JointBert/processed_data/intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="/home/sangdt/research/JointBert/processed_data/slot_label.txt", type=str, help="Slot Label file")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
parser.add_argument("--ignore_index", default=1, type=int,
                    help='Specifies a target value that is ignored and does not contribute to the input gradient')

parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# CRF option
parser.add_argument("--use_crf", action="store_true", help="Whether to use CRF")
parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")
parser.add_argument("--do_infer", action="store_false")
args = parser.parse_args()

tokenizer = RobertaTokenizer.from_pretrained("roberta-base")
intent_label_lst = get_intent_labels(args)
dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}
slot_label_lst = get_slot_labels(args)
dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}
def post_processing(input_ids, slots):
    list_raw_slots = []
    for index in range(input_ids.shape[0]):
        list_tokens = tokenizer.convert_ids_to_tokens(input_ids[index])
        list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
        list_index_g = [1] + list_index_g
        raw_slots = []
        for j, token in enumerate(list_tokens):
            if j in list_index_g:
                raw_slots.append(slots[index][j])
        list_raw_slots.append(raw_slots)
    return list_raw_slots

device = torch.device("cpu")
sentences = ["I hit a customer", "I would make a complaint"]
inputs = tokenizer([sent.strip() for sent in sentences], add_special_tokens=True, return_tensors='pt', padding=True, truncation=True)
# inputs['input_ids'] = inputs['input_ids'].to(device)
# inputs['attention_mask'] = inputs['attention_mask'].to(device)

onnx_inputs = {'input_ids': np.array(inputs['input_ids']),
               'attention_mask': np.array(inputs['attention_mask'])}
# ============================================================================

model_path = "/home/sangdt/research/JointBert/ckpt_no_crf/model.onnx"
sess = onnxruntime.InferenceSession(model_path, providers=['CUDAExecutionProvider'])

inputs_name = [x.name for x in sess.get_inputs()]
inputs_shape = [x.shape for x in sess.get_inputs()]
# get output
outputs_name = [x.name for x in sess.get_outputs()]
outputs_shape = [x.shape for x in sess.get_outputs()]

print(inputs_name, inputs_shape)
print(outputs_name, outputs_shape)

intent_preds, slot_preds = sess.run(outputs_name, onnx_inputs)


# ============================================================================
intention =  [dict_intents[i] for i in intent_preds.tolist()]
tmp_slots = [[dict_tags[i] for i in slot_pred.tolist()] for slot_pred in slot_preds]
slots = post_processing(inputs['input_ids'], tmp_slots)


print([{"tokens": sentence.strip().split(), "intention": intention, "slots": slot} 
                    for sentence, intention, slot in zip(sentences, intention, slots)])