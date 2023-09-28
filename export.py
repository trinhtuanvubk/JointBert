
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
from utils.util import get_intent_labels, get_slot_labels 
import argparse
import numpy as np

def post_processing(input_ids, slots):
    list_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
    list_index_g = [1] + list_index_g
    raw_slots = []
    for index, token in enumerate(list_tokens):
        if index in list_index_g:
            raw_slots.append(slots[index])
    return raw_slots

class JointRobertaWrapper(torch.nn.Module):
    """Class for wrap model"""
   
    def __init__(self, joint_roberta):
        super().__init__()
        self.model = joint_roberta

    def forward(self, input_ids, attention_mask):
        # intent_logits, slot_logits = self.model.predict(input_ids, attention_mask)  # sequence_output, pooler_output, (hidden_states), (attentions)
        outputs = self.model.roberta(input_ids, attention_mask)  # sequence_output, pooler_output, (hidden_states), (attentions)
        sequence_output = outputs[0]
        pooler_output = outputs[1]  # ([<s>] (equivalent to [CLS])

        intent_logits = self.model.intent_classifier(pooler_output)
        slot_logits = self.model.slot_classifier(sequence_output)
        transitions = torch.tensor(self.model.crf.transitions.data)
        start_ = torch.tensor(self.model.crf.start_transitions.data)
        end_ = torch.tensor(self.model.crf.end_transitions.data)

        # intent_preds = np.argmax(intent_logits.detach().cpu().numpy(), axis=1)
        # slot_preds = np.array(self.model.crf.decode(slot_logits))
        # intent_preds = torch.argmax(intent_logits.detach(), axis=1)
        # slot_preds = slot_logits.detach().cpu().numpy().tolist()
        # slot_preds_list = [torch.argmax(slot_pred, axis=1) for slot_pred in slot_logits.detach()]
        # slot_preds = torch.stack(slot_preds_list, dim=0)
        # slot_preds = (self.model.crf.decode(slot_logits))
        # return intent_preds, slot_preds
        return intent_logits, slot_logits, transitions, start_, end_
        # return intent_logits, slot_logits, self.model.crf.transitions.data, self.model.crf.start_transitions.data, self.model.crf.end_transitions.data


parser = argparse.ArgumentParser()
parser.add_argument("--sentence", default="customer service", required=False, type=str, help="Enter an input sentence")
parser.add_argument("--model_dir", default="/home/sangdt/research/JointBert/ckpt", required=False, type=str, help="Path to save, load model")
parser.add_argument("--intent_label_file", default="/home/sangdt/research/JointBert/processed_data/intent_label.txt", type=str, help="Intent Label file")
parser.add_argument("--slot_label_file", default="/home/sangdt/research/JointBert/processed_data/slot_label.txt", type=str, help="Slot Label file")
parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
parser.add_argument("--ignore_index", default=1, type=int,
                    help='Specifies a target value that is ignored and does not contribute to the input gradient')
# parser.add_argument("--do_infer", action='store_false')
parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

# CRF option
parser.add_argument("--use_crf", action="store_false", help="Whether to use CRF")
parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

args = parser.parse_args()



device = torch.device("cuda")
ckpt = "./onnx/"
onnx_path = "./onnx/model.onnx"
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')

dummy_inputs = tokenizer(["customer service", "I booked something"], add_special_tokens=True, truncation=True, return_tensors="pt", padding=True)

dummy_inputs["input_ids"] = dummy_inputs['input_ids'].to(device)
dummy_inputs["attention_mask"] = dummy_inputs['attention_mask'].to(device)
intent_label_lst = get_intent_labels(args)
dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}
slot_label_lst = get_slot_labels(args)
dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}
model_base = JointRoberta.from_pretrained(args.model_dir, args, intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst).to(device)
print(model_base.crf.transitions.data)
model = JointRobertaWrapper(model_base)

# intent_preds, slot_preds = model(**dummy_inputs)
# intention =  [dict_intents[i] for i in intent_preds][0]
# tmp_slots = [dict_tags[i] for i in slot_preds[0]]
# slots = post_processing(dummy_inputs['input_ids'][0], tmp_slots)

# # export
torch.onnx.export(
    model, 
    tuple([dummy_inputs["input_ids"], dummy_inputs["attention_mask"]]),
    f="./ckpt/model.onnx",  
    verbose=True,
    input_names=['input_ids', 'attention_mask'], 
    output_names=['intent_logits', 'slot_logits', 'transitions', 'start_transition', 'end_transition'], 
    dynamic_axes={'input_ids': {0: 'batch_size', 1: 'max_length'}, 
                  'attention_mask': {0: 'batch_size', 1: 'max_length'},
                  'intent_logits': {0: 'batch_size'},
                  'slot_logits': {0: 'batch_size', 1: 'max_length'}},
    do_constant_folding=True, 
)
