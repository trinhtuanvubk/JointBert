from model import JointRoberta
from transformers import RobertaTokenizer
import torch
from utils import get_intent_labels, get_slot_labels 
import argparse
import numpy as np

def post_processing(input_ids, slots):
    # input_ids: torch.size(5), slots: list(5)
    list_tokens = tokenizer.convert_ids_to_tokens(input_ids)
    list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
    list_index_g = [1] + list_index_g
    raw_slots = []
    for index, token in enumerate(list_tokens):
        if index in list_index_g:
            raw_slots.append(slots[index])
    return raw_slots

def infer(sentence, tokenizer, model, dict_intents, dict_tags, device):
    inputs = tokenizer([sentence.strip()], add_special_tokens=True, return_tensors='pt')
    inputs['input_ids'] = inputs['input_ids'].to(device)
    inputs['attention_mask'] = inputs['attention_mask'].to(device)
    with torch.no_grad():
        outputs = model.predict(**inputs)
        intent_logits, slot_logits = outputs
        intent_preds = np.argmax(intent_logits.detach().cpu().numpy(), axis=1).tolist()
        # shape: (1,30)
        slot_preds = np.array(model.crf.decode(slot_logits)).tolist()
        # len: 5
    intention =  [dict_intents[i] for i in intent_preds][0]
    tmp_slots = [dict_tags[i] for i in slot_preds[0]]
    slots = post_processing(inputs['input_ids'][0], tmp_slots)
    return {
        "tokens": sentence.strip().split(),
        "intention": intention,
        "slots": slots
    }
    

if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    parser.add_argument("--sentence", default=None, type=str, help="Enter an input sentence")
    parser.add_argument("--model_dir", default="/home/sangdt/research/JointBert/ckpt", type=str, help="Path to save, load model")
    parser.add_argument("--intent_label_file", default="./processed_data/intent_label.txt", type=str, help="Intent Label file")
    parser.add_argument("--slot_label_file", default="./processed_data/slot_label.txt", type=str, help="Slot Label file")
    parser.add_argument("--dropout_rate", default=0.1, type=float, help="Dropout for fully-connected layers")
    parser.add_argument("--ignore_index", default=1, type=int,
                        help='Specifies a target value that is ignored and does not contribute to the input gradient')

    parser.add_argument('--slot_loss_coef', type=float, default=1.0, help='Coefficient for the slot loss.')

    # CRF option
    parser.add_argument("--use_crf", action="store_false", help="Whether to use CRF")
    parser.add_argument("--slot_pad_label", default="PAD", type=str, help="Pad token for slot label pad (to be ignore when calculate loss)")

    args = parser.parse_args()

    intent_label_lst = get_intent_labels(args)
    dict_intents = {i: intent for i, intent in enumerate(intent_label_lst)}
    slot_label_lst = get_slot_labels(args)
    dict_tags = {i: tag for i, tag in enumerate(slot_label_lst)}

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = JointRoberta.from_pretrained(args.model_dir, args, intent_label_lst=intent_label_lst, slot_label_lst=slot_label_lst).to(device)
    tokenizer = RobertaTokenizer.from_pretrained("roberta-base")

    result = infer('customer service', tokenizer, model, dict_intents, dict_tags, device)
    print(result)