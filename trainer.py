import os
import logging
from tqdm import tqdm, trange

import numpy as np
import torch
from torch.utils.data import DataLoader
from transformers.models.roberta.modeling_roberta import RobertaConfig
from transformers import AdamW, get_linear_schedule_with_warmup
from utils import MODEL_CLASSES, compute_metrics, get_intent_labels, get_slot_labels, load_tokenizer


logger = logging.getLogger(__name__)

def transform(tokenizer, sentences, list_slots, list_intents, dict_tags, dict_intents):
    new_list_slots = []
    list_ids_slots = []
    list_ids_intents = []
    list_pre = tokenizer([sent.strip() for sent in sentences], padding=True, truncation=True, add_special_tokens=True)
    for i in range(len(sentences)):
        list_tokens = tokenizer.convert_ids_to_tokens(list_pre['input_ids'][i])
        list_index_g = [index for index, i in enumerate(list_tokens) if 'Ġ' in i]
        list_index_g = [1] + list_index_g
        new_slots = []
        for index, token in enumerate(list_tokens):
            if index not in list_index_g:
                if token in ['<s>', '</s>', '<pad>']:
                    ner_tag = 'O'
                else:
                    tmp = [i for i in list_index_g if i < index][-1]
                    ner_tag = list_slots[i][list_index_g.index(tmp)].replace("B-", "I-")
            else:
                ner_tag = list_slots[i][list_index_g.index(index)]
            new_slots.append(ner_tag)
        new_list_slots.append(new_slots)

    for slots in new_list_slots:
        list_ids_slots.append([dict_tags[i] for i in slots])
    
    for intent in list_intents:
        list_ids_intents.append(dict_intents[intent])

    return (
        torch.tensor(list_pre['input_ids']),
        torch.tensor(list_pre['attention_mask']),
        torch.tensor(list_ids_slots),
        torch.tensor(list_ids_intents),
        torch.tensor([[0] * len(list_pre['input_ids'][0]) for _ in range(len(sentences))])
    )


class Trainer(object):
    def __init__(self, args, train_dataset=None, test_dataset=None):
        self.args = args
        self.train_dataset = train_dataset
        self.test_dataset = test_dataset

        self.intent_label_lst = get_intent_labels(args)
        self.dict_intents = {intent: i for i, intent in enumerate(self.intent_label_lst)}
        self.slot_label_lst = get_slot_labels(args)
        self.dict_tags = {tag: i for i, tag in enumerate(self.slot_label_lst)}
        # Use cross entropy ignore index as padding label id so that only real label ids contribute to the loss later
        self.pad_token_label_id = args.ignore_index

        self.config_class, self.model_class, _ = MODEL_CLASSES[args.model_type]
        self.config = self.config_class.from_pretrained(args.model_name_or_path, finetuning_task='joint')
        self.tokenizer = load_tokenizer(args)
        self.model = self.model_class.from_pretrained(args.model_name_or_path,
                                                      config=self.config,
                                                      args=args,
                                                      intent_label_lst=self.intent_label_lst,
                                                      slot_label_lst=self.slot_label_lst)
        self.device = "cuda" if torch.cuda.is_available() and not args.no_cuda else "cpu"
        self.model.to(self.device)

    def train(self):
        train_dataloader = DataLoader(self.train_dataset, shuffle=True, batch_size=self.args.train_batch_size, collate_fn=self.collate_fn)

        if self.args.max_steps > 0:
            t_total = self.args.max_steps
            self.args.num_train_epochs = self.args.max_steps // (len(train_dataloader) // self.args.gradient_accumulation_steps) + 1
        else:
            t_total = len(train_dataloader) // self.args.gradient_accumulation_steps * self.args.num_train_epochs

        # Prepare optimizer and schedule (linear warmup and decay)
        no_decay = ['bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in self.model.named_parameters() if not any(nd in n for nd in no_decay)],
             'weight_decay': self.args.weight_decay},
            {'params': [p for n, p in self.model.named_parameters() if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self.args.learning_rate, eps=self.args.adam_epsilon)
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=self.args.warmup_steps, num_training_steps=t_total)
        
        # Train!
        logger.info("***** Running training *****")
        logger.info("  Num examples = %d", len(self.train_dataset))
        logger.info("  Num Epochs = %d", self.args.num_train_epochs)
        logger.info("  Total train batch size = %d", self.args.train_batch_size)
        logger.info("  Gradient Accumulation steps = %d", self.args.gradient_accumulation_steps)
        logger.info("  Total optimization steps = %d", t_total)
        logger.info("  Logging steps = %d", self.args.logging_steps)
        logger.info("  Save steps = %d", self.args.save_steps)

        global_step = 0
        tr_loss = 0.0
        self.model.zero_grad()

        train_iterator = trange(int(self.args.num_train_epochs), desc="Epoch")
        for _ in train_iterator:
            epoch_iterator = tqdm(train_dataloader, desc="Iteration")
            for step, batch in enumerate(epoch_iterator):
                self.model.train()
                batch = tuple(t.to(self.device) for t in batch)  # GPU or CPU

                inputs = {'input_ids': batch[0],
                        'attention_mask': batch[1],
                        'intent_label_ids': batch[3],
                        'slot_labels_ids': batch[2],
                        'token_type_ids': batch[4]}
                outputs = self.model(**inputs)
                loss = outputs[0]

                if self.args.gradient_accumulation_steps > 1:
                    loss = loss / self.args.gradient_accumulation_steps

                loss.backward()

                tr_loss += loss.item()
                if (step + 1) % self.args.gradient_accumulation_steps == 0:
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)

                    optimizer.step()
                    scheduler.step()  # Update learning rate schedule
                    self.model.zero_grad()
                    global_step += 1

                    if self.args.logging_steps > 0 and global_step % self.args.logging_steps == 0:
                        self.evaluate()

                    if self.args.save_steps > 0 and global_step % self.args.save_steps == 0:
                        self.save_model()

                if 0 < self.args.max_steps < global_step:
                    epoch_iterator.close()
                    break

            if 0 < self.args.max_steps < global_step:
                train_iterator.close()
                break

        return global_step, tr_loss / global_step

    def evaluate(self):

        eval_dataloader = DataLoader(self.test_dataset, batch_size=self.args.eval_batch_size, collate_fn=self.collate_fn)

        # Eval!
        logger.info("***** Running evaluation on test dataset *****",)
        logger.info("  Num examples = %d", len(self.test_dataset))
        logger.info("  Batch size = %d", self.args.eval_batch_size)
        eval_loss = 0.0
        nb_eval_steps = 0
        intent_preds = None
        slot_preds = None
        out_intent_label_ids = None
        out_slot_labels_ids = None

        self.model.eval()

        for batch in tqdm(eval_dataloader, desc="Evaluating"):
            batch = tuple(t.to(self.device) for t in batch)
            with torch.no_grad():
                inputs = {'input_ids': batch[0],
                            'attention_mask': batch[1],
                            'intent_label_ids': batch[3],
                            'slot_labels_ids': batch[2],
                            'token_type_ids': batch[4]}
                outputs = self.model(**inputs)
                tmp_eval_loss, (intent_logits, slot_logits) = outputs[:2]

                eval_loss += tmp_eval_loss.mean().item()
            nb_eval_steps += 1

            # Intent prediction
            if intent_preds is None:
                intent_preds = intent_logits.detach().cpu().numpy()
                out_intent_label_ids = inputs['intent_label_ids'].detach().cpu().numpy()
            else:
                intent_preds = np.append(intent_preds, intent_logits.detach().cpu().numpy(), axis=0)
                out_intent_label_ids = np.append(
                    out_intent_label_ids, inputs['intent_label_ids'].detach().cpu().numpy(), axis=0)

            # Slot prediction
            if slot_preds is None:
                if self.args.use_crf:
                    # decode() in `torchcrf` returns list with best index directly
                    slot_preds = np.array(self.model.crf.decode(slot_logits)).tolist()
                else:
                    slot_preds = slot_logits.detach().cpu().numpy().tolist()

                out_slot_labels_ids = inputs["slot_labels_ids"].detach().cpu().numpy().tolist()
            else:
                if self.args.use_crf:
                    slot_preds.extend(np.array(self.model.crf.decode(slot_logits)).tolist())
                else:
                    slot_preds.extend(slot_logits.detach().cpu().numpy().tolist())

                out_slot_labels_ids.extend(inputs["slot_labels_ids"].detach().cpu().numpy().tolist())

        eval_loss = eval_loss / nb_eval_steps
        results = {
            "loss": eval_loss
        }

        # Intent result
        intent_preds = np.argmax(intent_preds, axis=1)

        # Slot result
        if not self.args.use_crf:
            slot_preds =  [np.argmax(slot_pred, axis=2) for slot_pred in slot_preds]
        slot_label_map =  {i: label for i, label in enumerate(self.slot_label_lst)}
        out_slot_label_list = [[] for _ in range(len(self.test_dataset))]
        slot_preds_list = [[] for _ in range(len(self.test_dataset))]

        for i in range(len(out_slot_labels_ids)):
            for j in range(len(out_slot_labels_ids[i])):
                if out_slot_labels_ids[i][j] != self.pad_token_label_id:
                    out_slot_label_list[i].append(slot_label_map[out_slot_labels_ids[i][j]])
                    slot_preds_list[i].append(slot_label_map[slot_preds[i][j]])

        total_result = compute_metrics(intent_preds, out_intent_label_ids, slot_preds_list, out_slot_label_list)
        results.update(total_result)

        logger.info("***** Eval results *****")
        for key in sorted(results.keys()):
            logger.info("  %s = %s", key, str(results[key]))

        return results

    def collate_fn(self, batch):
        sentences = [example['text'] for example in batch]
        list_slots = [example['slots'] for example in batch]
        list_intents = [example['intention'] for example in batch]
        return transform(self.tokenizer, sentences, list_slots, list_intents, self.dict_tags, self.dict_intents)

    def save_model(self):
        # Save model checkpoint (Overwrite)
        if not os.path.exists(self.args.model_dir):
            os.makedirs(self.args.model_dir)
        model_to_save = self.model.module if hasattr(self.model, 'module') else self.model
        model_to_save.save_pretrained(self.args.model_dir)

        # Save training arguments together with the trained model
        torch.save(self.args, os.path.join(self.args.model_dir, 'training_args.bin'))
        logger.info("Saving model checkpoint to %s", self.args.model_dir)

    def load_model(self):
        # Check whether model exists
        if not os.path.exists(self.args.model_dir):
            raise Exception("Model doesn't exists! Train first!")

        try:
            self.model = self.model_class.from_pretrained(self.args.model_dir,
                                                          args=self.args,
                                                          intent_label_lst=self.intent_label_lst,
                                                          slot_label_lst=self.slot_label_lst)
            self.model.to(self.device)
            logger.info("***** Model Loaded *****")
        except:
            raise Exception("Some model files might be missing...")