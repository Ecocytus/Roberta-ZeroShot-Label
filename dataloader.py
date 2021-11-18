import torch
import numpy as np
import json
from torch.autograd import Variable

'''
Dataloader are based on dataloder from https://github.com/thunlp/VERNet
'''

INCORRECT_ID =0
CORRECT_ID = 1
SUB_ID = 2
PAD_ID = 3

def convert_label(string):
    lab_list = string.strip().split()
    new_labs = list()
    for lab in lab_list:
        if lab == "c":
            new_labs.append(CORRECT_ID)
        elif lab == "i":
            new_labs.append(INCORRECT_ID)
        else:
            raise ("Wrong Label")
    return new_labs

def reform_label(tokens, labels, tokenizer, max_seq_length):
    new_tokens = list()
    new_labels = list()
    assert len(tokens) == len(labels)
    for step, token in enumerate(tokens[:-1]):
        split_token = tokenizer.tokenize(token)
        if len(split_token) > 0:
            new_tokens.extend(split_token)
            new_labels.extend([labels[step]] + [SUB_ID] * (len(split_token) - 1))
    new_tokens = new_tokens[:max_seq_length] + [tokens[-1]]
    new_labels = new_labels[:max_seq_length] + [labels[-1]]
    if len(new_tokens) != len(new_labels):
        print (new_tokens)
        print (new_labels)
        print (tokens)
        print (labels)
    assert len(new_tokens) == len(new_labels)
    return new_tokens, new_labels

def tok2int_sent(example, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    src_tokens = example[0]
    src_labels = example[1]

    src_tokens, src_labels = reform_label(src_tokens, src_labels, tokenizer, max_seq_length)

    tokens = src_tokens
    tokens = ['<s>'] + tokens + ['</s>']
    labels = src_labels
    labels = [PAD_ID] + labels + [PAD_ID]

    tokens = tokens
    labels = labels
    input_ids = tokenizer.convert_tokens_to_ids(tokens)
    input_mask = [1] * len(input_ids)

    max_len = max_seq_length

    if len(input_ids) > max_len:
        # truncate too long sentences
        labels = labels[:max_len]
        input_ids = input_ids[:max_len]
        input_mask = input_mask[:max_len]
    else:
        # pad short sentences
        labels += [PAD_ID] * (max_len - len(input_ids))
        padding = [0] * (max_len - len(input_ids))
        input_ids += padding
        input_mask += padding

    assert len(input_ids) == max_len
    assert len(input_mask) == max_len
    assert len(labels) == max_len
    return input_ids, input_mask, labels

def tok2int_list(data, tokenizer, max_seq_length):
    """Loads a data file into a list of `InputBatch`s."""
    inp = list()
    msk = list()
    lab = list()
    sen_label = list()
    for example in data:
        input_id, input_mask, label = tok2int_sent(example, tokenizer, max_seq_length)
        sen_label.append(example[2])
        inp.append(input_id)
        msk.append(input_mask)
        lab.append(label)
    return inp, msk, lab, sen_label


def getSentenceLabels(labels):
    # sentence_label = [0, 0]
    # if 0 in labels:
    #     sentence_label[0] = 1
    # if 1 in labels:
    #     sentence_label[1] = 1
    # return sentence_label
    return [1] if INCORRECT_ID in labels else [0]

class DataLoader(object):
    ''' For data iteration '''

    def __init__(self, data_path, tokenizer, max_len, test=False, batch_size=64, src_flag=True):
        self.batch_size = batch_size
        self.tokenizer = tokenizer
        self.max_len = max_len
        self.data_path = data_path
        self.test = test
        self.src_flag = src_flag
        examples = self.read_file(data_path)
        self.examples = examples
        self.total_num = len(examples)
        if self.test:
            self.total_step = np.ceil(self.total_num * 1.0 / batch_size)
        else:
            self.total_step = self.total_num / batch_size
            self.shuffle()
        self.step = 0

    def read_file(self, data_path):
        data_list = list()
        with open(data_path) as fin:
            for line in fin:
                examples = list()
                line = line.strip()
                data = json.loads(line)
                if 'http' in data["src"]:
                    continue
                src_token = data["src"].strip().split()[:-1]
                src_label = [PAD_ID] * len(src_token)
                if self.src_flag:
                    src_label = convert_label(data["src_lab"])[:-1]
                    sentence_label =  getSentenceLabels(src_label)
                assert len(src_token) == len(src_label)
                data_list.append([src_token, src_label, sentence_label])
        return data_list


    def shuffle(self):
        np.random.shuffle(self.examples)

    def __iter__(self):
        return self

    def __next__(self):
        return self.next()

    def __len__(self):
        return self._n_batch

    def next(self):
        ''' Get the next batch '''
        if self.step < self.total_step:
            examples = self.examples[self.step * self.batch_size : (self.step+1)*self.batch_size]
            inp, msk, score, sen_label = tok2int_list(examples, self.tokenizer, self.max_len)

            inp_tensor = Variable(
                torch.LongTensor(inp))
            msk_tensor = Variable(
                torch.LongTensor(msk))
            score_tensor = Variable(
                torch.LongTensor(score))

            sen_label_tensor = Variable(
                torch.FloatTensor(sen_label))

            self.step += 1
            return inp_tensor, msk_tensor, score_tensor, sen_label_tensor

        else:
            self.step = 0
            if not self.test:
                self.shuffle()
            raise StopIteration()

