import os
import random

import torch
import torch.nn as nn
import torchtext
from transformers import BertTokenizer

from dispatcher.dispatcher_model import DispatcherModel

device = "cpu"

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/save_plain_dispatcher_wikitext_2_32')

batch_size = 20
eval_batch_size = 10

bptt = 512
emsize = 512  # embedding dimension
nhid = 512  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

criterion = nn.CrossEntropyLoss()
lr = 1e-4  # learning rate
best_val_loss = float("inf")
epochs = 40  # The number of epochs

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenizer(text):
    return bert_tokenizer.tokenize(text)


TEXT = torchtext.data.Field(tokenize=tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt, dropout).to(device)
checkpoint = torch.load(_save_filename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

inverse_vocab = {v: k for k, v in TEXT.vocab.stoi.items()}


def generate_text(prompt, num_choices=1, num_tokens=200):
    text = prompt
    is_between_brackets = False
    while len(text.split()) < num_tokens:
        tokens = tokenizer(text)
        data = TEXT.numericalize([tokens]).to(device)
        output = model(data).transpose(0, 1).cpu().detach().numpy()[0]
        item = output[-1]
        indices = range(len(item))
        indices_and_pos = zip(indices, item)
        indices_and_pos = sorted(indices_and_pos, key=lambda x: -x[1])
        choices = num_choices
        if is_between_brackets:
            choices = 1
        pos = random.choice(indices_and_pos[:choices])[0]
        new_word = inverse_vocab[pos]
        if new_word[:2] != '##':
            text += ' ' + inverse_vocab[pos]
        else:
            text += inverse_vocab[pos][2:]

        if inverse_vocab[pos] == '<' and not is_between_brackets:
            is_between_brackets = True
        if inverse_vocab[pos] == '>' and is_between_brackets:
            is_between_brackets = False

    return text.encode('ascii', errors='ignore').decode('ascii')


if __name__ == "__main__":
    text = 'The team won the cup'
    _choices = 1

    text = generate_text(text, num_choices=_choices)
    print(text)
