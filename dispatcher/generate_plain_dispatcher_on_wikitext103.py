import os
import random

import torch
import torch.nn as nn
from transformers import BertTokenizer

from dispatcher.dispatcher_model import DispatcherModel

device = "cpu"

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/save_plain_dispatcher_wikitext103')

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

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
ntokens = bert_tokenizer.vocab_size

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt, dropout).to(device)
checkpoint = torch.load(_save_filename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def generate_text(prompt, num_choices=1, num_tokens=200):
    text = prompt

    length = 0
    inputs = bert_tokenizer.encode(text, return_tensors='pt').to(device)
    inputs = inputs[:, :-1]
    is_between_brackets = False
    while length < num_tokens:
        output = model(inputs.transpose(0, 1)).transpose(0, 1).cpu().detach().numpy()[0]
        item = output[-1]
        indices = range(len(item))
        indices_and_pos = zip(indices, item)
        indices_and_pos = sorted(indices_and_pos, key=lambda x: -x[1])
        choices = num_choices
        if is_between_brackets:
            choices = 1
        pos = random.choice(indices_and_pos[:choices])[0]
        inputs = torch.cat([inputs, torch.tensor([[pos]]).to(device)], dim=1)
        if pos == 133 and not is_between_brackets:
            is_between_brackets = True
        if pos == 135 and is_between_brackets:
            is_between_brackets = False

        length += 1

    text = bert_tokenizer.decode(inputs[0][1:])
    text = text.replace('< eos > ', '\n')
    text = text.replace('< unk >', '<unk>')
    return text.encode('ascii', errors='ignore').decode('ascii')


if __name__ == "__main__":
    text = 'The team won the cup'
    _choices = 1

    text = generate_text(text, num_choices=_choices)
    print(text)
