import os
import random

import torch
from transformers import GPT2Tokenizer

from dispatcher.dispatcher_model import DispatcherModel

device = "cpu"

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/save_dispatcher_after_openwebtext_wikitext2')

bptt = 1024
emsize = 480  # embedding dimension
nhid = 480  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # the number of heads in the multiheadattention models
dropout = 0.  # the dropout value

bert_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ntokens = bert_tokenizer.vocab_size

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt).to(device)
checkpoint = torch.load(_save_filename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()


def generate_text(prompt, num_choices=1, num_tokens=200):
    length = 0
    inputs = bert_tokenizer.encode(prompt, return_tensors='pt').to(device)

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
        length += 1

        if pos == 1279 and not is_between_brackets:
            is_between_brackets = True
        if pos == 29 and is_between_brackets:
            is_between_brackets = False

    text = bert_tokenizer.decode(inputs[0])
    text = text.replace('<eos> ', '\n')

    return text.encode('ascii', errors='ignore').decode('ascii')


if __name__ == "__main__":
    text = 'The team won the cup'
    _choices = 1

    text = generate_text(text, num_choices=_choices)
    print(text)
