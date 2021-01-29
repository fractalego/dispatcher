import math
import os
import torch
import torch.nn as nn
import torchtext
from transformers import GPT2Tokenizer

from dispatcher.dispatcher_model import DispatcherModel
from dispatcher.utils import count_parameters
from dispatcher.utils import get_batch

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/save_dispatcher_after_openwebtext_wikitext2')

eval_batch_size = 10

bptt = 1024
emsize = 480  # embedding dimension
nhid = 480  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # the number of heads in the multiheadattention models

bert_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

ntokens = bert_tokenizer.vocab_size

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt).to(device)
checkpoint = torch.load(_save_filename)
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch.autograd.set_detect_anomaly(True)
ntokens = bert_tokenizer.vocab_size
criterion = nn.CrossEntropyLoss()


def tokenizer(text):
    return text


TEXT = torchtext.data.Field(tokenize=tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
_, _, test_txt = torchtext.datasets.WikiText2.splits(TEXT)


def batchify(data, bsz):
    data = torch.Tensor(bert_tokenizer.encode(''.join(data.examples[0].text)))
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.long().to(device)


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


test_data = batchify(test_txt, eval_batch_size)
loss = evaluate(model, test_data)

print('Num parameters:', count_parameters(model))
print('Test perplexity:', math.exp(loss))
