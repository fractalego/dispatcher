import math
import os

import torch
import torch.nn as nn
import torchtext
from transformers import BertTokenizer

from dispatcher.msa_model import TransformerModel
from dispatcher.utils import get_batch

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
epochs = 60  # The number of epochs

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/save_ma')

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenizer(text):
    return bert_tokenizer.tokenize(text)


TEXT = torchtext.data.Field(tokenize=tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
train_txt, val_txt, test_txt = torchtext.datasets.WikiText2.splits(TEXT)
TEXT.build_vocab(train_txt)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = len(TEXT.vocab.stoi)  # the size of vocabulary
torch.autograd.set_detect_anomaly(True)

model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers, dropout).to(device)

optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def batchify(data, bsz):
    data = TEXT.numericalize([data.examples[0].text])
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


import time

train_data = batchify(train_txt, batch_size)
val_data = batchify(val_txt, eval_batch_size)
test_data = batchify(test_txt, eval_batch_size)


def train():
    import numpy as np

    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    ntokens = len(TEXT.vocab.stoi)

    times = []
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i, bptt)
        optimizer.zero_grad()
        start_time2 = time.time()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        end_time2 = time.time()
        times.append(end_time2 - start_time2)

        total_loss += loss.item()
        log_interval = 200

        if batch % log_interval == 0 and batch > 0:
            cur_loss = total_loss / log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | '
                  'lr {:02.2f} | '  # ms/batch {:5.2f} | '
                  'loss {:5.2f} | ppl {:8.2f} |'
                  'perplexity {:8.2f} |'.format(
                epoch, batch, len(train_data) // bptt,  # scheduler.get_lr()[0],
                              elapsed * 1000 / log_interval,
                cur_loss, math.exp(cur_loss), math.exp(cur_loss * emsize / bptt)))
            total_loss = 0
            start_time = time.time()
            # print('average time:', np.mean(times))
            times = []


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    ntokens = len(TEXT.vocab.stoi)
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i, bptt)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_model = None

if __name__ == '__main__':
    for epoch in range(1, epochs + 1):
        epoch_start_time = time.time()
        train()
        val_loss = evaluate(model, val_data)
        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
              'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                         val_loss, math.exp(val_loss)))
        print('-' * 89)

        test_loss = evaluate(model, test_data)
        print('=' * 89)
        print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
            test_loss, math.exp(test_loss)))
        print('=' * 89)

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_model = model

    import os

    _path = os.path.dirname(__file__)
    _save_filename = os.path.join(_path, '../data/save_ma')

    torch.save({
        'epoch': epoch,
        'model_state_dict': best_model.state_dict()},
        _save_filename + str(epoch))

    test_loss = evaluate(best_model, test_data)
    print('=' * 89)
    print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
        test_loss, math.exp(test_loss)))
    print('=' * 89)
