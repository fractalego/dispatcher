import math
import os
import random
import sys

import time
import torch
import torch.nn as nn
import torchtext
import subprocess

from tqdm import tqdm
from transformers import GPT2Tokenizer
from dispatcher.msa_model import TransformerModel

batch_size = 6
eval_batch_size = 2

bptt = 1024
emsize = 480  # embedding dimension
nhid = 480  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 12  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1
dropout = 0.0  # the dropout value

criterion = nn.CrossEntropyLoss()
lr = 6e-5  # learning rate
best_val_loss = float("inf")
epochs = 60  # The number of epochs

_path = os.path.dirname(__file__)
_save_filename = os.path.join(_path, '../data/dispatcher-openwebtext-no-dropout')

bert_tokenizer = GPT2Tokenizer.from_pretrained('gpt2')


def tokenizer(text):
    return bert_tokenizer.tokenize(text)


TEXT = torchtext.data.Field(tokenize=tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = bert_tokenizer.vocab_size
torch.autograd.set_detect_anomaly(True)


def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


model = TransformerModel(ntokens, emsize, nhead, nhid, nlayers).to(device)
print('num_parameters:', count_parameters(model))
optimizer = torch.optim.Adam(model.parameters(), lr=lr)


def batchify(data, bsz):
    data = bert_tokenizer.encode(data, return_tensors='pt')[0]
    # Divide the dataset into bsz parts.
    nbatch = data.size(0) // bsz
    # Trim off any extra elements that wouldn't cleanly fit (remainders).
    data = data.narrow(0, 0, nbatch * bsz)
    # Evenly divide the data across the bsz batches.
    data = data.view(bsz, -1).t().contiguous()
    return data.to(device)


def get_batch(source, i):
    seq_len = min(bptt, len(source) - 1 - i)
    data = source[i:i + seq_len]
    target = source[i + 1:i + 1 + seq_len].view(-1)
    return data, target




def train():
    model.train()  # Turn on the train mode
    total_loss = 0.
    start_time = time.time()
    for batch, i in enumerate(range(0, train_data.size(0) - 1, bptt)):
        data, targets = get_batch(train_data, i)
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output.view(-1, ntokens), targets)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 0.5)
        optimizer.step()
        total_loss += loss.item()

    if not batch:
        batch = 1

    cur_loss = total_loss / batch
    elapsed = time.time() - start_time
    print('|{:5d}/{:5d} batches | '
          'lr {:02.2f} | '  # ms/batch {:5.2f} | '
          'loss {:5.2f} | ppl {:8.2f} |'.format(
        batch, len(train_data) // bptt,  # scheduler.get_lr()[0],
               elapsed * 1000 / batch,
        cur_loss, math.exp(cur_loss)))
    sys.stdout.flush()


def evaluate(eval_model, data_source):
    eval_model.eval()  # Turn on the evaluation mode
    total_loss = 0.
    with torch.no_grad():
        for i in range(0, data_source.size(0) - 1, bptt):
            data, targets = get_batch(data_source, i)
            output = eval_model(data)
            output_flat = output.view(-1, ntokens)
            total_loss += len(data) * criterion(output_flat, targets).item()
    return total_loss / (len(data_source) - 1)


best_model = None

if __name__ == '__main__':

    datapath = os.path.join(_path, '../data/openwebtext/')
    files = [file for file in os.listdir(datapath) if '.xz' not in file]
    random.shuffle(files)
    train_ratio = 1.
    dev_size = 100
    train_files = files[:int(len(files) * train_ratio) - dev_size]
    dev_files = files[int(len(files) * train_ratio) - dev_size:int(len(files) * train_ratio)]
    print(dev_files)
    sys.stdout.flush()
    trace_every = 400

    all_val_txt = ''
    for tarball in tqdm(dev_files):
        cmd = ['tar', '-Oxvf', os.path.join(datapath, tarball)]
        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        all_val_txt += result.stdout.decode('utf-8')
    val_data = batchify(all_val_txt, eval_batch_size)

    index = 0
    for epoch in range(epochs):
        random.shuffle(train_files)
        print(train_files)
        for tarball in tqdm(train_files):
            cmd = ['tar', '-Oxvf', os.path.join(datapath, tarball)]
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            train_txt = result.stdout.decode('utf-8')
            train_data = batchify(train_txt, batch_size)
            train()
            del train_txt

            if (index + 1) % trace_every == 0:
                val_loss = evaluate(model, val_data)
                print('Validation:')
                print('-' * 89)
                print('| end of epoch {:3d} | valid loss {:5.2f} | '
                      'valid ppl {:8.2f}'.format(index, val_loss, math.exp(val_loss)))
                print('-' * 89)
                sys.stdout.flush()

                torch.save({
                    'epoch': index,
                    'model_state_dict': model.state_dict()},
                    _save_filename + str(epoch) + '_' + str(index))
            index += 1

    val_loss = evaluate(model, val_data)
    print('Final Validation:')
    print('-' * 89)
    print('| valid loss {:5.2f} | valid ppl {:8.2f}'.format(val_loss, math.exp(val_loss)))
    print('-' * 89)

    torch.save({
        'epoch': -1,
        'model_state_dict': model.state_dict()},
        _save_filename + 'final')
