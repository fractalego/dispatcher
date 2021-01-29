import math
import torch
import torchtext
import torch.nn as nn
from transformers import BertTokenizer
from dispatcher.dispatcher_model import DispatcherModel
from dispatcher.utils import count_parameters, get_batch

eval_batch_size = 10

bptt = 512
emsize = 512  # embedding dimension
nhid = 512  # the dimension of the feedforward network model in nn.TransformerEncoder
nlayers = 6  # the number of nn.TransformerEncoderLayer in nn.TransformerEncoder
nhead = 1  # the number of heads in the multiheadattention models
dropout = 0.2  # the dropout value

bert_tokenizer = BertTokenizer.from_pretrained('bert-base-cased')


def tokenizer(text):
    return text.split()


TEXT = torchtext.data.Field(tokenize=tokenizer,
                            init_token='<sos>',
                            eos_token='<eos>',
                            lower=True)
_, _, test_txt = torchtext.datasets.WikiText103.splits(TEXT)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
ntokens = bert_tokenizer.vocab_size

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt, dropout).to(device)
checkpoint = torch.load('../data/save_plain_dispatcher_wikitext103')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()
criterion = nn.CrossEntropyLoss()


def batchify(data, bsz):
    all_data = []
    _step = 65536 * 16
    for index in range(0, len(data.examples[0].text), _step):
        all_data += bert_tokenizer.encode(' '.join(data.examples[0].text[index:index + _step]))

    data = torch.Tensor(all_data)
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
