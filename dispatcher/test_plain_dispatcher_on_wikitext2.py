import math
import torch
from dispatcher.train_plain_dispatcher_on_wikitext2 import evaluate, test_data, ntokens, emsize, nhead, nhid, nlayers, bptt, dropout, device
from dispatcher.dispatcher_model import DispatcherModel
from dispatcher.utils import count_parameters

model = DispatcherModel(ntokens, emsize, nhead, nhid, nlayers, bptt, dropout).to(device)
checkpoint = torch.load('../data/save_plain_dispatcher_wikitext_2_32')
model.load_state_dict(checkpoint['model_state_dict'])
model.eval()

print('Num parameters:', count_parameters(model))

loss = evaluate(model, test_data)
print('Test perplexity:', math.exp(loss))