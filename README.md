# Code for the paper [Dispatcher: A Message Passing Approach To Language Modelling](https://arxiv.org/abs/2105.03994)

This code illustrates the Dispatcher algorithm as presented in the paper. 

![shift_and_sum](https://user-images.githubusercontent.com/9697264/189527146-e9ba47d0-81fb-4f5e-8bfb-be8f25769a2b.gif)

## Installation
```bash
virtualenv --python=/usr/bin/python3 .env
pip install -r requirements.txt
```

## Training the models
The models can be trained anew using the following scripts

```bash
train_dispatcher_after_openwebtext_wikitext2.py
train_dispatcher_after_openwebtext_wikitext103.py
train_msa_wikitext2.py
train_msa_wikitext103.py
train_plain_dispatcher_on_wikitext2.py
train_plain_dispatcher_on_wikitext103.py 
```

## Evaluation
The perplexity of the pre-trained models can be evaluated using the following scripts
```bash
test_dispatcher_after_openwebtext_on_wikitext2.py
test_dispatcher_after_openwebtext_on_wikitext103.py
test_plain_dispatcher_on_wikitext2.py
test_plain_dispatcher_on_wikitext103.py
```

The plain dispatcher has about 30% more parameters on Wikitext103 because of a slighly different tokenization technique.
The vocabulary of tokens is smaller on Wikitext2 to achieve a better performance.

## Code
The Dispatcher is identical to the Transformer architecture with one crucial difference:
the self-attention layer is substituted with the Dispatcher layer.

This algorithm - explained in the paper - is contained in the file [dispatcher_model.py](dispatcher/dispatcher_model.py)
The following code is this work's main contribution:
```python
class DispatcherLayer(nn.Module):
    def __init__(self, embed_dim, num_heads, bptt, dropout=0.):
        super(DispatcherLayer, self).__init__()

        self._levels = int(math.log(bptt, 2))
        self._shifts = [pow(2, i) for i in range(self._levels)]

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.dropout = dropout

        self.linear_in = nn.Linear(self.head_dim, self.head_dim, bias=False)
        self.internal_attention = nn.Linear(self.head_dim, self._levels, bias=False)
        self.linear_out = nn.Linear(self.head_dim, self.head_dim, bias=False)

    def forward(self, value, mask):
        inp = value.transpose(1, 0)
        batch_length = inp.shape[0]
        length = inp.shape[1]
        inp = inp.reshape(batch_length * self.num_heads, length, self.head_dim)

        V = self.linear_in(inp)

        coefficient_tensor = F.sigmoid(self.internal_attention(inp)) * mask.detach()
        coefficient = torch.chunk(coefficient_tensor, chunks=self._levels, dim=2)

        for c, shift in zip(coefficient, self._shifts):
            if shift > length:
                break
            if self.training and random.uniform(0, 1) < self.dropout:
                continue
            V += c * torch.roll(V, shifts=shift, dims=1)

        out = self.linear_out(V)
        out = out.reshape(batch_length, length, self.embed_dim)
        return out.transpose(1, 0)
```

The main loop is in the forward() method, where the _shift and sum_ steps are applied (see the paper).

A second file contains the "standard" Masked Self-Attention model [msa_model.py](dispatcher/msa_model.py).
The two models are nearly identical, with the exception of the Dispatcher layers.



## Run the code
A notebook is included here to run the code and generate texts using the various models: 
[dispatcher.ipynb](notebooks/dispatcher.ipynb).





