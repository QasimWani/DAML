### Response Decoder class implementation

import torch

from torch import nn
from torch import optim
import torch.nn.functional as F
from torch.autograd import Variable

import numpy as np
import math
import copy, random, time, logging

from torch.distributions import Categorical

import pdb
from attn import Attention


#### Reference: https://github.com/qbetterk/DAML

class ResponseDecoder(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, degree_size, dropout_rate, gru, proj, emb, vocab):
        super().__init__()
        self.emb = emb
        self.attn_z = Attention(hidden_size)
        self.attn_u = Attention(hidden_size)
        self.gru = gru
        self.proj = proj
        self.proj_copy1 = nn.Linear(hidden_size, hidden_size)
        self.proj_copy2 = nn.Linear(hidden_size, hidden_size)
        self.dropout_rate = dropout_rate
        self.vocab = vocab

    def get_sparse_selective_input(self, x_input_np):
        result = np.zeros((x_input_np.shape[0], x_input_np.shape[1], cfg.vocab_size + x_input_np.shape[0]), dtype=np.float32)
        result.fill(1e-10)
        reqs = ['address', 'phone', 'postcode', 'pricerange', 'area']
        for t in range(x_input_np.shape[0] - 1):
            for b in range(x_input_np.shape[1]):
                w = x_input_np[t][b]
                word = self.vocab.decode(w)
                if word in reqs:
                    slot = self.vocab.encode(word + '_SLOT')
                    result[t + 1][b][slot] = 1.0
                else:
                    if w == 2 or w >= cfg.vocab_size:
                        result[t+1][b][cfg.vocab_size + t] = 5.0
                    else:
                        result[t+1][b][w] = 1.0
        result_np = result.transpose((1, 0, 2))
        result = torch.from_numpy(result_np).float()
        return result

    def forward(self, z_enc_out, u_enc_out, u_input_np, m_t_input, degree_input, last_hidden, z_input_np):
        sparse_z_input = Variable(self.get_sparse_selective_input(z_input_np), requires_grad=False)

        m_embed = self.emb(m_t_input)
        z_context = self.attn_z(last_hidden, z_enc_out)
        u_context = self.attn_u(last_hidden, u_enc_out)
        gru_in = torch.cat([m_embed, u_context, z_context, degree_input.unsqueeze(0)], dim=2)
        gru_out, last_hidden = self.gru(gru_in, last_hidden)
        gen_score = self.proj(torch.cat([z_context, u_context, gru_out], 2)).squeeze(0)
        z_copy_score = F.tanh(self.proj_copy2(z_enc_out.transpose(0, 1)))
        z_copy_score = torch.matmul(z_copy_score, gru_out.squeeze(0).unsqueeze(2)).squeeze(2)
        z_copy_score = z_copy_score.cpu()
        z_copy_score_max = torch.max(z_copy_score, dim=1, keepdim=True)[0]
        z_copy_score = torch.exp(z_copy_score - z_copy_score_max)  # [B,T]
        z_copy_score = torch.log(torch.bmm(z_copy_score.unsqueeze(1), sparse_z_input)).squeeze(
            1) + z_copy_score_max  # [B,V]
        z_copy_score = cuda_(z_copy_score)

        scores = F.softmax(torch.cat([gen_score, z_copy_score], dim=1), dim=1)
        gen_score, z_copy_score = scores[:, :cfg.vocab_size], \
                                  scores[:, cfg.vocab_size:]
        proba = gen_score + z_copy_score[:, :cfg.vocab_size]  # [B,V]
        proba = torch.cat([proba, z_copy_score[:, cfg.vocab_size:]], 1)
        return proba, last_hidden, gru_out    