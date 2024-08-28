"""
    Multi-head attention remastered with 5min
    @author: Qianyue He
    @date:   2024.8.26
"""

import torch
import numpy as np

class MHAttention(torch.nn):
    def __init__(self, input_dim, out_dim, num_head = 8):
        # we need by-3 multiplication, since Q, K, V they all need to be mapped
        self.out_dim  = out_dim
        self.num_head = num_head 
        self.input_mapping = torch.nn.Linear(input_dim, out_dim * 3)
        self.out_mapping   = torch.nn.Linear(out_dim, out_dim)
        self.norm_coeff    = 1 / torch.sqrt(out_dim / num_head)

    def forward(self, input_seq: torch.Tensor):
        """ Input shape: (N_batch, N_seq_length, N_input_encoding_length)
            Nothing too difficult
        """
        batch_size, seq_len, _ = input_seq.shape
        qkv: torch.Tensor = self.input_mapping(input_seq)
        q, k, v = qkv.split((self.out_dim, self.out_dim, self.out_dim), dim = -1)       # slip q k v
        q: torch.Tensor = q.view(batch_size, seq_len, self.num_head, self.out_dim // self.num_head)
        k: torch.Tensor = k.view(batch_size, seq_len, self.num_head, self.out_dim // self.num_head)
        v: torch.Tensor = v.view(batch_size, seq_len, self.num_head, self.out_dim // self.num_head)

        qTk_scaled = q[..., None] @ k[..., None, :] * self.norm_coeff           # scaled qT mul K
        # this shape is (N, L, H, C, C)
        scores = torch.softmax(qTk_scaled, dim = -1)                            # softmax proba

        attention_output = (scores @ v.unsqueeze(dim = -1)).squeeze().view(batch_size, seq_len, -1)
        return self.out_mapping(attention_output)
