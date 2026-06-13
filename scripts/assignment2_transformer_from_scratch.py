import os

if "CONDA_DEFAULT_ENV" in os.environ:

    assert os.environ["CONDA_DEFAULT_ENV"] == "AI-24", "current environment is not AI-24"



import torch

import torch.nn as nn

import torch.optim as optim

import torch.utils.data as data

import math





if torch.cuda.is_available() is True:

    device = torch.device("cuda")

else:

    device = torch.device("cpu")


# example

example_arange1 = torch.arange(5)

print(example_arange1)

example_arange2 = torch.arange(1, 4)

print(example_arange2)

example_arange3 = torch.arange(1, 2.5, 0.5)

print(example_arange3)


a = torch.randn(4)

print(a)


exm_pow = torch.pow(a, 2)

print(exm_pow)


b = torch.arange(1., 5.)

print(b)


exm_pow2 = torch.pow(b, exm_pow)

print(exm_pow2)


# example

a = torch.randn(4)

a

torch.pow(a, 2)



exp = torch.arange(1., 5.)

a = torch.arange(1., 5.)

a

exp

torch.pow(a, exp)


class PositionalEncoding(nn.Module):

    def __init__(self, dim, seq_len_max):

        super(PositionalEncoding, self).__init__()

        PE = torch.zeros(seq_len_max, dim) # zeros : Returns a tensor filled with the scalar value 0, with the shape defined by the variable argument size.

        ######################### TO DO #########################

        # sinusoidal positional encoding 계산

        position = torch.arange(0, seq_len_max, dtype=torch.float32).unsqueeze(1)  # (seq_len_max, 1)

        div_term = torch.exp(torch.arange(0, dim, 2, dtype=torch.float32) * (-math.log(10000.0) / dim))



        # 짝수 인덱스: sin, 홀수 인덱스: cos

        PE[:, 0::2] = torch.sin(position * div_term)

        PE[:, 1::2] = torch.cos(position * div_term)

        ######################### TO DO #########################



        ######################### DO NOT CHANGE #########################

        # Positional Encoding is not learnable parameters.

        self.register_buffer('PE', PE.unsqueeze(0))

        ######################### DO NOT CHANGE #########################



    def forward(self, X):

        return X + self.PE[:, :X.size(1)]



# # vector x vector

tensor1 = torch.randn(3)

tensor2 = torch.randn(3)

torch.matmul(tensor1, tensor2).size()


# matrix x vector

tensor1 = torch.randn(3, 4)

tensor2 = torch.randn(4)

torch.matmul(tensor1, tensor2).size()


# batched matrix x broadcasted vector

tensor1 = torch.randn(10, 3, 4)

tensor2 = torch.randn(4)

torch.matmul(tensor1, tensor2).size()


# batched matrix x batched matrix

tensor1 = torch.randn(10, 3, 4)

tensor2 = torch.randn(10, 4, 5)

torch.matmul(tensor1, tensor2).size()


# batched matrix x broadcasted matrix

tensor1 = torch.randn(10, 3, 4)

tensor2 = torch.randn(4, 5)

torch.matmul(tensor1, tensor2).size()


import torch.nn as nn



class MultiHeadAttention(nn.Module):

    def __init__(self, dim, head_num):

        super(MultiHeadAttention, self).__init__()



        self.dim = dim

        self.head_num = head_num

        self.word_dim = dim // head_num



        ######################### TO DO #########################

        # Weight matrices for the linear transformation of Query, Key, and Value (dim x dim)

        self.W_q = nn.Linear(dim, dim)

        self.W_k = nn.Linear(dim, dim)

        self.W_v = nn.Linear(dim, dim)

        # Weight matrix for the linear transformation after combining the multi-head attention results

        self.W_o = nn.Linear(dim, dim)

        ######################### TO DO #########################



    def scaled_dot_product(self, Q, K, V, mask=None):

        ######################### TO DO #########################

        # Calculate the dot-product between Q and K, then scale by the square root of word_dim

        # Q, K, V : (batch, head_num, seq_len, word_dim)

        scores = torch.matmul(Q, K.transpose(-2, -1)) / math.sqrt(self.word_dim)



        # If a mask is provided, apply the mask (masked positions will not attend)

        if mask is not None:

            # mask == 0 인 위치를 매우 작은 값으로 채워서 softmax 후 거의 0이 되게 함

            scores = scores.masked_fill(mask == 0, -1e9)



        # Apply the softmax function to calculate the attention probabilities

        probs = torch.softmax(scores, dim=-1)



        # Multiply the attention probabilities by V to get the final attention values (heads)

        heads = torch.matmul(probs, V)

        ######################### TO DO #########################

        return heads



    def split(self, X):

        batch_num, seq_len, dim = X.size()

        return X.view(batch_num, seq_len, self.head_num, self.word_dim).transpose(1, 2)



    def combine(self, X):

        batch_num, _, seq_len, _ = X.size()

        return X.transpose(1, 2).contiguous().view(batch_num, seq_len, self.dim)



    def forward(self, X_Q, X_K, X_V, mask=None):

        Q = self.split(self.W_q(X_Q))

        K = self.split(self.W_k(X_K))

        V = self.split(self.W_v(X_V))



        heads = self.scaled_dot_product(Q, K, V, mask)

        output = self.W_o(self.combine(heads))

        return output



# NLP Example

batch, sentence_length, embedding_dim = 20, 5, 10

embedding = torch.randn(batch, sentence_length, embedding_dim)

layer_norm = nn.LayerNorm(embedding_dim)

nlp_output = layer_norm(embedding)

print("NLP Output:")

print(nlp_output)



# Image Example

N, C, H, W = 20, 5, 10, 10

input_tensor = torch.randn(N, C, H, W)

layer_norm_image = nn.LayerNorm([C, H, W])

output_image = layer_norm_image(input_tensor)

print("\nImage Output:")

print(output_image)


m = nn.Dropout(p=0.2)

input = torch.randn(20, 16)

output = m(input)

print(output)


class FFN(nn.Module):

    def __init__(self, dim, FFN_dim):

        super(FFN, self).__init__()

        self.FFN_layer = nn.Sequential(nn.Linear(dim, FFN_dim),

                                       nn.ReLU(),

                                       nn.Linear(FFN_dim, dim))

    def forward(self, X):

        return self.FFN_layer(X)



class EncoderLayer(nn.Module):

    def __init__(self, dim, head_num, FFN_dim, dropout):

        super(EncoderLayer, self).__init__()

        ######################### TO DO #########################

        # self-attention, FFN, layer norm, dropout 정의

        self.self_attn = MultiHeadAttention(dim, head_num)

        self.ffn = FFN(dim, FFN_dim)



        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)



        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        ######################### TO DO #########################



    def forward(self, X, mask):

        ######################### TO DO #########################

        # Multi-head self-attention 서브 레이어

        attn_output = self.self_attn(X, X, X, mask)

        X = self.norm1(X + self.dropout1(attn_output))  # residual + norm



        # Position-wise FFN 서브 레이어

        ffn_output = self.ffn(X)

        output = self.norm2(X + self.dropout2(ffn_output))  # residual + norm

        ######################### TO DO #########################

        return output



class DecoderLayer(nn.Module):

    def __init__(self, dim, head_num, FFN_dim, dropout):

        super(DecoderLayer, self).__init__()

        ######################### TO DO #########################

        # 1) masked self-attention (decoder용)

        # 2) encoder-decoder cross-attention

        # 3) FFN

        # 4) 각 sub-layer마다 LayerNorm + Dropout

        self.self_attn = MultiHeadAttention(dim, head_num)

        self.cross_attn = MultiHeadAttention(dim, head_num)

        self.ffn = FFN(dim, FFN_dim)



        self.norm1 = nn.LayerNorm(dim)

        self.norm2 = nn.LayerNorm(dim)

        self.norm3 = nn.LayerNorm(dim)



        self.dropout1 = nn.Dropout(dropout)

        self.dropout2 = nn.Dropout(dropout)

        self.dropout3 = nn.Dropout(dropout)

        ######################### TO DO #########################



    def forward(self, X, enc_output, cross_attn_mask, self_attn_mask):

        ######################### TO DO #########################

        # 주의: Transformer.forward에서

        # dec_layer(dec_output, enc_output, self_attn_mask, cross_attn_mask) 로 호출하므로

        # 여기의 self_attn_mask 인자는 디코더 자기자신용(masked) 마스크,

        # cross_attn_mask 인자는 encoder-decoder cross-attention용 마스크가 들어온다.



        # 1) Masked self-attention

        self_attn_output = self.self_attn(X, X, X, self_attn_mask)

        X = self.norm1(X + self.dropout1(self_attn_output))



        # 2) Encoder-Decoder cross-attention

        cross_attn_output = self.cross_attn(X, enc_output, enc_output, cross_attn_mask)

        X2 = self.norm2(X + self.dropout2(cross_attn_output))



        # 3) FFN

        ffn_output = self.ffn(X2)

        output = self.norm3(X2 + self.dropout3(ffn_output))

        ######################### TO DO #########################

        return output



class Transformer(nn.Module):

    def __init__(self, input_lib_size, output_lib_size, dim, head_num, layer_num, \

                 FFN_dim, seq_len_max, dropout):

        super(Transformer, self).__init__()

        self.enc_embeds = nn.Embedding(input_lib_size, dim)

        self.dec_embeds = nn.Embedding(output_lib_size, dim)

        self.pe = PositionalEncoding(dim, seq_len_max)



        self.encoder = nn.ModuleList([EncoderLayer(dim, head_num, FFN_dim, dropout) \

                                             for _ in range(layer_num)])

        self.decoder = nn.ModuleList([DecoderLayer(dim, head_num, FFN_dim, dropout) \

                                             for _ in range(layer_num)])

        self.Linear = nn.Linear(dim, output_lib_size)

        self.dropout = nn.Dropout(dropout)



    def generate_mask(self, src, tgt):

        self_attn_mask = (src != 0).unsqueeze(1).unsqueeze(2)

        cross_attn_mask = (tgt != 0).unsqueeze(1).unsqueeze(3)

        seq_length = tgt.size(1)

        nopeak_mask = (1 - torch.triu(torch.ones(1, seq_length, seq_length), diagonal=1)).bool()

        nopeak_mask = nopeak_mask.to(device)

        cross_attn_mask = cross_attn_mask & nopeak_mask

        return self_attn_mask, cross_attn_mask



    def forward(self, src, tgt):

        self_attn_mask, cross_attn_mask = self.generate_mask(src, tgt)

        src_embeds = self.dropout(self.pe(self.enc_embeds(src)))

        tgt_embeds = self.dropout(self.pe(self.dec_embeds(tgt)))



        enc_output = src_embeds

        for enc_layer in self.encoder:

            enc_output = enc_layer(enc_output, self_attn_mask)



        dec_output = tgt_embeds

        for dec_layer in self.decoder:

            dec_output = dec_layer(dec_output, enc_output, self_attn_mask, cross_attn_mask)



        output = self.Linear(dec_output)

        return output


input_lib_size = 5000

output_lib_size = 5000

dim = 512

head_num = 4

layer_num = 3

FFN_dim = 2048

seq_len_max = 100

dropout = 0.1



transformer = Transformer(input_lib_size, output_lib_size, dim, head_num, layer_num, \

                          FFN_dim, seq_len_max, dropout)

transformer = transformer.to(device)



# Generate random sample data

src_data = torch.randint(1, input_lib_size, (64, seq_len_max)).to(device)

tgt_data = torch.randint(1, output_lib_size, (64, seq_len_max)).to(device)


criterion = nn.CrossEntropyLoss(ignore_index=0)

optimizer = optim.Adam(transformer.parameters(), lr=0.0001, betas=(0.9, 0.98), eps=1e-9)



transformer.train()



for epoch in range(100):

    optimizer.zero_grad()

    output = transformer(src_data, tgt_data[:, :-1])

    loss = criterion(output.contiguous().view(-1, output_lib_size), tgt_data[:, 1:].contiguous().view(-1))

    loss.backward()

    optimizer.step()

    print(f"Epoch: {epoch+1}, Loss: {loss.item()}")


