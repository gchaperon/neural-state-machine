import torch
import torch.nn as nn
from torch.nn.utils.rnn import PackedSequence, pad_packed_sequence


class NormalizeWordsModel(nn.Module):
    def __init__(self, vocab):
        super(NormalizeWordsModel, self).__init__()

        self.vocab = vocab
        self.default_embed = torch.rand(300)
        self.W = nn.Parameter(torch.eye(300))

    def forward(self, input):
        # input should be PackedSequence
        words, *rest = input
        C = torch.cat((self.vocab.T, self.default_embed.view(-1, 1)), dim=1)
        P = torch.softmax(words @ self.W @ C, dim=1)
        V = P[:, -1:] * words + P[:, :-1] @ self.vocab
        return PackedSequence(V, *rest)


class InstructionsDecoder(nn.Module):
    def __init__(self, n_instructions, *args, **kwargs):
        super(InstructionsDecoder, self).__init__()
        self.n_instructions = n_instructions
        self.rnn = nn.RNNCell(*args, **kwargs)

    def forward(self, input):
        hx = None
        hiddens = []
        for _ in range(self.n_instructions):
            hx = self.rnn(input, hx)
            hiddens.append(hx)
        return torch.cat([t.unsqueeze(1) for t in hiddens], dim=1)


class InstructionsModel(nn.Module):
    def __init__(self, vocab, n_instructions):
        super(InstructionsModel, self).__init__()
        self.n_instructions = n_instructions

        self.tagger = NormalizeWordsModel(vocab)
        self.encoder = nn.LSTM(input_size=300, hidden_size=300, dropout=0.0)
        self.decoder = InstructionsDecoder(
            n_instructions, input_size=300, hidden_size=300
        )

    def forward(self, input):
        # input should be a PackedSequence
        V = self.tagger(input)
        _, (q, _) = self.encoder(V)
        H = self.decoder(q.squeeze())
        # Unpack sequences
        V, lens_unpacked = pad_packed_sequence(V, batch_first=True)
        # Prepare mask for attention
        seq_len = V.size(1)

        mask = (
            torch.cat(
                [
                    torch.ones(seq_len - l, dtype=torch.long) * i
                    for i, l in enumerate(lens_unpacked)
                ]
            ),
            torch.arange(self.n_instructions)[:, None],
            torch.cat([torch.arange(l, seq_len) for l in lens_unpacked]),
        )
        tmp = H @ V.transpose(1, 2)
        tmp[mask] = float("-inf")
        R = torch.softmax(tmp, dim=-1) @ V
        return R
