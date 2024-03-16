import math
import torch
from torch import nn, Tensor
from torch.nn import TransformerEncoder, TransformerEncoderLayer

torch.seed(0)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout)

        position = torch.arange(max_len).unsqueeze(1)
        div_term = torch.exp(
            torch.arange(0, d_model, 2) * (-math.log(10000.0) / d_model)
        )
        pe = torch.zeros(max_len, 1, d_model)
        pe[:, 0, 0::2] = torch.sin(position * div_term)
        pe[:, 0, 1::2] = torch.cos(position * div_term)
        self.register_buffer("pe", pe)

    def forward(self, x):
        """
        Arguments:
            x: Tensor, shape ``[seq_len, batch_size, embedding_dim]``
        """
        x = x + self.pe[: x.size(0)]
        return self.dropout(x)


# Remember to try causal attention
class TransformerModel(nn.Module):
    def __init__(
        self, d_model: int, nhead: int, d_hid: int, nlayers: int, dropout: float = 0.1
    ):
        """
        ntoken: The size of the vocabulary (total number of unique tokens).
        d_model: The dimensionality of the token embeddings (the size of the vectors that represent each token).
        nhead: The number of attention heads in the multi-head attention mechanisms.
        d_hid: The dimensionality of the feedforward network model in the transformer encoder.
        nlayers: The number of sub-encoder-layers in the transformer encoder.
        dropout: The dropout rate, a regularization technique to prevent overfitting.


        """
        super().__init__()
        self.model_type = "Transformer"
        self.pos_encoder = PositionalEncoding(d_model, dropout)
        encoder_layers = TransformerEncoderLayer(
            d_model, nhead, d_hid, dropout, batch_first=True
        )
        self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
        ##Use linear layer instead of traidiontal embedding layer due to continous data
        self.input_linear = nn.Linear(3, d_model)
        self.d_model = d_model
        self.output_linear = nn.Linear(d_model, 3)

        self.init_weights()

    def init_weights(self) -> None:
        initrange = 0.1
        self.input_linear.weight.data.uniform_(-initrange, initrange)
        self.input_linear.bias.data.zero_()
        self.output_linear.bias.data.zero_()
        self.output_linear.weight.data.uniform_(-initrange, initrange)

    def generate_square_subsequent_mask(self, size: int) -> Tensor:
        """
        This mask is required for causal attention,
        it creates an upper triangular matrix of -inf at to and 
        0s at bottom that hides future states from the current state
        """
        mask = (torch.triu(torch.ones(size, size)) == 1).transpose(0, 1)
        mask = (
            mask.float()
            .masked_fill(mask == 0, float("-inf"))
            .masked_fill(mask == 1, float(0.0))
        )
        return mask

    def forward(self, input: Tensor) -> Tensor:
        # input: Tensor, shape ``[batch_size, seq_len, feature_size]``
        input = self.input_linear(input) * math.sqrt(self.d_model)
        input = self.pos_encoder(input)

        if self.training:
            src_mask = self.generate_square_subsequent_mask(input.size(1)).to(
                input.device
            )
            output = self.transformer_encoder(input, mask=src_mask)
        else:
            output = self.transformer_encoder(input)

        # output shape ``[batch_size, feature_size]``, to match with label
        output = self.output_linear(output[:, -1, :])
        return output
