import torch
from torch import nn
import torch.nn.functional as F


class CharacterConvolutions(nn.Module):

    def __init__(self, in_size, embed_dim):
        super(CharacterConvolutions, self).__init__()
        self.embedding = nn.Embedding(in_size, embed_dim)
        conv_layer_params = [[1, 32], [2, 32], [3, 64], [4, 128], [5, 256], [6, 512], [7, 1024]]
        self.conv_layers = nn.ModuleList([nn.Conv2d(embed_dim, out_dim, (1, ksize)) for ksize, out_dim in conv_layer_params])
        self.relu = nn.ReLU()

    # x has shape [batch_size, seq_len, characters]
    def forward(self, x):
        x = self.embedding(x) # [batch_size, seq_len, characters, char_embed_dim]
        x = x.permute(0, 3, 1, 2) # [batch_size, char_embed_dim, seq_len, characters]
        convs = [c(x) for c in self.conv_layers]
        pools = [self.relu(F.max_pool2d(c, kernel_size=(1, c.shape[-1]))).squeeze(-1).permute(0, 2, 1) for c in convs]
        return torch.cat(pools, dim=-1)
    

class ELMoLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, dropout):
        super(ELMoLSTM, self).__init__()
        self.in_proj = nn.Linear(input_size, hidden_size)
        self.lstm1 = nn.LSTM(hidden_size, hidden_size, batch_first=True, bidirectional=True)
        self.middle_proj = nn.Linear(hidden_size * 2, input_size)
        self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional=True)
        self.out_proj = nn.Linear(hidden_size * 2, input_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.in_proj(x)
        layer1 = self.middle_proj(self.lstm1(x)[0] + x.repeat(1, 1, 2))
        layer2 = self.out_proj(self.lstm2(self.dropout(layer1))[0])
        return layer1, layer2
    

class ELMo(nn.Module):

    def __init__(self, input_size, char_embed_dim=16, output_size=128, hidden_size=1024, dropout=0.5):
        super(ELMo, self).__init__()
        self.character_convolutions = CharacterConvolutions(input_size, char_embed_dim)
        self.highway = nn.Linear(2048, 2048)
        self.highway_gate = nn.Linear(2048, 2048)
        self.in_proj = nn.Linear(2048, output_size)
        self.elmo_lstm = ELMoLSTM(output_size, hidden_size=hidden_size, dropout=dropout)

    def forward(self, x):
        x = self.character_convolutions(x)
        h = self.highway(x)
        h_gate = F.sigmoid(self.highway_gate(x))
        layer1 = self.in_proj((h * h_gate) + (x * (1 - h_gate)))
        layer2, layer3 = self.elmo_lstm(layer1)
        return torch.cat([x.unsqueeze(-1) for x in [layer1, layer2, layer3]], dim=-1)
    

class ELMoPretrainModel(nn.Module):

    def __init__(self, num_chars, output_size, num_words, dropout=0.5):
        super(ELMoPretrainModel, self).__init__()
        self.elmo = ELMo(num_chars, output_size=output_size)
        self.dropout = nn.Dropout(p=dropout)
        self.out_proj = nn.Linear(output_size, num_words)

    def forward(self, x):
        _, _, x = self.elmo(x)
        return self.out_proj(self.dropout(x))
    

class ELMoFinetuneModel(nn.Module):

    def __init__(self, input_size, pretrained_model_path, device):
        super(ELMoFinetuneModel, self).__init__()
        self.elmo = ELMo(input_size)
        self.elmo.load_state_dict(torch.load(pretrained_model_path, map_location=device))
        self.elmo = self.elmo.eval()
        for p in self.elmo.parameters():
            p.required_grad = False
        self.summation_weights = nn.Parameter(torch.randn(3))
        self.scaling_term = nn.Parameter(torch.randn(1))

    def forward(self, x):
        elmo_embeddings = self.elmo(x)
        summed_embeddings = (elmo_embeddings * self.summation_weights.softmax(dim=0).view(1, 1, 1, 3)).sum(dim=-1)
        return summed_embeddings * self.scaling_term