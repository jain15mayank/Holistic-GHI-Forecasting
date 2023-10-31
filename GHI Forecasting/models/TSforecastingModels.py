import os
import sys
import math
sys.path.append('../datasets')
from TSforecastingDatasets import TimeSeriesDataset, PatchTSDataset

import torch
import torch.nn as nn
from torchinfo import summary

# Define the LSTM model
# Currently similar to IGARSS 2023 paper with hidden_size=32, num_layers=2, and output_size=n time steps
class LSTMModel(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(LSTMModel, self).__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        self.fc = nn.Sequential(
            nn.Linear(hidden_size, output_size), 
            nn.Tanhshrink(),
        )

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])  # Only use the last time step's output for forecasting
        return out

class TimeSeriesTransformerEnc(nn.Module):
    def __init__(self, input_size, d_model, nhead, num_layers, output_size, dim_feedforward_encoder=2048):
        super(TimeSeriesTransformerEnc, self).__init__()
        self.num_enc_layers = num_layers
        self.embedding = nn.Linear(input_size, d_model)
        self.transformerEnc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward_encoder, batch_first=True),
            num_layers=num_layers
        )
        self.fc = nn.Sequential(
            nn.Linear(d_model, output_size), 
            nn.Tanhshrink(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformerEnc(x)
        x = self.fc(x[:, -1, :])  # Only use the last time step's output for forecasting
        return x

class TimeSeriesTransformer(nn.Module):
    def __init__(self, input_size, output_size, d_model, nhead,
                 num_encoder_layers, num_decoder_layers, dim_feedforward=2048, dropout=0.1):
        super(TimeSeriesTransformer, self).__init__()
        self.embedding = nn.Linear(input_size, d_model)
        self.transformer = nn.Transformer(d_model, nhead, num_encoder_layers, num_decoder_layers, dim_feedforward, dropout)
        self.fc = nn.Sequential(
            nn.Linear(d_model, output_size), 
            nn.Tanhshrink(),
        )

    def forward(self, x):
        x = self.embedding(x)
        x = self.transformerEnc(x)
        x = self.fc(x[:, -1, :])  # Only use the last time step's output for forecasting
        return x

############################################################################################
## PatchTST - like implementation
############################################################################################
# sin-cos pos_encoding
def positional_encoding(q_len, d_model, normalize=True):
    pe = torch.zeros(q_len, d_model)
    position = torch.arange(0, q_len).unsqueeze(1)
    div_term = torch.exp(torch.arange(0, d_model, 2) * -(math.log(10000.0) / d_model))
    pe[:, 0::2] = torch.sin(position * div_term)
    pe[:, 1::2] = torch.cos(position * div_term)
    if normalize:
        pe = pe - pe.mean()
        pe = pe / (pe.std() * 10)
    return nn.Parameter(pe, requires_grad=True)

class Flatten_Head(nn.Module):
    def __init__(self, individual, n_vars, nf, target_window, head_dropout=0):
        super().__init__()
        
        self.individual = individual
        self.n_vars = n_vars
        
        if self.individual:
            self.linears = nn.ModuleList()
            self.dropouts = nn.ModuleList()
            self.flattens = nn.ModuleList()
            for i in range(self.n_vars):
                self.flattens.append(nn.Flatten(start_dim=-2))
                self.linears.append(nn.Linear(nf, target_window))
                self.dropouts.append(nn.Dropout(head_dropout))
        else:
            self.flatten = nn.Flatten(start_dim=-2)
            self.linear = nn.Linear(nf, target_window)
            self.dropout = nn.Dropout(head_dropout)
            
    def forward(self, x):                                 # x: [bs, nvars, d_model, patch_num]
        if self.individual:
            x_out = []
            for i in range(self.n_vars):
                z = self.flattens[i](x[:,i,:,:])          # z: [bs, d_model * patch_num]
                z = self.linears[i](z)                    # z: [bs, target_window]
                z = self.dropouts[i](z)
                x_out.append(z)
            x = torch.stack(x_out, dim=1)                 # x: [bs, nvars, target_window]
        else:
            x = self.flatten(x)
            x = self.linear(x)
            x = self.dropout(x)
        return x

class PatchTransformer(nn.Module):
    def __init__(self, in_features, context_window, d_model, nhead, num_layers, target_window, dim_feedforward_encoder=2048,
                 patch_len=16, stride=8, dropout=0.2, head_dropout=0):
        super(PatchTransformer, self).__init__()
        self.seq_len = context_window
        self.n_vars = in_features
        self.patch_len = patch_len
        self.stride = stride
        self.num_enc_layers = num_layers
        self.embedding = nn.Linear(patch_len, d_model)
        self.transformerEnc = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model, nhead, dim_feedforward=dim_feedforward_encoder, batch_first=True),
            num_layers=num_layers
        )
        # Residual dropout
        self.dropout = nn.Dropout(dropout)
        # Positional encoding
        patch_num = int((context_window - patch_len)/stride + 1)
        self.W_pos = positional_encoding(patch_num, d_model)
        # Flatten Head
        self.head = Flatten_Head(False, self.n_vars, d_model*patch_num, target_window, head_dropout=head_dropout)

    def forward(self, x):                                                                   # x: [bs, seq_len, nvars]
        # Perform Patching
        x = x.permute(0,2,1)                                                                # x: [bs, nvars, seq_len]
        x = x.unfold(dimension=-1, size=self.patch_len, step=self.stride)                   # x: [bs, nvars, patch_num, patch_len]
        # Input Encoding
        x = self.embedding(x)                                                               # x: [bs, nvars, patch_num, d_model]
        u = torch.reshape(x, (x.shape[0]*x.shape[1],x.shape[2],x.shape[3]))                 # u: [bs * nvars, patch_num, d_model]
        u = self.dropout(u + self.W_pos)                                                    # u: [bs * nvars, patch_num, d_model]
        # Encoder
        z = self.transformerEnc(u)                                                          # z: [bs * nvars, patch_num, d_model]
        z = torch.reshape(z, (-1, self.n_vars, z.shape[-2], z.shape[-1]))                   # z: [bs, nvars, patch_num, d_model]
        z = z.permute(0,1,3,2)                                                              # z: [bs, nvars, d_model, patch_num]
        # Flatten Head
        z = self.head(z)                                                                    # z: [bs, nvars, target_window]
        z = z.permute(0,2,1)                                                                # z: [bs, target_window, nvars]
        return z


############################################################################################
## PatchTST - like implementation
############################################################################################

if __name__=="__main__":
    dataPath = os.path.join(os.getcwd(), '..', 'datasets', 'data')
    startDate = "20200101"
    endDate = "20210807"
    contWindowsFilePath = os.path.join(dataPath, 'contWindows.pkl')
    # dataset = TimeSeriesDataset(dataPath, startDate, endDate, contWindowsFilePath=contWindowsFilePath)
    dataset = PatchTSDataset(dataPath, startDate, endDate, contWindowsFilePath=contWindowsFilePath)

    history_shape = dataset.__getitem__(0)[0].shape
    target_shape = dataset.__getitem__(0)[1].shape
    print(history_shape, target_shape)

    input_features = history_shape[1]
    output_steps = target_shape[0]

    if torch.cuda.is_available():
        device = torch.device('cuda', 0)
    else:
        device = torch.device('cpu')
    device = torch.device('cpu')
    # model = LSTMModel(input_features, hidden_size=32, num_layers=2, output_size = output_steps)
    # model = TimeSeriesTransformer(input_features, d_model=32, nhead=4, num_layers=2, output_size = output_steps)
    model = PatchTransformer(in_features=input_features, context_window=history_shape[0],
                             d_model=128, nhead=16, num_layers=3, target_window=output_steps)
    model.to(device)

    batch_size = 256
    summary(model, [(batch_size, history_shape[0], history_shape[1])], device=device)