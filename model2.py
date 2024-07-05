import torch.nn as nn
import torch.nn.functional as F
import torch

class RSM1D(nn.Module):
    def __init__(self, channels_in=None, channels_out=None):
        super().__init__()
        self.channels_in = channels_in
        self.channels_out = channels_out

        self.conv1 = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        self.conv3 = nn.Conv1d(in_channels=channels_out, out_channels=channels_out, bias=False, kernel_size=3, padding=1)
        
        self.bn1 = nn.BatchNorm1d(channels_out)
        self.bn2 = nn.BatchNorm1d(channels_out)
        self.bn3 = nn.BatchNorm1d(channels_out)

        self.nin = nn.Conv1d(in_channels=channels_in, out_channels=channels_out, bias=False, kernel_size=1)

    def forward(self, xx):
        yy = F.leaky_relu(self.bn1(self.conv1(xx)))
        yy = F.leaky_relu(self.bn2(self.conv2(yy)))
        yy = self.conv3(yy)
        xx = self.nin(xx)

        xx = self.bn3(xx + yy)
        xx = F.leaky_relu(xx)
        return xx

class SSDNet1D(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=16, kernel_size=7, padding=3, bias=False)
        self.bn1 = nn.BatchNorm1d(16)

        self.RSM1 = RSM1D(channels_in=16, channels_out=32)
        self.RSM2 = RSM1D(channels_in=32, channels_out=64)
        self.RSM3 = RSM1D(channels_in=64, channels_out=128)
        self.RSM4 = RSM1D(channels_in=128, channels_out=256)  # Genişletildi

        self.lstm = nn.LSTM(input_size=256, hidden_size=128, batch_first=True, bidirectional=True)  # LSTM eklendi

        self.fc1 = nn.Linear(in_features=256, out_features=128)  # Genişletildi
        self.fc2 = nn.Linear(in_features=128, out_features=64)
        self.out = nn.Linear(in_features=64, out_features=2)

        self.dropout = nn.Dropout(p=0.5)  # Dropout eklendi

    def forward(self, x):
        x = F.leaky_relu(self.bn1(self.conv1(x)))
        x = F.max_pool1d(x, kernel_size=4)

        # stacked ResNet-Style Modules
        x = self.RSM1(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM2(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM3(x)
        x = F.max_pool1d(x, kernel_size=4)
        x = self.RSM4(x)
        x = F.max_pool1d(x, kernel_size=375)

        x = torch.flatten(x, start_dim=1)
        x, _ = self.lstm(x.unsqueeze(-1))  # LSTM katmanı
        x = x[:, -1, :]  # Son hidden state
        x = F.leaky_relu(self.fc1(x))
        x = self.dropout(x)  # Dropout eklendi
        x = F.leaky_relu(self.fc2(x))
        x = self.out(x)
        return x


if __name__ == '__main__':
    Res_TSSDNet = SSDNet1D()
    num_params_1D = sum(i.numel() for i in Res_TSSDNet.parameters() if i.requires_grad)
    x1 = torch.randn(2, 1, 96000)
    y1 = Res_TSSDNet(x1)

    print('End of Program.')
