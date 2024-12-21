# model.py

import torch
import torch.nn as nn

class UNet(nn.Module):
    def __init__(self, in_channels=33, out_channels=1, dropout_prob=0.3):
        super(UNet, self).__init__()
        self.dropout_prob = dropout_prob

        # Encoder
        self.enc1 = self.contract_block(in_channels, 64, 7, 3)
        self.enc2 = self.contract_block(64, 128, 3, 1)
        self.enc3 = self.contract_block(128, 256, 3, 1)

        # Decoder
        self.dec1 = self.expand_block(256, 128, 3, 1)
        self.dec2 = self.expand_block(128, 64, 3, 1)
        self.dec3 = nn.ConvTranspose2d(64, out_channels, kernel_size=3, padding=1)


        # Dropout
        self.dropout = nn.Dropout(self.dropout_prob)

    def forward(self, x):
        # Encoder
        enc1 = self.enc1(x)
        enc2 = self.enc2(enc1)
        bottleneck = self.enc3(enc2)

        # Apply dropout to bottleneck
        bottleneck = self.dropout(bottleneck)

        # Decoder with skip connections
        dec1 = self.dec1(bottleneck)
        dec2 = self.dec2(dec1 + enc2)
        dec3 = self.dec3(dec2 + enc1)

        return dec3

    def contract_block(self, in_channels, out_channels, kernel_size, padding):
        contract = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.dropout_prob),  # Apply dropout
            nn.Conv2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return contract

    def expand_block(self, in_channels, out_channels, kernel_size, padding):
        expand = nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
            nn.Dropout(self.dropout_prob),  # Apply dropout
            nn.ConvTranspose2d(out_channels, out_channels, kernel_size=kernel_size, padding=padding),
            nn.ReLU(),
            nn.BatchNorm2d(out_channels),
        )
        return expand
