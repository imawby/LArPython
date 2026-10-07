import torch
import torch.nn as nn
import torch.nn.functional as F

########################################################################################################

class ViewScaler(nn.Module):
    def __init__(self):
        super().__init__()
        self.scale = nn.Parameter(torch.ones(1))
        self.shift = nn.Parameter(torch.zeros(1))

    def forward(self, x):
        return x * self.scale + self.shift

########################################################################################################    

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, filters, kernel_size=3):
        super().__init__()
        pad = kernel_size // 2
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size, padding=pad, bias=False)
        self.bn1 = nn.BatchNorm2d(filters)
        self.conv2 = nn.Conv2d(filters, filters, kernel_size, padding=pad, bias=False)
        self.bn2 = nn.BatchNorm2d(filters)
        self.dropout = nn.Dropout2d(0.1)
        
        # Match channels if needed
        self.project = None
        if in_channels != filters:
            self.project = nn.Sequential(
                nn.Conv2d(in_channels, filters, 1, bias=False),
                nn.BatchNorm2d(filters),
            )

    def forward(self, x):
        shortcut = x
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))

        if self.project is not None:
            shortcut = self.project(shortcut)

        out = out + shortcut
        out = F.relu(out)
        out = self.dropout(out)
        
        return out

########################################################################################################
    
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=3):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out = torch.amax(x, dim=1, keepdim=True)
        combined = torch.cat([avg_out, max_out], dim=1)
        attention_map = self.sigmoid(self.conv(combined))

        return attention_map

########################################################################################################
    
class SharedEncoder(nn.Module):
    def __init__(self):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(2, 32, 3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
        )

        self.block1 = nn.Sequential(
            ResidualBlock(32, 32),
        )
        self.pool1 = nn.MaxPool2d(2)

        self.block2 = nn.Sequential(
            ResidualBlock(32, 64),
        )

        self.block3 = nn.Sequential(
            ResidualBlock(64, 128),
        )

        self.spatial_attention = SpatialAttention()
        
    def forward(self, x):
        x = self.stem(x)
        x = self.block1(x)
        x = self.pool1(x)
        x = self.block2(x)
        x = self.block3(x)

        attention_weights = self.spatial_attention(x)
        x = x * (1 + attention_weights)        
        
        gap = x.mean(dim=(2,3))
        gmp = x.amax(dim=(2,3))
        std = x.std((2,3))
        x = torch.cat([gap, gmp, std], dim=1)
        
        return x

########################################################################################################
    
class IvysaurusModel(nn.Module):
    def __init__(self, dimensions, nclasses, nTrackVars, nShowerVars):
        super().__init__()

        self.encoder = SharedEncoder()  # shared across all views & start/end

        # One scaler per view
        self.scalerU = ViewScaler()
        self.scalerV = ViewScaler()
        self.scalerW = ViewScaler()

        # Each branch: three (gap, gmp, std) 128 output for start and end grid in each of three views
        encoder_out = 384 * 2 * 3
        combined_feat = encoder_out + nTrackVars + nShowerVars

        self.head = nn.Sequential(
            nn.Linear(combined_feat, 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(256, 128, bias=False),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3))
        self.out = nn.Linear(128, nclasses)

    def _branch(self, start_scaled, start_mask, end_scaled, end_mask):
        start_combined = torch.cat([start_scaled, start_mask], dim=1)  # (N, 2, H, W)
        end_combined   = torch.cat([end_scaled, end_mask], dim=1)      # (N, 2, H, W)
    
        start_feat = self.encoder(start_combined)  # (N, 128)
        end_feat   = self.encoder(end_combined)    # (N, 128)
    
        return torch.cat([start_feat, end_feat], dim=1)  # (N, 256)
    
    def forward(self,
                startU, startU_mask, endU, endU_mask,
                startV, startV_mask, endV, endV_mask,
                startW, startW_mask, endW, endW_mask,
                trackVars, showerVars):

        startU = startU.permute(0, 3, 1, 2)
        startV = startV.permute(0, 3, 1, 2)
        startW = startW.permute(0, 3, 1, 2)
        endU = endU.permute(0, 3, 1, 2)
        endV = endV.permute(0, 3, 1, 2)
        endW = endW.permute(0, 3, 1, 2)        
        startU_mask = startU_mask.permute(0, 3, 1, 2)
        startV_mask = startV_mask.permute(0, 3, 1, 2)
        startW_mask = startW_mask.permute(0, 3, 1, 2)
        endU_mask = endU_mask.permute(0, 3, 1, 2)
        endV_mask = endV_mask.permute(0, 3, 1, 2)
        endW_mask = endW_mask.permute(0, 3, 1, 2)        

        startU = self.scalerU(startU)
        endU = self.scalerU(endU)
        startV = self.scalerV(startV)
        endV = self.scalerV(endV)
        startW = self.scalerW(startW)
        endW = self.scalerW(endW)        

        branchU = self._branch(startU, startU_mask, endU, endU_mask)
        branchV = self._branch(startV, startV_mask, endV, endV_mask)
        branchW = self._branch(startW, startW_mask, endW, endW_mask)

        combined = torch.cat([branchU, branchV, branchW,
                              trackVars, showerVars], dim=1)

        combined = self.head(combined)
        logits = self.out(combined)
        return logits
    
