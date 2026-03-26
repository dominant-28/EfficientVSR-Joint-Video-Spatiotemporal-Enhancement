import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops import deform_conv2d


class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super().__init__()
        self.conv = nn.Sequential(
            nn.BatchNorm2d(in_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False)
        )

    def forward(self,x):
        out =self.conv(x)
        return torch.cat([x, out], dim=1) 


class DenseBlock(nn.Module):
          def __init__(self, in_channels, growth_rate=32, num_layers=4):
                super().__init__()
                layers = []
                ch = in_channels
                for _ in range(num_layers):
                     layers.append(DenseLayer(ch, growth_rate))
                     ch+= growth_rate
                self.block =nn.Sequential(*layers)
                self.out_channels =ch

                self.compress = nn.Conv2d(ch,128, kernel_size=1,bias=False) 
          def forward(self,x):
                out =self.block(x)
                return self.compress(out)      


class GatedFFN(nn.Module):
      def __init__(self,channels):
            super().__init__()
            self.conv_value = nn.Conv2d(channels, channels, kernel_size=3, padding=1,bias=False)
            self.conv_gate = nn.Conv2d(channels, channels, kernel_size=3,padding=1, bias=False)
            self.bn = nn.BatchNorm2d(channels)

      def forward(self, x):
            value =self.conv_value(x)
            gate = torch.sigmoid(self.conv_gate(x))
            return self.bn(value*gate) + x
      

class SEBlock(nn.Module):
      def __init__(self, channels, reduction=16):
            super().__init__()
            self.pool = nn.AdaptiveAvgPool2d(1)
            self.fc =nn.Sequential(
                  nn.Linear(channels,channels//reduction, bias =False),
                  nn. ReLU(inplace= True),
                  nn.Linear(channels//reduction, channels,bias=False),
                  nn.Sigmoid()
            )   

      def forward(self,x):
            b,c,_,_=x.shape
            w = self.pool(x).view(b,c)
            w = self.fc(w).view(b,c,1,1)
            return x*w         

class DeformableAlignment(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.offset_conv = nn.Conv2d(
            channels * 2,  
            27,                 
            kernel_size=3,
            padding=1,
            bias=True
        )
        self.main_conv = nn.Conv2d(channels, channels, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(channels)

    def forward(self, f1_feat, f3_feat):
        
        concat = torch.cat([f1_feat, f3_feat], dim=1)
        offset_mask = self.offset_conv(concat)

        offset = offset_mask[:, :18, :, :]       
        mask   = torch.sigmoid(offset_mask[:, 18:, :, :]) 

        weight = self.main_conv.weight
        aligned = deform_conv2d(
            f1_feat, offset, weight,
            mask=mask,
            padding=1
        )

        return self.bn(aligned) 

class RoPE2D(nn.Module):
   
    def __init__(self, dim):
        super().__init__()
        assert dim % 4 == 0, "dim must be divisible by 4 for 2D RoPE"
        self.dim = dim

    def _get_freq(self, seq_len, device):
        i = torch.arange(0, self.dim // 2, device=device).float()
        freq = 1.0 / (10000 ** (2 * i / self.dim))
        t = torch.arange(seq_len, device=device).float()
        freqs = torch.outer(t, freq)   
        return torch.cat([freqs, freqs], dim=-1) 

    def _rotate(self, x, freqs):
        half = x.shape[-1] // 2
        x1, x2 = x[..., :half], x[..., half:]
        cos = freqs[..., :half].cos()
        sin = freqs[..., :half].sin()
        return torch.cat([x1 * cos - x2 * sin,
                          x1 * sin + x2 * cos], dim=-1)

    def forward(self, q, k, H, W):
      
        device = q.device
        freq_h = self._get_freq(H, device)   
        freq_w = self._get_freq(W, device)  


        freq_h = freq_h.unsqueeze(1).expand(-1, W, -1)   
        freq_w = freq_w.unsqueeze(0).expand(H, -1, -1)   
        freqs  = (freq_h + freq_w).reshape(H * W, -1)    

        head_dim = q.shape[-1]
        freqs = freqs[..., :head_dim]

        q_rot = self._rotate(q, freqs.unsqueeze(0).unsqueeze(0))
        k_rot = self._rotate(k, freqs.unsqueeze(0).unsqueeze(0))

        return q_rot, k_rot


class CrossFrameAttention(nn.Module):
    def __init__(self, channels, num_heads=4, window_size=8):
        super().__init__()
        self.channels    = channels
        self.num_heads   = num_heads
        self.window_size = window_size
        self.head_dim    = channels // num_heads
        self.scale       = self.head_dim ** -0.5

        self.q_proj = nn.Linear(channels, channels, bias=False)
        self.k_proj = nn.Linear(channels, channels, bias=False)
        self.v_proj = nn.Linear(channels, channels, bias=False)
        self.out_proj = nn.Linear(channels, channels, bias=False)

        self.rope = RoPE2D(self.head_dim)
        self.norm = nn.LayerNorm(channels)

    def _window_partition(self, x, window_size):
        
        B, C, H, W = x.shape
        x = x.permute(0, 2, 3, 1) 
        x = x.view(B,
                   H // window_size, window_size,
                   W // window_size, window_size, C)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous()
        x = x.view(-1, window_size * window_size, C)
        return x

    def _window_reverse(self, x, window_size, H, W, B):
        C = x.shape[-1]
        x = x.view(B,
                   H // window_size,
                   W // window_size,
                   window_size, window_size, C)
        x = x.permute(0, 5, 1, 3, 2, 4).contiguous()
        x = x.view(B, C, H, W)
        return x

    def forward(self, temporal_feat, spatial_feat):
        B, C, H, W = temporal_feat.shape
        ws = self.window_size
        pad_h = (ws - H % ws) % ws
        pad_w = (ws - W % ws) % ws
        if pad_h > 0 or pad_w > 0:
            temporal_feat = F.pad(temporal_feat, (0, pad_w, 0, pad_h))
            spatial_feat  = F.pad(spatial_feat,  (0, pad_w, 0, pad_h))

        _, _, Hp, Wp = temporal_feat.shape

        q_win = self._window_partition(temporal_feat, ws) 
        k_win = self._window_partition(spatial_feat,  ws)
        v_win = self._window_partition(spatial_feat,  ws)

        Q = self.q_proj(q_win)
        K = self.k_proj(k_win)
        V = self.v_proj(v_win)

        BW = Q.shape[0]
        N  = ws * ws
        Q = Q.view(BW, N, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(BW, N, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(BW, N, self.num_heads, self.head_dim).transpose(1, 2)

        Q, K = self.rope(Q, K, ws, ws)

        attn = (Q @ K.transpose(-2, -1)) * self.scale
        attn = F.softmax(attn, dim=-1)

        out = (attn @ V)  
        out = out.transpose(1, 2).contiguous().view(BW, N, C)
        out = self.out_proj(out)

        out = self._window_reverse(out, ws, Hp, Wp, B)

        if pad_h > 0 or pad_w > 0:
            out = out[:, :, :H, :W]

        return self.norm(out.permute(0, 2, 3, 1)).permute(0, 3, 1, 2) + temporal_feat

class CBAM(nn.Module):
    def __init__(self, channels, reduction=16):
        super().__init__()
        
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.channel_fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False)
        )
        self.spatial_conv = nn.Conv2d(2, 1, kernel_size=7, padding=3, bias=False)

    def forward(self, x):
        b, c, _, _ = x.shape

        avg = self.channel_fc(self.avg_pool(x).view(b, c))
        mx  = self.channel_fc(self.max_pool(x).view(b, c))
        ch_attn = torch.sigmoid(avg + mx).view(b, c, 1, 1)
        x = x * ch_attn

        avg_sp = x.mean(dim=1, keepdim=True)
        max_sp = x.max(dim=1, keepdim=True)[0]
        sp_attn = torch.sigmoid(self.spatial_conv(
            torch.cat([avg_sp, max_sp], dim=1)
        ))
        x = x * sp_attn

        return x


class ResBlock(nn.Module):
    def __init__(self, channels):
        super().__init__()
        self.block = nn.Sequential(
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 3, padding=1, bias=False),
            nn.BatchNorm2d(channels)
        )

    def forward(self, x):
        return x + self.block(x)

class EfficientVSR(nn.Module):

    def __init__(self, base_channels=128):
        super().__init__()

        self.input_conv = nn.Conv2d(3, 32, kernel_size=3, padding=1, bias=False)
        self.encoder = DenseBlock(in_channels=32, growth_rate=32, num_layers=4)

        self.deform_align  = DeformableAlignment(base_channels)
        self.temporal_res1 = ResBlock(base_channels)
        self.temporal_res2 = ResBlock(base_channels)
        self.temporal_res3 = ResBlock(base_channels)
        self.temporal_gffn = GatedFFN(base_channels)

        self.se_attn      = SEBlock(base_channels)
        self.spatial_res1 = ResBlock(base_channels)
        self.spatial_res2 = ResBlock(base_channels)
        self.spatial_res3 = ResBlock(base_channels)
        self.spatial_gffn = GatedFFN(base_channels)

       
        self.cross_attn = CrossFrameAttention(
            channels=base_channels,
            num_heads=4,
            window_size=8
        )

        self.fuse_conv = nn.Conv2d(base_channels * 2, base_channels, 1, bias=False)
        self.cbam = CBAM(base_channels)


        self.upsample = nn.Sequential(
            nn.Conv2d(base_channels, 48, kernel_size=3, padding=1, bias=False),
            nn.PixelShuffle(2),    
            nn.Conv2d(12, 3, kernel_size=3, padding=1, bias=False)
        )

    def _encode(self, frame):
        x = self.input_conv(frame)
        x = self.encoder(x)
        return x  

    def forward(self, lr1, lr3):
     
        feat1 = self._encode(lr1)  
        feat3 = self._encode(lr3)   

        temp_feat = self.deform_align(feat1, feat3)
        temp_feat = self.temporal_res1(temp_feat)
        temp_feat = self.temporal_res2(temp_feat)
        temp_feat = self.temporal_res3(temp_feat)
        temp_feat = self.temporal_gffn(temp_feat)   
     
        spat_input = (feat1 + feat3) / 2.0
        spat_feat  = self.se_attn(spat_input)
        spat_feat  = self.spatial_res1(spat_feat)
        spat_feat  = self.spatial_res2(spat_feat)
        spat_feat  = self.spatial_res3(spat_feat)
        spat_feat  = self.spatial_gffn(spat_feat)   

        fused = self.cross_attn(temp_feat, spat_feat)   

        combined = torch.cat([fused, spat_feat], dim=1) 
        combined = self.fuse_conv(combined)               
        combined = self.cbam(combined)                    

        out = self.upsample(combined)  

        return torch.clamp(out, 0.0, 1.0)


if __name__ == '__main__':

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    model = EfficientVSR(base_channels=128).to(device)

    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTotal parameters     : {total_params:,}")
    print(f"Trainable parameters : {trainable_params:,}")
    print(f"Model size (approx)  : {total_params * 4 / 1024 / 1024:.2f} MB")

    lr1 = torch.randn(2, 3, 32, 32).to(device)   
    lr3 = torch.randn(2, 3, 32, 32).to(device)

    with torch.no_grad():
        out = model(lr1, lr3)

    print(f"\ Forward pass successful!")
    print(f"   Input  shape : {lr1.shape}")   
    print(f"   Output shape : {out.shape}")  
    print(f"   Output range : [{out.min():.3f}, {out.max():.3f}]")