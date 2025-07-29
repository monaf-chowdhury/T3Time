import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from layers.StandardNorm import Normalize
from layers.Cross_Modal_Align import CrossModal


class AdaptiveDynamicHeadsCMA(nn.Module):
    """
    Small network to compute per-head weights from the concatenated heads
    Input shape: [B, N, H*C]      Output shape: [B, N, H]
    """
    def __init__(self, num_heads, num_nodes, channel, device):
        super().__init__()
        self.num_heads = num_heads
        self.num_nodes = num_nodes
        self.channel = channel
        self.device = device
        
        self.gate_mlp = nn.Sequential(
            nn.Linear(num_heads * channel, 128),
            nn.LayerNorm(128),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(128, num_heads)
        ).to(device)

    def forward(self, cma_outputs):

        B, C, N = cma_outputs[0].shape
        H = self.num_heads

        combined = torch.cat(cma_outputs, dim=1)                        # [B, H*C, N]
        combined_permute = combined.permute(0, 2, 1)                    # [B, N, H*C]

        gates = self.gate_mlp(combined_permute)                         # raw scores: [B, N, H]
        gates = F.softmax(gates, dim=-1)                                # [B, N, H]

        combined_heads = combined.view(B, H, C, N).permute(0, 1, 3, 2)  # [B, H, N, C]
        gates = gates.permute(0, 2, 1).unsqueeze(-1)                    # [B, H, N, 1]
        
        weighted_heads = combined_heads * gates                         # [B, H, C, N] * [B, H, N, 1] → broadcasting
        weighted_heads = weighted_heads.permute(0, 1, 3, 2)             # back to [B, H, C, N]
        
        fused = weighted_heads.sum(dim=1)                               # [B, C, N]

        return fused

class RichHorizonGate(nn.Module):
    """
    Each channel has its own gate that depends both 
    on the global context (pooled) and on the forecast horizon.
    """
    def __init__(self, embed_dim):

        super().__init__()
        self.gate_mlp = nn.Sequential(
            nn.Linear(embed_dim + 1, 64),
            nn.ReLU(),
            nn.Linear(64, embed_dim),
            nn.Sigmoid()
        )

    def forward(self, embedding: torch.Tensor, horizon: int) -> torch.Tensor:

        B, C, N = embedding.size()
        pooled_embed = embedding.mean(dim=2)                                # [B, C]
        horizon_tensor = torch.full((B, 1), float(horizon) / 1000.0, device=embedding.device)

        gating_input = torch.cat([pooled_embed, horizon_tensor], dim=1)     # [B, C+1]
        gate = self.gate_mlp(gating_input).unsqueeze(-1)                    # [B, C, 1]
        return gate
    
class FrequencyAttentionPooling(nn.Module):
    """
    Learnable, attention-weighted pooling over frequency bins
    """
    def __init__(self, embed_dim):
        super().__init__()
        self.freq_attention = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Linear(embed_dim // 2, 1)
        )

    def forward(self, freq_enc_out):

        attn_logits  = self.freq_attention(freq_enc_out)           # [B*N, Lf, 1]
        attn_weights = F.softmax(attn_logits, dim=1)               # normalize over Lf

        pooled_freq  = (freq_enc_out * attn_weights).sum(dim=1)    # [B*N, C]
        return pooled_freq


class TriModal(nn.Module):
    def __init__(
        self,
        device = "cuda:7",
        channel = 32,
        num_nodes = 7,
        seq_len = 96,
        pred_len = 96,
        dropout_n = 0.1,
        d_llm = 768,
        e_layer = 1,
        d_layer = 1,
        d_ff=32,
        head =8
    ):
        super().__init__()

        self.device = device
        self.channel = channel
        self.num_nodes = num_nodes
        self.seq_len = seq_len
        self.pred_len = pred_len
        self.dropout_n= dropout_n
        self.d_llm = d_llm
        self.e_layer = e_layer
        self.d_layer = d_layer
        self.d_ff = d_ff
        self.head = head

        self.normalize_layers = Normalize(self.num_nodes, affine=False).to(self.device)
        self.length_to_feature = nn.Linear(self.seq_len, self.channel).to(self.device)
        self.num_cma_heads = 4 

        # Time Series Encoder
        self.ts_encoder_layer = nn.TransformerEncoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, 
                                                           norm_first = True,dropout = self.dropout_n).to(self.device)
        self.ts_encoder = nn.TransformerEncoder(self.ts_encoder_layer, num_layers = self.e_layer).to(self.device)

        # Prompt Encoder
        self.prompt_encoder_layer = nn.TransformerEncoderLayer(d_model = self.d_llm, nhead = self.head, batch_first=True, 
                                                            norm_first = True,dropout = self.dropout_n).to(self.device)
        self.prompt_encoder = nn.TransformerEncoder(self.prompt_encoder_layer, num_layers = self.e_layer).to(self.device)
        
        # Spectral Encoder
        self.Lf = seq_len // 2 + 1   
        self.freq_token_proj = nn.Linear(1, self.channel).to(self.device)
        self.freq_attn_layer = nn.TransformerEncoderLayer(d_model=self.channel, nhead=self.head, batch_first=True,
                                                            norm_first=True, dropout=self.dropout_n).to(self.device)
        self.freq_encoder = nn.TransformerEncoder(self.freq_attn_layer, num_layers=1).to(self.device)

        self.freq_pool = FrequencyAttentionPooling(self.channel).to(self.device)    # Dynamic frequency‐domain pooling
        self.rich_horizon_gate = RichHorizonGate(self.channel).to(self.device)

        # multi head CMA
        self.cma_heads = nn.ModuleList([
            CrossModal(d_model= self.num_nodes, n_heads= 1, d_ff=self.d_ff, norm='LayerNorm', attn_dropout=self.dropout_n, 
                                dropout=self.dropout_n, pre_norm=True, activation="gelu", res_attention=True, n_layers=1, 
                                store_attn=False).to(self.device)  # single head internally
            for _ in range(self.num_cma_heads)
        ])

        # Aggregate multi heads
        self.adaptive_dynamic_heads_cma = AdaptiveDynamicHeadsCMA(num_heads=self.num_cma_heads, num_nodes=self.num_nodes, channel=self.channel, device=self.device)
   
        # Residual connection 
        self.residual_alpha = nn.Parameter(torch.ones(self.channel) * 0.5).to(self.device)  

        # Transformer decoder
        self.decoder_layer = nn.TransformerDecoderLayer(d_model = self.channel, nhead = self.head, batch_first=True, norm_first = True, dropout = self.dropout_n).to(self.device)
        self.decoder = nn.TransformerDecoder(self.decoder_layer, num_layers = self.d_layer).to(self.device)

        # Projection
        self.c_to_length = nn.Linear(self.channel, self.pred_len, bias=True).to(self.device)

    def param_num(self):
        return sum([param.nelement() for param in self.parameters()])

    def count_trainable_params(self):
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
    
    def frequency_domain_processing(self, input_data):

        freq_complex    = torch.fft.rfft(input_data, dim=-1)    # [B, N, Lf]
        freq_mag        = torch.abs(freq_complex)
        B, N, Lf        = freq_mag.shape
        
        freq_tokens = freq_mag.unsqueeze(-1)                    # [B, N, Lf, 1]
        freq_tokens = freq_tokens.reshape(B*N, Lf, 1)           # [B*N, Lf, 1]
        freq_tokens = self.freq_token_proj(freq_tokens)         # [B*N, Lf, C]

        freq_enc_out = self.freq_encoder(freq_tokens)           # [B*N, Lf, C]
        freq_enc_out = self.freq_pool(freq_enc_out)             # [B*N, C]

        freq_enc_out = freq_enc_out.reshape(B, N, self.channel) # [B, N, C] 
        return freq_enc_out
    

    def forward(self, input_data, input_data_mark, embeddings):
        input_data = input_data.float()
        input_data_mark = input_data_mark.float()
        embeddings = embeddings.float()
        embeddings = embeddings.squeeze(-1)                                 # [B, E, N]
        embeddings = embeddings.permute(0,2,1)                              # [B, N, E]

        #------ RevIN
        input_data = self.normalize_layers(input_data, 'norm')
        input_data = input_data.permute(0,2,1)                              # [B, N, L]

        #------ Frequency Encoding
        freq_enc_out = self.frequency_domain_processing(input_data)         # [B, N, C]          
        input_data = self.length_to_feature(input_data)                     # [B, N, C]

        #------ Time Series Encoding
        enc_out = self.ts_encoder(input_data)                               # [B, N, C]
        enc_out = enc_out.permute(0,2,1)                                    # [B, C, N]
  
        #------ Rich Horizon Gate
        gate = self.rich_horizon_gate(enc_out, self.pred_len)               # [B, C, 1]
        enc_out = gate * freq_enc_out.permute(0,2,1) + (1 - gate) * enc_out # [B, C, N]
        
        #------ Prompt encoding 
        embeddings = self.prompt_encoder(embeddings)                        # [B, N, E]
        embeddings = embeddings.permute(0,2,1)                              # [B, E, N]

        #------ Aggregating Multiple CMA Heads
        cma_outputs = []
        for cma_head in self.cma_heads:
            head_out = cma_head(enc_out, embeddings, embeddings)            # [B,C,N]
            cma_outputs.append(head_out)

        fused = self.adaptive_dynamic_heads_cma(cma_outputs)                # [B, C, N]

        #------ Residual Fusion 
        B, C, N = fused.shape
        alpha = self.residual_alpha.view(1, C, 1)      
        cross_out = alpha * fused + (1 - alpha) * enc_out                   # [B, C, N]
        cross_out = cross_out.permute(0, 2, 1)                              # [B, N, C]

        #------ Decoder
        dec_out = self.decoder(cross_out, cross_out)                        # [B, N, C]

        #------ Projection
        dec_out = self.c_to_length(dec_out)                                 # [B, N, L]
        dec_out = dec_out.permute(0,2,1)                                    # [B, L, N]

        #------ Denorm
        dec_out = self.normalize_layers(dec_out, 'denorm')

        return dec_out