# src/models/models.py

import timm
import torch
import torch.nn as nn
import torch.nn.functional as F


# -------------------------------------------------
# Utility: backbone loader
# -------------------------------------------------
def load_backbone(backbone_name, pretrained=True, num_classes=0, features_only=False):
    if features_only:
        model = timm.create_model(backbone_name, pretrained=pretrained, features_only=True)
        feat_dim = model.feature_info[-1]["num_chs"]
        return model, feat_dim
    else:
        model = timm.create_model(backbone_name, pretrained=pretrained, num_classes=num_classes)
        return model


# -------------------------------------------------
# Cross Attention Block
# -------------------------------------------------
class CrossAttentionBlock(nn.Module):
    def __init__(self, dim_q, dim_kv, embed_dim, num_heads):
        super().__init__()
        self.q_proj = nn.Linear(dim_q, embed_dim)
        self.k_proj = nn.Linear(dim_kv, embed_dim)
        self.v_proj = nn.Linear(dim_kv, embed_dim)

        self.attn = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
        self.norm = nn.LayerNorm(embed_dim)

    def forward(self, q_feat, kv_feat):
        q = self.q_proj(q_feat)
        k = self.k_proj(kv_feat)
        v = self.v_proj(kv_feat)

        out, _ = self.attn(q, k, v)
        out = self.norm(out + q)
        return out


# -------------------------------------------------
# A1: Local only (ConvNeXt)
# -------------------------------------------------
class A1_LocalOnly(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.model = timm.create_model(
            cfg["local_backbone"],
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------------------------
# A2: Global only (Swin)
# -------------------------------------------------
class A2_GlobalOnly(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.model = timm.create_model(
            cfg["global_backbone"],
            pretrained=True,
            num_classes=num_classes
        )

    def forward(self, x):
        return self.model(x)


# -------------------------------------------------
# A3: Sequential Concat (CNN -> Transformer)
# -------------------------------------------------
class A3_SequentialConcat(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.local, ldim = load_backbone(cfg["local_backbone"], features_only=True)

        embed_dim = cfg["embed_dim"]
        nhead = cfg["nhead"]
        depth = cfg["depth"]

        self.in_proj = nn.Linear(ldim, embed_dim)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.fc = nn.Sequential(
            nn.Linear(ldim + embed_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        feat = self.local(x)[-1]  # [B,C,H,W]
        B, C, H, W = feat.shape

        local_pool = feat.mean(dim=[2, 3])

        tokens = feat.flatten(2).transpose(1, 2)  # [B,HW,C]
        tokens = self.in_proj(tokens)

        g = self.transformer(tokens)
        global_pool = g.mean(dim=1)

        fused = torch.cat([local_pool, global_pool], dim=1)
        return self.fc(fused)


# -------------------------------------------------
# A4: Sequential Local -> Global (CLS token)
# -------------------------------------------------
class A4_SequentialLocalToGlobal(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.local, ldim = load_backbone(cfg["local_backbone"], features_only=True)

        embed_dim = cfg["embed_dim"]
        nhead = cfg["nhead"]
        depth = cfg["depth"]

        self.in_proj = nn.Linear(ldim, embed_dim)
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=embed_dim,
            nhead=nhead,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers=depth)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        feat = self.local(x)[-1]
        B, C, H, W = feat.shape

        tokens = feat.flatten(2).transpose(1, 2)
        tokens = self.in_proj(tokens)

        cls = self.cls_token.expand(B, -1, -1)
        tokens = torch.cat([cls, tokens], dim=1)

        out = self.transformer(tokens)
        return self.head(out[:, 0])


# -------------------------------------------------
# A5: Dual Branch Concat
# -------------------------------------------------
class A5_DualBranchConcat(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.local = timm.create_model(cfg["local_backbone"], pretrained=True, num_classes=0)
        self.global_ = timm.create_model(cfg["global_backbone"], pretrained=True, num_classes=0)

        dim = self.local.num_features + self.global_.num_features

        self.fc = nn.Sequential(
            nn.Linear(dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        l = self.local(x)
        g = self.global_(x)
        fused = torch.cat([l, g], dim=1)
        return self.fc(fused)


# -------------------------------------------------
# A6: Dual Branch Cross Attention
# -------------------------------------------------
class A6_CrossAttention(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.local, ldim = load_backbone(cfg["local_backbone"], features_only=True)
        self.global_, gdim = load_backbone(cfg["global_backbone"], features_only=True)

        embed_dim = cfg["embed_dim"]
        num_heads = cfg["num_heads"]

        self.cross_attn = CrossAttentionBlock(ldim, gdim, embed_dim, num_heads)

        self.head = nn.Sequential(
            nn.LayerNorm(embed_dim),
            nn.Linear(embed_dim, num_classes),
        )

    def forward(self, x):
        lf = self.local(x)[-1]
        gf = self.global_(x)[-1]

        l_tokens = lf.flatten(2).transpose(1, 2)
        g_tokens = gf.flatten(2).transpose(1, 2)

        fused = self.cross_attn(l_tokens, g_tokens)
        pooled = fused.mean(dim=1)

        return self.head(pooled)


# -------------------------------------------------
# A7: Final Proposed Model (2-stage cross attention)
# -------------------------------------------------
class A7_FinalModel(nn.Module):
    def __init__(self, cfg, num_classes):
        super().__init__()
        self.local, ldim = load_backbone(cfg["local_backbone"], features_only=True)
        self.global_, gdim = load_backbone(cfg["global_backbone"], features_only=True)

        embed_dim = cfg["embed_dim"]
        num_heads = cfg["num_heads"]
        hidden_dim = cfg["hidden_dim"]

        self.cross1 = CrossAttentionBlock(ldim, gdim, embed_dim, num_heads)
        self.cross2 = CrossAttentionBlock(embed_dim, embed_dim, embed_dim, num_heads)

        self.fc = nn.Sequential(
            nn.Linear(embed_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x):
        lf = self.local(x)[-1]
        gf = self.global_(x)[-1]

        l_tokens = lf.flatten(2).transpose(1, 2)
        g_tokens = gf.flatten(2).transpose(1, 2)

        f1 = self.cross1(l_tokens, g_tokens)
        f2 = self.cross2(f1, f1)

        pooled = f2.mean(dim=1)
        return self.fc(pooled)


# -------------------------------------------------
# Factory
# -------------------------------------------------
def build_model(model_name: str, cfg: dict, num_classes: int):
    model_cfg = cfg["experiments_cfg"]["models"][model_name]

    # ===== BASELINE MODELS =====
    if model_cfg["type"] == "baseline":
        return timm.create_model(
            model_cfg["backbone"],
            pretrained=True,
            num_classes=num_classes
        )

    # ===== PROPOSED / ABLATION =====
    if model_name == "A1":
        return A1_LocalOnly(model_cfg, num_classes)

    elif model_name == "A2":
        return A2_GlobalOnly(model_cfg, num_classes)

    elif model_name == "A3":
        return A3_SequentialConcat(model_cfg, num_classes)

    elif model_name == "A4":
        return A4_SequentialLocalToGlobal(model_cfg, num_classes)

    elif model_name == "A5":
        return A5_DualBranchConcat(model_cfg, num_classes)

    elif model_name == "A6":
        return A6_CrossAttention(model_cfg, num_classes)

    elif model_name == "A7":
        return A7_FinalModel(model_cfg, num_classes)

    else:
        raise ValueError(f"Unknown model name: {model_name}")
