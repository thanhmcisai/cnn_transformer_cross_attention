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


# ---------------------------------------------------------------------------
# Resubmission model registry
# ---------------------------------------------------------------------------
# The classes below supersede the early repository prototype above. They keep
# all resubmission experiments on one shared model implementation so training,
# transfer, robustness, and interpretability scripts load identical networks.

DEFAULT_IMG_SIZE = 256


def _cfg_get(model_cfg, key, default):
    return model_cfg[key] if key in model_cfg else default


def _load_features_backbone(name, pretrained=True):
    model = timm.create_model(name, pretrained=pretrained, features_only=True)
    return model, model.feature_info[-1]["num_chs"]


def _as_nchw(x, channels):
    if x.ndim == 4 and x.shape[1] != channels and x.shape[-1] == channels:
        return x.permute(0, 3, 1, 2).contiguous()
    return x


def _make_encoder(dim, num_heads, depth):
    layer = nn.TransformerEncoderLayer(
        d_model=dim,
        nhead=num_heads,
        batch_first=True,
        norm_first=True,
    )
    return nn.TransformerEncoder(layer, num_layers=depth)


class FeatureAdapter(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, out_channels, kernel_size=1)
        self.norm = nn.LayerNorm(out_channels)

    def forward(self, feat):
        feat = _as_nchw(feat, self.proj.in_channels)
        feat = self.proj(feat)
        return self.norm(feat.flatten(2).transpose(1, 2))


class TimmClassifier(nn.Module):
    def __init__(self, backbone, num_classes, pretrained=True, resize_to=None):
        super().__init__()
        self.resize_to = resize_to
        self.model = timm.create_model(backbone, pretrained=pretrained, num_classes=num_classes)

    def forward(self, x):
        if self.resize_to is not None and x.shape[-2:] != (self.resize_to, self.resize_to):
            x = F.interpolate(x, size=(self.resize_to, self.resize_to), mode="bilinear", align_corners=False)
        return self.model(x)


class A1CNN(nn.Module):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            _cfg_get(model_cfg, "local_backbone", "convnextv2_tiny"),
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def tokens(self, x):
        feat = self.model.forward_features(x)
        feat = _as_nchw(feat, self.model.num_features)
        return feat.flatten(2).transpose(1, 2)


class A2Transformer(nn.Module):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__()
        self.model = timm.create_model(
            _cfg_get(model_cfg, "global_backbone", "swinv2_tiny_window8_256"),
            pretrained=pretrained,
            num_classes=num_classes,
        )

    def forward(self, x):
        return self.model(x)

    def tokens(self, x):
        feat = self.model.forward_features(x)
        feat = _as_nchw(feat, self.model.num_features)
        return feat.flatten(2).transpose(1, 2)


class SequentialBase(nn.Module):
    def __init__(self, model_cfg, pretrained=True):
        super().__init__()
        dim = _cfg_get(model_cfg, "embed_dim", 768)
        depth = _cfg_get(model_cfg, "depth", 4)
        num_heads = _cfg_get(model_cfg, "num_heads", 8)
        img_size = _cfg_get(model_cfg, "img_size", DEFAULT_IMG_SIZE)
        local_backbone = _cfg_get(model_cfg, "local_backbone", "convnextv2_tiny")

        self.cnn, channels = _load_features_backbone(local_backbone, pretrained=pretrained)
        self.proj = nn.Conv2d(channels, dim, kernel_size=1)
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // 32) ** 2, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.transformer = _make_encoder(dim, num_heads, depth)
        self.dim = dim

    def tokens(self, x):
        feat = self.cnn(x)[-1]
        feat = _as_nchw(feat, self.proj.in_channels)
        feat = self.proj(feat)
        tokens = feat.flatten(2).transpose(1, 2)
        if tokens.shape[1] == self.pos_emb.shape[1]:
            pos = self.pos_emb
        else:
            pos = F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        return tokens + pos


class A3SequentialConcat(SequentialBase):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__(model_cfg, pretrained=pretrained)
        self.head = nn.Sequential(nn.LayerNorm(self.dim * 2), nn.Linear(self.dim * 2, num_classes))

    def forward(self, x):
        local_tokens = self.tokens(x)
        global_tokens = self.transformer(local_tokens)
        return self.head(torch.cat([local_tokens.mean(1), global_tokens.mean(1)], dim=1))


class A4SequentialSingle(SequentialBase):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__(model_cfg, pretrained=pretrained)
        self.head = nn.Sequential(nn.LayerNorm(self.dim), nn.Linear(self.dim, num_classes))

    def forward(self, x):
        return self.head(self.transformer(self.tokens(x)).mean(1))


class A5ParallelConcat(nn.Module):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__()
        dim = _cfg_get(model_cfg, "embed_dim", 768)
        self.local, local_ch = _load_features_backbone(
            _cfg_get(model_cfg, "local_backbone", "convnextv2_tiny"), pretrained=pretrained
        )
        self.global_, global_ch = _load_features_backbone(
            _cfg_get(model_cfg, "global_backbone", "swinv2_tiny_window8_256"), pretrained=pretrained
        )
        self.local_adapter = FeatureAdapter(local_ch, dim)
        self.global_adapter = FeatureAdapter(global_ch, dim)
        self.head = nn.Sequential(nn.LayerNorm(dim * 2), nn.Linear(dim * 2, num_classes))

    def forward(self, x):
        local_tokens = self.local_adapter(self.local(x)[-1])
        global_tokens = self.global_adapter(self.global_(x)[-1])
        return self.head(torch.cat([local_tokens.mean(1), global_tokens.mean(1)], dim=1))


class A6ParallelCrossAttention(nn.Module):
    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__()
        dim = _cfg_get(model_cfg, "embed_dim", 768)
        depth = _cfg_get(model_cfg, "depth", 4)
        num_heads = _cfg_get(model_cfg, "num_heads", 8)
        self.local, local_ch = _load_features_backbone(
            _cfg_get(model_cfg, "local_backbone", "convnextv2_tiny"), pretrained=pretrained
        )
        self.global_, global_ch = _load_features_backbone(
            _cfg_get(model_cfg, "global_backbone", "swinv2_tiny_window8_256"), pretrained=pretrained
        )
        self.local_adapter = FeatureAdapter(local_ch, dim)
        self.global_adapter = FeatureAdapter(global_ch, dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.refine = _make_encoder(dim, num_heads, depth)
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def forward(self, x):
        local_tokens = self.local_adapter(self.local(x)[-1])
        global_tokens = self.global_adapter(self.global_(x)[-1])
        attn_out, _ = self.cross_attn(local_tokens, global_tokens, global_tokens)
        fused = local_tokens + torch.sigmoid(self.gate) * attn_out
        return self.head(self.refine(fused).mean(1))


class WiT(nn.Module):
    """Sequential CNN -> Transformer plus query-guided cross-attention."""

    def __init__(self, model_cfg, num_classes, pretrained=True):
        super().__init__()
        dim = _cfg_get(model_cfg, "embed_dim", 768)
        depth = _cfg_get(model_cfg, "depth", 4)
        num_heads = _cfg_get(model_cfg, "num_heads", 8)
        img_size = _cfg_get(model_cfg, "img_size", DEFAULT_IMG_SIZE)
        self.query_direction = _cfg_get(model_cfg, "query_direction", "local")
        self.cnn, channels = _load_features_backbone(
            _cfg_get(model_cfg, "local_backbone", "convnextv2_tiny"), pretrained=pretrained
        )
        self.proj = nn.Conv2d(channels, dim, kernel_size=1)
        self.pos_emb = nn.Parameter(torch.zeros(1, (img_size // 32) ** 2, dim))
        nn.init.trunc_normal_(self.pos_emb, std=0.02)
        self.transformer = _make_encoder(dim, num_heads, depth)
        self.q_norm = nn.LayerNorm(dim)
        self.kv_norm = nn.LayerNorm(dim)
        self.cross_attn = nn.MultiheadAttention(dim, num_heads, batch_first=True)
        self.gate = nn.Parameter(torch.tensor(0.0))
        self.head = nn.Sequential(nn.LayerNorm(dim), nn.Linear(dim, num_classes))

    def tokens(self, x):
        feat = self.cnn(x)[-1]
        feat = _as_nchw(feat, self.proj.in_channels)
        feat = self.proj(feat)
        local_tokens = feat.flatten(2).transpose(1, 2)
        if local_tokens.shape[1] == self.pos_emb.shape[1]:
            pos = self.pos_emb
        else:
            pos = F.interpolate(
                self.pos_emb.transpose(1, 2),
                size=local_tokens.shape[1],
                mode="linear",
                align_corners=False,
            ).transpose(1, 2)
        local_tokens = local_tokens + pos
        return local_tokens, self.transformer(local_tokens)

    def fused_tokens(self, x, need_weights=False):
        local_tokens, global_tokens = self.tokens(x)
        if self.query_direction == "global":
            query, key_value, anchor = self.q_norm(global_tokens), self.kv_norm(local_tokens), global_tokens
        else:
            query, key_value, anchor = self.q_norm(local_tokens), self.kv_norm(global_tokens), local_tokens
        attn_out, attn_weights = self.cross_attn(query, key_value, key_value, need_weights=need_weights)
        fused = anchor + torch.sigmoid(self.gate) * attn_out
        return (fused, attn_weights) if need_weights else fused

    def forward(self, x):
        return self.head(self.fused_tokens(x).mean(1))


MODEL_ALIASES = {
    "A1": "A1_CNN",
    "A2": "A2_Transformer",
    "A3": "A3_Seq_Concat",
    "A4": "A4_Seq_Single",
    "A5": "A5_Par_Concat",
    "A6": "A6_Par_CrossAttn",
    "A7": "WiT",
}


def canonical_model_name(model_name):
    return MODEL_ALIASES.get(model_name, model_name)


def get_model_cfg_from_registry(cfg, model_name):
    canonical = canonical_model_name(model_name)
    models_cfg = cfg["experiments_cfg"]["models"]
    if canonical not in models_cfg:
        raise ValueError(f"Unknown model name: {model_name}")
    return models_cfg[canonical]


def build_model(model_name, cfg, num_classes, pretrained=True):
    canonical = canonical_model_name(model_name)
    model_cfg = get_model_cfg_from_registry(cfg, canonical)
    model_type = model_cfg.get("type", "baseline")

    if canonical == "A1_CNN":
        return A1CNN(model_cfg, num_classes, pretrained=pretrained)
    if canonical == "A2_Transformer":
        return A2Transformer(model_cfg, num_classes, pretrained=pretrained)
    if canonical == "A3_Seq_Concat":
        return A3SequentialConcat(model_cfg, num_classes, pretrained=pretrained)
    if canonical == "A4_Seq_Single":
        return A4SequentialSingle(model_cfg, num_classes, pretrained=pretrained)
    if canonical == "A5_Par_Concat":
        return A5ParallelConcat(model_cfg, num_classes, pretrained=pretrained)
    if canonical == "A6_Par_CrossAttn":
        return A6ParallelCrossAttention(model_cfg, num_classes, pretrained=pretrained)
    if canonical.startswith("WiT"):
        return WiT(model_cfg, num_classes, pretrained=pretrained)
    if model_type == "baseline":
        return TimmClassifier(
            model_cfg["backbone"],
            num_classes,
            pretrained=pretrained,
            resize_to=model_cfg.get("resize_to"),
        )
    raise ValueError(f"Unknown model name: {model_name}")
