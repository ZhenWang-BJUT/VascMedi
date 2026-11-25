"""原始 cotstrong 骨干的三 Token、多参考 NWGM 预测版本。

主预测为 (1-lambda) 个体条件 logits + lambda 个训练参考条件 logits 均值。
这里的干预量是固定模型下的预测响应，不是临床因果效应。

基于 StrokeCausalModel 扩展，将单一中介分解为 8 个病理模块中介：
  M1: 分支形态 (7 维)   - 血管网络复杂度
  M2: 弯曲度 (15 维)    - 微循环阻力/血管扭曲
  M3: 血管密度 (3 维)   - 微循环灌注
  M4: 动脉管径 (5 维)   - 动脉硬化/狭窄
  M5: 静脉管径 (5 维)   - 静脉淤血/回流障碍
  M6: 血管弓形态 (2 维) - 血管弓结构异常
  M7: 视盘视杯 (18 维)  - 颅内压/神经退行性变
  M8: 动静脉比 (3 维)   - 微循环整体状态

解释接口：model_path_effects 和 tabular_intervention_response；均使用最终预测函数。
默认损失：融合 logits 的主 CE、真实配对 CE、八模块重建、
Token/Query 相关性及模块独立性约束；旧融合向量辅助损失默认关闭。
"""

from typing import Optional, Dict
import os
import logging

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from peft import get_peft_model, prepare_model_for_kbit_training

from .llama.configuration_llama import LlamaConfig
from .llama.modeling_llama import LlamaForCausalLM
from .lora import generate_peft_config

if torch.cuda.is_available():
    device = torch.device("cuda")
else:
    device = torch.device("cpu")

logger = logging.getLogger(__name__)


def _visual_global_features(flat_local, img_global, source):
    """Select an explicitly named global descriptor; no CLIP parameter changes."""
    if source == "patch_mean":
        return flat_local.mean(dim=1)
    if source == "clip_pooler":
        return img_global.reshape(-1, img_global.size(-1))
    raise ValueError("visual_global_source must be patch_mean or clip_pooler")


def _semantic_readout_mask(attention_mask, image_token_mask, scope):
    """Causal suffix states can see the image; prefix text states cannot."""
    if scope == "all_tokens":
        return attention_mask
    if scope != "after_image":
        raise ValueError("semantic_readout_scope must be all_tokens or after_image")
    image_token_mask = image_token_mask.bool()
    if image_token_mask.shape != attention_mask.shape or not torch.all(image_token_mask.sum(1) == 1):
        raise ValueError("after_image semantic pooling requires exactly one merged image token")
    positions = torch.arange(attention_mask.size(1), device=attention_mask.device).unsqueeze(0)
    image_position = image_token_mask.long().argmax(1, keepdim=True)
    result = attention_mask.bool() & (positions > image_position)
    if not result.any(1).all():
        raise ValueError("No image-conditioned suffix text remains for semantic pooling")
    return result.to(attention_mask.dtype)


def _reconstruction_observed_mask(tabular_features, tabular_observed_mask, view_mask):
    """Select genuinely measured table cells in image-valid view slots."""
    if tabular_observed_mask is None:
        observed = torch.ones_like(tabular_features, dtype=torch.bool)
    else:
        observed = torch.as_tensor(
            tabular_observed_mask, device=tabular_features.device, dtype=torch.bool
        )
        if observed.shape != tabular_features.shape:
            raise ValueError("tabular_observed_mask must match tabular_features [B,V,58]")
    if view_mask.shape != tabular_features.shape[:2]:
        raise ValueError("view_mask must match tabular_features [B,V]")
    return observed & view_mask.to(device=observed.device, dtype=torch.bool).unsqueeze(-1)


# ============================================================================
# 子模块
# ============================================================================

# 8 个病理模块的特征索引定义（去掉 sample_id, v_idx, PatientID 后的 0-57）
FEATURE_MODULES = {
    "branching":       list(range(0, 7)),     # 分支夹角、分形维数、总管径、4个PD区间管径
    "tortuosity":      list(range(7, 22)),    # 总弯曲率5 + 动脉弯曲度5 + 静脉弯曲度5
    "density":         list(range(22, 25)),   # 总密度、黄斑3mm密度、黄斑5mm密度
    "arterial_caliber": list(range(25, 30)),  # 动脉平均管径 + 4个PD区间
    "venular_caliber": list(range(30, 35)),   # 静脉平均管径 + 4个PD区间
    "vascular_arch":   list(range(35, 37)),   # 血管弓与视盘外1PD/2PD夹角
    "optic_disc_cup":  list(range(37, 55)),   # 视盘8列 + 视杯10列
    "av_ratio":        list(range(55, 58)),   # 主血管弓A/V比 + 上/下侧
}

MODULE_NAMES = list(FEATURE_MODULES.keys())
N_MODULES = len(MODULE_NAMES)
COT_SEMANTIC_MODULE = "cot_semantic"


def build_random_module_partition(total_dim: int = 58, seed: int = 42):
    """M4 ablation: 随机将 total_dim 个特征划分为 N_MODULES 个等大小桶（保持原始模块大小）。

    返回与 FEATURE_MODULES 形状一致的 dict，但索引被随机重排。
    """
    rng = np.random.default_rng(seed)
    perm = rng.permutation(total_dim).tolist()
    cursor = 0
    random_modules = {}
    for name, indices in FEATURE_MODULES.items():
        k = len(indices)
        random_modules[name] = sorted(perm[cursor:cursor + k])
        cursor += k
    return random_modules

class ModularMediatorEncoder(nn.Module):
    """
    多模块中介编码器：将 58 维血管特征按病理机制分为 N 个独立模块编码。
    每个模块输出固定维度 module_dim 的表示，拼接后投影回 hidden_dim。

    M4 ablation: 通过传入自定义 feature_modules 可使用随机模块划分。
    """
    def __init__(self, hidden_dim: int = 256, dropout: float = 0.1,
                 feature_modules: Optional[Dict[str, list]] = None):
        super().__init__()
        if feature_modules is None:
            feature_modules = {n: list(idx) for n, idx in FEATURE_MODULES.items()}
        self.feature_modules = feature_modules
        self.module_names = list(feature_modules.keys())
        self.hidden_dim = hidden_dim
        n_modules = len(self.module_names)
        self.module_dim = hidden_dim // max(n_modules, 1)

        # 每个模块独立的编码器
        self.encoders = nn.ModuleDict()
        for name, indices in feature_modules.items():
            input_dim = len(indices)
            self.encoders[name] = nn.Sequential(
                nn.Linear(input_dim, self.module_dim * 2),
                nn.LayerNorm(self.module_dim * 2),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(self.module_dim * 2, self.module_dim),
                nn.LayerNorm(self.module_dim),
                nn.ReLU(),
            )

        # 拼接所有模块后投影回 hidden_dim
        total_dim = self.module_dim * n_modules
        self.aggregate_proj = nn.Sequential(
            nn.Linear(total_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

    def forward(self, x, return_aggregate=True):
        """
        x: [B, 58] 原始表格特征（已 z-score 标准化）
        返回:
          f_t: [B, hidden_dim] 聚合表示（兼容旧流程）
          modules: dict {name: [B, module_dim]} 各模块独立表示
        """
        modules = {}
        for name, indices in self.feature_modules.items():
            m_input = x[:, indices]
            modules[name] = self.encoders[name](m_input)

        f_t = None
        if return_aggregate:
            cat_modules = torch.cat([modules[n] for n in self.module_names], dim=-1)
            f_t = self.aggregate_proj(cat_modules)
        return f_t, modules


class LocalVascularAttention(nn.Module):
    """
    局部血管注意力模块
    对 ViT token 特征做 top-k 注意力加权聚合 → Z_v_loc
    """
    def __init__(self, feat_dim: int, top_k: int = 64):
        super().__init__()
        self.query_w = nn.Linear(feat_dim, 1, bias=False)
        self.top_k = top_k

    def forward(self, F_v):
        """
        F_v: [B, N, feat_dim]  (N=257 for ViT-L)
        返回: Z_v_loc [B, feat_dim], alpha_full [B, N]
        """
        alpha = self.query_w(F_v).squeeze(-1)
        alpha_full = torch.softmax(alpha, dim=-1)

        k = min(self.top_k, F_v.size(1))
        topk_vals, topk_idx = torch.topk(alpha, k, dim=-1)
        topk_alpha = torch.softmax(topk_vals, dim=-1)

        idx_exp = topk_idx.unsqueeze(-1).expand(-1, -1, F_v.size(-1))
        topk_feats = torch.gather(F_v, 1, idx_exp)

        Z_v_loc = (topk_alpha.unsqueeze(-1) * topk_feats).sum(dim=1)
        return Z_v_loc, alpha_full


class FrontDoorAttention(nn.Module):
    """
    两层共享的查询到结构化中介 Cross-Attention。

    输入 F:[B,1,D]，M:[B,3,D]；key_padding_mask 中 True 表示无效 Token。
    """
    def __init__(self, hidden_dim: int, num_heads: int):
        super().__init__()
        self.attn1 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden_dim)
        self.attn2 = nn.MultiheadAttention(hidden_dim, num_heads, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden_dim)
        # R01/R02: query determines attention weights, without entering values
        # through the first-layer residual. This is distinct from freezing rho.
        self.query_value_residual = True
        self.query_gate_enabled = False
        self.query_residual_logit = nn.Parameter(torch.tensor(0.0), requires_grad=False)

    def forward(self, F_seq, M_seq, key_padding_mask=None, return_attention=False):
        """
        F_seq: [B, 1, d]
        M_seq: [B, 3, d]
        """
        H1, weights1 = self.attn1(
            F_seq, M_seq, M_seq, key_padding_mask=key_padding_mask,
            need_weights=return_attention, average_attn_weights=False,
        )
        rho = self.query_residual_logit.sigmoid() if self.query_gate_enabled else 1.0
        query_content = rho * F_seq if self.query_value_residual else torch.zeros_like(F_seq)
        F_tilde = self.norm1(query_content + H1)

        H2, weights2 = self.attn2(
            F_tilde, M_seq, M_seq, key_padding_mask=key_padding_mask,
            need_weights=return_attention, average_attn_weights=False,
        )
        F_fd = self.norm2(F_tilde + H2)

        if return_attention:
            return F_fd.squeeze(1), {"layer1": weights1, "layer2": weights2}
        return F_fd.squeeze(1)



class CoTSemanticMediator(nn.Module):
    """
    从 LLaMA 的 prompt/CoT 隐状态抽取任务语义 Token。
    旧的结构化中介融合仅供单 Token 对照；默认三 Token 路径不调用它。
    """
    def __init__(
        self,
        llama_dim: int,
        hidden_dim: int,
        dropout: float = 0.1,
        gate_init_bias: float = 6.0,
    ):
        super().__init__()
        self.token_proj = nn.Sequential(
            nn.Linear(llama_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
        )
        self.semantic_query = nn.Linear(hidden_dim, 1, bias=False)
        self.fusion_residual = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
        self.residual_scale = nn.Parameter(torch.zeros(1))
        self.fusion_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.output_norm = nn.Identity()
        nn.init.zeros_(self.fusion_gate.weight)
        nn.init.constant_(self.fusion_gate.bias, gate_init_bias)
        nn.init.zeros_(self.fusion_residual[-1].weight)
        nn.init.zeros_(self.fusion_residual[-1].bias)

    def extract_semantic(self, last_hidden, attention_mask):
        token_sem = self.token_proj(last_hidden)
        scores = self.semantic_query(token_sem).squeeze(-1)
        scores = scores.masked_fill(attention_mask == 0, torch.finfo(scores.dtype).min)
        weights = torch.softmax(scores, dim=-1)
        return (weights.unsqueeze(-1) * token_sem).sum(dim=1)

    def forward(self, last_hidden, attention_mask, m_struct):
        # Legacy single-token ablation only. The three-token path calls
        # extract_semantic and never constructs the old fused mediator.
        m_cot = self.extract_semantic(last_hidden, attention_mask)
        fusion_input = torch.cat([m_struct, m_cot, m_struct * m_cot], dim=-1)
        gate = torch.sigmoid(self.fusion_gate(fusion_input))
        residual = self.fusion_residual(fusion_input)
        m_fused = self.output_norm(
            gate * m_struct + (1.0 - gate) * m_cot + self.residual_scale * residual
        )
        return m_fused, m_cot


# ============================================================================
# 主模型
# ============================================================================

class ImageResidualQuery(nn.Module):
    """Add the LLaMA image-position state to the existing pooled query.

    The residual starts at one percent of the projected image state.  Both
    the current patient and training-reference patients use this same module.
    """

    def __init__(self, llama_dim: int, hidden_dim: int):
        super().__init__()
        self.image_norm = nn.LayerNorm(llama_dim)
        self.image_proj = nn.Linear(llama_dim, hidden_dim)
        self.gate = nn.Parameter(torch.tensor(0.01, dtype=torch.float32))

    def forward(self, pooled_query: torch.Tensor, image_hidden: torch.Tensor):
        image_hidden = image_hidden.to(
            device=pooled_query.device, dtype=pooled_query.dtype
        )
        residual = self.image_proj(self.image_norm(image_hidden))
        return pooled_query + self.gate.to(dtype=residual.dtype) * residual


class StrokeCausalModelModular(nn.Module):
    """
    原始 cotstrong 多模态骨干 + 结构化中介 + 多参考共享条件预测头。

    默认三 Token 路径保留原始 8 模块编码和 Bi-MHA，但不构造旧的
    单向量中介。融合 logits 是两条条件预测分支的输出聚合。
    解释函数只量化固定模型的预测响应，不提供临床因果识别。
    """

    def __init__(
        self,
        train_config,
        use_cache,
        tokenizer,
        image_token_index,
        kwargs,
        wandb_run,
        class_weights: Optional[torch.FloatTensor] = None,
        tabular_input_dim: int = 58,
        hidden_dim: int = 256,
        top_k: int = 64,
        num_heads: int = 8,
        alpha: float = 2.0,       # 增加中介约束强度（从1.0→2.0）
        beta: float = 0.1,
        gamma: float = 0.0,
        delta: float = 0.0,       # 移除完全独立约束（原0.3→0.0）
    ):
        super().__init__()

        self.image_token_index = image_token_index
        self.ignore_index = -100
        self.stop_token_id = 128001
        self.pad_token_id = self.stop_token_id
        self.context_length = train_config.context_length
        self.train_config = train_config
        self.hidden_dim = hidden_dim
        self.visual_global_source = str(getattr(train_config, "visual_global_source", "patch_mean"))
        self.semantic_readout_scope = str(getattr(train_config, "semantic_readout_scope", "all_tokens"))
        if self.visual_global_source not in {"patch_mean", "clip_pooler"}:
            raise ValueError("Unknown visual_global_source")
        if self.semantic_readout_scope not in {"all_tokens", "after_image"}:
            raise ValueError("Unknown semantic_readout_scope")
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.delta = delta
        self.mediator_cls_weight = float(getattr(train_config, "mediator_cls_weight", 0.0))
        # 新增真实配对辅助监督；原 cotstrong 没有这个项。
        self.tortuosity_module_drop_prob = float(getattr(train_config, "tortuosity_module_drop_prob", 0.0))
        if not 0 <= self.tortuosity_module_drop_prob <= 1:
            raise ValueError("tortuosity_module_drop_prob must be in [0,1]")
        self.pure_visual_path = bool(getattr(train_config, "pure_visual_path", False))
        self.image_aux_weight = float(getattr(train_config, "image_aux_weight", 0.0))
        self.raw_tab_residual = float(getattr(train_config, "raw_tab_residual", 0.0))
        self.tab_token_drop_prob = float(getattr(train_config, "tab_token_drop_prob", 0.0))
        if self.image_aux_weight < 0 or self.raw_tab_residual < 0 or not 0 <= self.tab_token_drop_prob < 1:
            raise ValueError("Invalid v19 intervention settings")
        self.observed_cls_weight = float(getattr(train_config, "observed_cls_weight", 0.1))
        if self.observed_cls_weight < 0:
            raise ValueError("observed_cls_weight must be nonnegative")
        self.auxiliary_mode = str(getattr(train_config, "auxiliary_mode", "off")).lower()
        if self.auxiliary_mode not in {"off", "token_query"}:
            raise ValueError("auxiliary_mode must be 'off' or 'token_query'")
        self.query_mode = str(getattr(train_config, "query_mode", "legacy")).lower()
        if self.query_mode not in {"legacy", "image_residual"}:
            raise ValueError("query_mode must be 'legacy' or 'image_residual'")
        if self.auxiliary_mode == "off" and (self.gamma != 0 or self.mediator_cls_weight != 0):
            raise ValueError("auxiliary_mode=off requires gamma=0 and mediator_cls_weight=0")
        self.tokenizer = tokenizer

        # ── Priority-1 ablation toggles ─────────────────────────────────
        self.ab_degenerate     = bool(getattr(train_config, "ablation_degenerate_to_single", False))
        self.ab_no_shapley     = bool(getattr(train_config, "ablation_no_shapley", False))
        self.ab_random_modules = bool(getattr(train_config, "ablation_random_modules", False))
        self.ab_single_decoder = bool(getattr(train_config, "ablation_single_decoder", False))
        self.ab_disable_cot_mediator = bool(getattr(train_config, "ablation_disable_cot_mediator", False))
        self.ab_disable_front_door = bool(getattr(train_config, "ablation_disable_front_door", False))
        self.ab_disable_bidir_guidance = bool(getattr(train_config, "ablation_disable_bidir_guidance", False))
        self.ab_disable_visual_to_tab = bool(getattr(train_config, "ablation_disable_visual_to_tab", False))
        self.ab_disable_tab_to_visual = bool(getattr(train_config, "ablation_disable_tab_to_visual", False))
        self.ab_disable_module_mediator_proj = bool(getattr(train_config, "ablation_disable_module_mediator_proj", False))
        self.ab_mediator_source = str(getattr(train_config, "ablation_mediator_source", "full")).lower()
        if self.ab_mediator_source not in {"full", "visual_only", "tabular_only"}:
            self.ab_mediator_source = "full"

        # M4: 随机模块划分（在初始化时根据 seed 决定）
        if self.ab_random_modules:
            seed = int(getattr(train_config, "seed", 42))
            self.feature_modules = build_random_module_partition(tabular_input_dim, seed=seed)
        else:
            self.feature_modules = {n: list(idx) for n, idx in FEATURE_MODULES.items()}
        self.module_names = list(self.feature_modules.keys())
        self.visual_features_dir = getattr(train_config, "visual_features_dir", "") or kwargs.get("visual_features_dir", "")
        self._visual_features_dir_logged = False
        self._missing_feature_logged_samples = set()
        self._missing_feature_log_limit = 10
        if class_weights is not None:
            self.register_buffer(
                "loss_class_weights",
                torch.as_tensor(class_weights, dtype=torch.float32),
            )
        else:
            self.register_buffer("loss_class_weights", None)

        llama_dim = 4096
        vit_dim = 1024
        module_dim = hidden_dim // N_MODULES  # 32

        # ── MLP Projector（与旧版相同）──────────────────────────────
        self.multi_modal_projector = nn.Sequential(
            nn.Linear(vit_dim, llama_dim),
            nn.GELU(),
            nn.Linear(llama_dim, llama_dim),
        )

        # ── LLaMA 骨干（与旧版相同）──────────────────────────────────
        use_quantization = train_config.quantization and not getattr(train_config, 'enable_fsdp', False)
        rank = int(os.environ.get("RANK", "0"))
        use_low_cpu_loading = bool(
            getattr(train_config, "low_cpu_fsdp", False) or getattr(train_config, "enable_fsdp", False)
        )
        rank0_only_loading = bool(getattr(train_config, "enable_fsdp", False) and getattr(train_config, "low_cpu_fsdp", False))
        if rank0_only_loading and rank != 0:
            llama_config = LlamaConfig.from_pretrained(train_config.model_name)
            with torch.device("meta"):
                self.language_model = LlamaForCausalLM(llama_config)
        else:
            self.language_model = LlamaForCausalLM.from_pretrained(
                train_config.model_name,
                load_in_4bit=True if use_quantization else None,
                device_map="auto" if use_quantization else None,
                low_cpu_mem_usage=use_low_cpu_loading or use_quantization,
                offload_state_dict=use_low_cpu_loading,
                use_cache=use_cache,
                attn_implementation="sdpa" if train_config.use_fast_kernels else None,
            )
        self.base_vocab_size = int(
            self.language_model.get_input_embeddings().weight.shape[0]
        )
        if len(tokenizer) > self.base_vocab_size:
            self.language_model.resize_token_embeddings(len(tokenizer))
        if use_quantization:
            self.language_model = prepare_model_for_kbit_training(self.language_model)
        if train_config.use_peft:
            peft_config = generate_peft_config(train_config, kwargs)
            self.language_model = get_peft_model(self.language_model, peft_config)
            self.language_model.print_trainable_parameters()
            if wandb_run:
                wandb_run.config.update(peft_config)
        if getattr(train_config, "gradient_checkpointing", False):
            if hasattr(self.language_model, "gradient_checkpointing_enable"):
                self.language_model.gradient_checkpointing_enable()
            if hasattr(self.language_model, "enable_input_require_grads"):
                self.language_model.enable_input_require_grads()
            self.language_model.config.use_cache = False

        # ── 视觉解混淆（与旧版相同）──────────────────────────────────
        self.local_vascular_attn = LocalVascularAttention(vit_dim, top_k)
        self.visual_mediator_proj = nn.Sequential(
            nn.Linear(vit_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # ── 【改动1】多模块表格编码器（替代 TabularEncoder）───────────
        self.modular_encoder = ModularMediatorEncoder(
            hidden_dim, dropout=0.1, feature_modules=self.feature_modules,
        )
        # ── 【方法A】CG-DMC: 跨模态引导的双中介构建 ───────────────────
        # Step 1: 将视觉局部/全局特征映射到统一 hidden 空间
        self.visual_local_mapper = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.visual_global_mapper = nn.Sequential(
            nn.Linear(vit_dim, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )

        # Step 2: 将每个表格病理模块提升为 cross-attention token
        self.tab_module_to_hidden = nn.ModuleDict({
            name: nn.Linear(module_dim, hidden_dim)
            for name in MODULE_NAMES
        })
        self.hidden_to_tab_module = nn.ModuleDict({
            name: nn.Linear(hidden_dim, module_dim)
            for name in MODULE_NAMES
        })

        # Step 3: 双向跨模态引导
        self.tab_to_visual_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.visual_to_tab_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_guided_norm_v = nn.LayerNorm(hidden_dim)
        self.cross_guided_norm_q = nn.LayerNorm(hidden_dim)

        # Step 4: 表格引导的视觉中介 M_i
        self.visual_base_proj = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.visual_mediator_integrator = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.visual_mediator_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.visual_mediator_norm = nn.LayerNorm(hidden_dim)

        # Step 5: 视觉校正的表格模块表示
        self.tabular_module_gates = nn.ModuleDict({
            name: nn.Linear(module_dim * 2, module_dim)
            for name in MODULE_NAMES
        })
        self.tabular_module_norms = nn.ModuleDict({
            name: nn.LayerNorm(module_dim)
            for name in MODULE_NAMES
        })

        # Step 6: 旧版双中介门控层，保留用于加载旧 checkpoint/消融对比
        self.dual_mediator_residual = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.dual_mediator_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        self.dual_mediator_norm = nn.LayerNorm(hidden_dim)

        # ── 跨模态交互（保留旧层，便于加载旧 checkpoint/消融）──────────
        self.cross_modal_attn = nn.MultiheadAttention(
            embed_dim=hidden_dim, num_heads=num_heads, batch_first=True
        )
        self.cross_modal_norm = nn.LayerNorm(hidden_dim)

        # ── 模块级表格中介投影 ───────────────────────────────────────
        # 每个 M_k 只由对应医学表格模块生成，保持模块中介的医学语义纯度。
        self.module_mediator_projs = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(module_dim, module_dim),
                nn.LayerNorm(module_dim),
                nn.ReLU(),
            )
            for name in MODULE_NAMES
        })
        # 聚合所有表格模块中介 → M_tab
        self.module_aggregate = nn.Sequential(
            nn.Linear(module_dim * N_MODULES, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        # 融合表格结构中介 M_tab 与视觉中介 M_i → M_struct
        self.struct_mediator_residual = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.LayerNorm(hidden_dim),
            nn.ReLU(),
        )
        self.struct_mediator_gate = nn.Linear(hidden_dim * 3, hidden_dim)
        nn.init.constant_(self.struct_mediator_gate.bias, 1.0)
        self.struct_mediator_norm = nn.LayerNorm(hidden_dim)

        # ── 【方案A】各模块独立输出投影：module_dim → slot_dim ────────
        # 旧版 slot 参数仅保留以兼容已有骨干；新版主预测使用三类 Token。
        self.slot_dim = hidden_dim // (N_MODULES + 1)   # 256//9 = 28
        self.module_out_projs = nn.ModuleDict({
            name: nn.Linear(module_dim, self.slot_dim)
            for name in MODULE_NAMES
        })
        # m_t（跨模态交互）独立占一个 slot
        self.m_t_proj = nn.Linear(hidden_dim, self.slot_dim)
        # 拼接后对齐到 hidden_dim（N_MODULES+1 个 slot → hidden_dim）
        self.slot_align = nn.Linear((N_MODULES + 1) * self.slot_dim, hidden_dim)

        # ── 原始融合特征投影（与旧版相同）─────────────────────────────
        self.feature_proj = nn.Linear(llama_dim, hidden_dim)

        # ── 三种来源保持独立身份；先投影再加类型嵌入 ────────────────
        self.mediator_token_projs = nn.ModuleList([
            nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim))
            for _ in range(3)
        ])
        self.mediator_type_embedding = nn.Embedding(3, hidden_dim)
        nn.init.normal_(self.mediator_type_embedding.weight, std=0.02)

        # 参考 Query 缓存只由训练患者构建，不能被模型 checkpoint 当作参数存储。
        self.register_buffer(
            "reference_queries", torch.empty(0, hidden_dim), persistent=False
        )
        self.reference_lambda = float(getattr(train_config, "reference_lambda", 0.0))
        self.reference_chunk_size = max(1, int(getattr(train_config, "reference_chunk_size", 8)))
        self.mediator_token_mode = str(
            getattr(train_config, "mediator_token_mode", "three")
        ).lower()
        self.reference_aggregation = str(
            getattr(train_config, "reference_aggregation", "logit_mean")
        ).lower()
        if self.semantic_readout_scope != "all_tokens" and self.mediator_token_mode != "three":
            raise ValueError("after_image semantic readout is defined for three final tokens")
        if self.mediator_token_mode not in {"single", "three", "pooled_three"}:
            raise ValueError("mediator_token_mode must be 'single', 'three', or 'pooled_three'")
        if self.reference_aggregation not in {"feature_mean", "logit_mean"}:
            raise ValueError(
                "reference_aggregation must be 'feature_mean' or 'logit_mean'"
            )

        # ── 保留原始两层 Cross-Attention + 残差 + LayerNorm ─────────
        front_door_type = str(getattr(train_config, "front_door_type", "legacy")).lower()
        if front_door_type not in ("legacy", "torch", "mha"):
            raise ValueError(
                "refnwgm_v5 requires the original two-layer Cross-Attention "
                "(front_door_type=legacy/torch/mha)"
            )
        self.front_door_attn = FrontDoorAttention(hidden_dim, num_heads)

        # ── CoT 语义中介：从 LLaMA prompt hidden states 中抽取 M_cot ─────
        cot_gate_init_bias = float(getattr(train_config, "cot_gate_init_bias", 6.0))
        self.cot_semantic_mediator = CoTSemanticMediator(
            llama_dim, hidden_dim, gate_init_bias=cot_gate_init_bias,
        )

        # ── 分类头（与旧版相同）──────────────────────────────────────
        self.classifier = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim // 2, 2),
        )

        # ── 【改动3】模块级中介解码器（使用MLP而非单层Linear）──────────
        # 每个模块用MLP解码自己对应的原始特征，增强表达能力
        # M4 时使用 self.feature_modules（随机分组）而非全局 FEATURE_MODULES
        self.module_decoders = nn.ModuleDict({
            name: nn.Sequential(
                nn.Linear(module_dim, module_dim * 2),
                nn.LayerNorm(module_dim * 2),
                nn.ReLU(),
                nn.Dropout(0.2),
                nn.Linear(module_dim * 2, len(indices)),
            )
            for name, indices in self.feature_modules.items()
        })

        # ── M7 ablation: 单一统一解码器（256 → 58）替代 8 个 module_decoders ──
        if self.ab_single_decoder:
            self.single_module_decoder = nn.Sequential(
                nn.Linear(module_dim * N_MODULES, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(), nn.Dropout(0.2),
                nn.Linear(hidden_dim, tabular_input_dim),
            )

        # ── M1/MM8 ablation: fair single tabular structural mediator ──
        # 仅将 8 个医学 M_k 退化为一个 M_tab_single；M_i/M_cot/front-door 保持主路径一致。
        if self.ab_degenerate:
            self.degenerate_tab_encoder = nn.Sequential(
                nn.Linear(tabular_input_dim, hidden_dim * 2),
                nn.LayerNorm(hidden_dim * 2), nn.ReLU(), nn.Dropout(0.1),
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(),
                nn.Linear(hidden_dim, hidden_dim),
            )
            # 保留旧层名用于加载旧 checkpoint；fair MM8 主路径不再使用 joint early fusion。
            self.degenerate_joint_proj = nn.Sequential(
                nn.Linear(hidden_dim * 2, hidden_dim),
                nn.LayerNorm(hidden_dim), nn.ReLU(),
            )
            self.degenerate_decoder = nn.Linear(hidden_dim, tabular_input_dim)

        # These compatibility layers have no path to the v5 loss. Exclude them
        # from the optimizer and the v5 checkpoint so trainable counts reflect
        # the network that actually predicts.
        unused = (
            self.visual_mediator_proj, self.dual_mediator_residual,
            self.dual_mediator_gate, self.dual_mediator_norm,
            self.cross_modal_attn, self.cross_modal_norm,
            self.module_out_projs, self.m_t_proj, self.slot_align,
            self.modular_encoder.aggregate_proj,
        )
        if self.mediator_token_mode != "single":
            unused += (
                self.struct_mediator_residual, self.struct_mediator_gate,
                self.struct_mediator_norm,
                self.cot_semantic_mediator.fusion_residual,
                self.cot_semantic_mediator.fusion_gate,
            )
            self.cot_semantic_mediator.residual_scale.requires_grad_(False)
        for module in unused:
            module.requires_grad_(False)
        # Construct this after every v5 layer so enabling the residual does
        # not consume RNG before shared modules are initialized. This makes
        # seeded legacy and image-residual ablations directly comparable.
        if self.query_mode == "image_residual":
            self.query_image_adapter = ImageResidualQuery(llama_dim, hidden_dim)

        # Add v19 layers after every original layer, without advancing the
        # trainer's RNG. Fully observed modules retain the original path.
        with torch.random.fork_rng(devices=[]):
            self.image_aux_classifier = nn.Linear(hidden_dim, 2)
        self.image_aux_classifier.requires_grad_(self.image_aux_weight > 0)
        with torch.random.fork_rng(devices=[]):
            def scorer(dim):
                layer = nn.Sequential(nn.Linear(dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))
                nn.init.zeros_(layer[-1].weight)
                nn.init.zeros_(layer[-1].bias)
                return layer
            self.tortuosity_unavailable_embedding = nn.Parameter(
                torch.zeros(module_dim), requires_grad=self.tortuosity_module_drop_prob > 0)
            self.image_view_scorer = scorer(hidden_dim)
            self.module_view_scorers = nn.ModuleDict({n: scorer(module_dim) for n in MODULE_NAMES})
            self.pure_visual_gate_logit = nn.Parameter(torch.tensor(float(np.log(0.1 / 0.9))),
                                                       requires_grad=self.pure_visual_path)

        with torch.random.fork_rng(devices=[]):
            self.module_missing_embeddings = nn.ModuleDict()
            self.module_confidence_gates = nn.ModuleDict()
            for name, indices in self.feature_modules.items():
                embedding = nn.Linear(len(indices), module_dim, bias=False)
                nn.init.zeros_(embedding.weight)
                self.module_missing_embeddings[name] = embedding
                gate = nn.Linear(2, 1)
                nn.init.zeros_(gate.weight)
                nn.init.constant_(gate.bias, -2.0)
                self.module_confidence_gates[name] = gate

        self.configure_v19()
        self.query_value_residual = bool(getattr(train_config, "query_value_residual", True))
        self.front_door_attn.query_value_residual = self.query_value_residual
        if not self.query_value_residual:
            self.front_door_attn.query_residual_logit.requires_grad_(False)
        # New branches are appended after the RM03 layers, preserving shared initialization.
        self.v20_mode = str(getattr(train_config, "comparison_mode", "full"))
        self.v20_unified = bool(getattr(train_config, "unified_vascular_encoder", False))
        self.v20_drop_image = bool(getattr(train_config, "disable_image_token", False))
        self.v20_drop_tabular = bool(getattr(train_config, "disable_tabular_token", False))
        if self.v20_mode not in {"full", "image_only", "direct_fusion"}:
            raise ValueError("Unsupported comparison_mode")
        if self.v20_mode != "full":
            if (self.learn_reference_lambda or self.reference_lambda != 0 or self.reference_cls_weight != 0
                    or self.alpha != 0 or self.delta != 0 or self.mediator_cls_weight != 0
                    or self.observed_cls_weight != 0 or self.image_aux_weight != 0
                    or not self.ab_disable_front_door):
                raise ValueError("Direct comparison requires CE-only, no reference/mediator supervision")
        with torch.random.fork_rng(devices=[]):
            if self.v20_mode == "direct_fusion":
                self.v20_direct_tab = nn.Sequential(
                    nn.Linear(tabular_input_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden_dim * 2, hidden_dim))
                self.v20_direct_fusion = nn.Sequential(
                    nn.Linear(hidden_dim * 2, hidden_dim), nn.LayerNorm(hidden_dim), nn.ReLU())
            if self.v20_unified:
                self.v20_unified_encoder = nn.Sequential(
                    nn.Linear(tabular_input_dim * 2, hidden_dim * 2), nn.LayerNorm(hidden_dim * 2),
                    nn.ReLU(), nn.Dropout(.1), nn.Linear(hidden_dim * 2, hidden_dim),
                    nn.LayerNorm(hidden_dim), nn.ReLU())
                self.v20_unified_norm = nn.LayerNorm(hidden_dim)
                self.v20_unified_decoder = nn.Linear(hidden_dim, tabular_input_dim)
                self.v20_unified_scorer = nn.Sequential(
                    nn.Linear(hidden_dim + 1, 32), nn.Tanh(), nn.Linear(32, 1))
                nn.init.zeros_(self.v20_unified_scorer[-1].weight)
                nn.init.zeros_(self.v20_unified_scorer[-1].bias)
        if self.v20_unified and self.delta != 0:
            raise ValueError("Cross-module penalty is undefined for unified encoding; use delta=0")

    def v26_settings(self):
        return {key: getattr(self.train_config, key, default) for key, default in (
            ("visual_global_source", "patch_mean"), ("semantic_readout_scope", "all_tokens"),
            ("query_value_residual", True),
            ("comparison_mode", "full"), ("unified_vascular_encoder", False),
            ("disable_image_token", False), ("disable_tabular_token", False),
            ("freeze_query_residual", False), ("ablation_disable_cot_mediator", False),
            ("ablation_disable_bidir_guidance", False), ("ablation_disable_tab_to_visual", False),
            ("ablation_disable_visual_to_tab", False), ("mediator_token_mode", "three"),
            ("reference_aggregation", "logit_mean"), ("query_residual_gate", False),
            ("learn_reference_lambda", False), ("reference_cls_weight", 0.),
            ("auxiliary_mode", "off"), ("mediator_cls_weight", 0.),
            ("observed_cls_weight", .1), ("image_aux_weight", 0.))}

    def _v20_token_mask(self, mask):
        mask = mask.clone()
        mask[:, 0] &= not self.v20_drop_image
        mask[:, 1] &= not self.v20_drop_tabular
        if not mask.any(dim=1).all():
            raise ValueError("Every patient needs at least one unmasked mediator token")
        return mask

    def _v20_build_unified(self, flat_tab, z_v_loc, z_v_glob, observed, drop):
        # ONE vascular token enters both interaction directions; eight-module encoder is bypassed.
        obs = observed.to(dtype=flat_tab.dtype).clone()
        values = flat_tab.clone()
        if drop is not None:
            cols = self.feature_modules["tortuosity"]
            available = (~drop.bool()).to(values.dtype).unsqueeze(-1)
            values[:, cols] = values[:, cols] * available
            obs[:, cols] = obs[:, cols] * available
        tab = self.v20_unified_encoder(torch.cat([values, obs], dim=-1))
        il = self.visual_local_mapper(z_v_loc)
        ig = self.visual_global_mapper(z_v_glob)
        visual = torch.stack([ig, il], dim=1)
        pure = self.visual_base_proj(torch.cat([ig, il], dim=-1))
        if self.ab_disable_bidir_guidance or self.ab_disable_tab_to_visual:
            delta_v = torch.zeros_like(visual)
            enhanced_v = visual
        else:
            delta_v, _ = self.tab_to_visual_attn(visual, tab[:, None], tab[:, None])
            enhanced_v = self.cross_guided_norm_v(visual + delta_v)
        if not (self.ab_disable_bidir_guidance or self.ab_disable_visual_to_tab):
            delta_t, _ = self.visual_to_tab_attn(tab[:, None], visual, visual)
            tab = self.v20_unified_norm(tab + delta_t[:, 0])
        ig, il = enhanced_v[:, 0], enhanced_v[:, 1]
        fusion = torch.cat([ig, il, delta_v.mean(dim=1)], dim=-1)
        image = self.visual_mediator_norm(self.visual_base_proj(torch.cat([ig, il], dim=-1))
                    + self.visual_mediator_gate(fusion).sigmoid() * self.visual_mediator_integrator(fusion))
        if self.pure_visual_path:
            image = self.visual_mediator_norm(pure + self.pure_visual_gate_logit.sigmoid() * (image - pure))
        # Chunks preserve diagnostics tensor shapes only; they are not module encoders.
        chunks = tab.chunk(len(MODULE_NAMES), dim=-1)
        return dict(zip(MODULE_NAMES, chunks)), image, pure


    def effective_reference_lambda(self):
        if not self.learn_reference_lambda:
            return self.reference_lambda
        value = self.reference_lambda_logit.sigmoid()
        return value if self.lambda_learning_active else value.detach()

    def configure_v19(self):
        cfg = self.train_config
        self.learn_reference_lambda = bool(getattr(cfg, "learn_reference_lambda", False))
        self.lambda_learning_active = False
        self.reference_cls_weight = float(getattr(cfg, "reference_cls_weight", 0.0))
        initial = float(getattr(cfg, "reference_lambda", 0.25))
        if not 0 <= initial <= 1 or (self.learn_reference_lambda and not 0 < initial < 1):
            raise ValueError("Invalid initial lambda")
        if self.reference_cls_weight < 0:
            raise ValueError("reference_cls_weight must be nonnegative")
        clipped = min(1-1e-6, max(1e-6, initial))
        self.reference_lambda_logit = nn.Parameter(
            torch.tensor(float(np.log(clipped/(1-clipped)))), requires_grad=self.learn_reference_lambda)
        gated = bool(getattr(cfg, "query_residual_gate", False))
        rho = float(getattr(cfg, "query_residual_init", 0.5))
        if not 0 < rho < 1:
            raise ValueError("query_residual_init must be in (0,1)")
        self.front_door_attn.query_gate_enabled = gated
        self.front_door_attn.query_residual_logit.data.fill_(float(np.log(rho/(1-rho))))
        self.front_door_attn.query_residual_logit.requires_grad_(
            gated and not bool(getattr(cfg, "freeze_query_residual", False)))

    def move_auxiliary_modules(self, device):
        modules = [
            self.multi_modal_projector,
            self.local_vascular_attn,
            self.visual_mediator_proj,
            self.modular_encoder,
            self.module_missing_embeddings,
            self.module_confidence_gates,
            self.image_aux_classifier,
            self.image_view_scorer,
            self.module_view_scorers,
            self.visual_local_mapper,
            self.visual_global_mapper,
            self.tab_module_to_hidden,
            self.hidden_to_tab_module,
            self.tab_to_visual_attn,
            self.visual_to_tab_attn,
            self.cross_guided_norm_v,
            self.cross_guided_norm_q,
            self.visual_base_proj,
            self.visual_mediator_integrator,
            self.visual_mediator_gate,
            self.visual_mediator_norm,
            self.tabular_module_gates,
            self.tabular_module_norms,
            self.dual_mediator_residual,
            self.dual_mediator_gate,
            self.dual_mediator_norm,
            self.cross_modal_attn,
            self.cross_modal_norm,
            self.module_mediator_projs,
            self.module_aggregate,
            self.struct_mediator_residual,
            self.struct_mediator_gate,
            self.struct_mediator_norm,
            self.m_t_proj,
            self.slot_align,
            self.feature_proj,
            self.mediator_token_projs,
            self.mediator_type_embedding,
            self.front_door_attn,
            self.cot_semantic_mediator,
            self.classifier,
        ]
        modules.extend(module for name, module in self.named_children() if name.startswith("v20_"))
        for m in modules:
            m.to(device)
        self.reference_lambda_logit.data = self.reference_lambda_logit.data.to(device)
        self.pure_visual_gate_logit.data = self.pure_visual_gate_logit.data.to(device)
        self.tortuosity_unavailable_embedding.data = self.tortuosity_unavailable_embedding.data.to(device)
        if self.query_mode == "image_residual":
            self.query_image_adapter.to(device)
        self.module_mediator_projs.to(device)
        self.module_out_projs.to(device)
        self.module_decoders.to(device)
        if self.ab_single_decoder:
            self.single_module_decoder.to(device)
        if self.ab_degenerate:
            self.degenerate_tab_encoder.to(device)
            self.degenerate_joint_proj.to(device)
            self.degenerate_decoder.to(device)
        return self

    # ────────────────────────────────────────────────────────────────────────
    # 特征加载（与旧版完全相同）
    # ────────────────────────────────────────────────────────────────────────

    def _resolve_visual_features_dir(self):
        candidates = [self.visual_features_dir]

        for candidate in candidates:
            if (
                candidate
                and os.path.isdir(os.path.join(candidate, "pooler_output"))
                and os.path.isdir(os.path.join(candidate, "last_hidden_state"))
            ):
                if not self._visual_features_dir_logged:
                    logger.info("Resolved visual_features_dir=%s", candidate)
                    self._visual_features_dir_logged = True
                return candidate

        fallback = candidates[0] if candidates else ""
        if not self._visual_features_dir_logged:
            logger.warning(
                "No valid visual_features_dir found. Last tried path=%s",
                fallback or "<empty>",
            )
            self._visual_features_dir_logged = True
        return fallback

    @staticmethod
    def _load_npz_first_array(npz_path):
        tensors = np.load(npz_path)
        return tensors[tensors.files[0]]

    def _load_views_for_sample(self, visual_features_dir, sid):
        # Preserve the original suffix/v_idx slots. Compacting 0/2/3 into 0/1/2
        # would pair image view 2 with tabular view 1 in the Bi-MHA encoder.
        local_views = [None] * 4
        global_views = [None] * 4
        missing_view_keys = []

        for v_idx, suffix in enumerate(["_0", "_1", "_2", "_3"]):
            g_path = os.path.join(visual_features_dir, "pooler_output", f"{sid}{suffix}.npz")
            l_path = os.path.join(visual_features_dir, "last_hidden_state", f"{sid}{suffix}.npz")
            if os.path.exists(g_path) and os.path.exists(l_path):
                global_views[v_idx] = self._load_npz_first_array(g_path)
                local_views[v_idx] = self._load_npz_first_array(l_path)
            else:
                missing = []
                if not os.path.exists(g_path):
                    missing.append("pooler_output")
                if not os.path.exists(l_path):
                    missing.append("last_hidden_state")
                missing_view_keys.append((suffix, ",".join(missing)))

        if any(view is not None for view in local_views):
            return local_views, global_views, missing_view_keys, "per_view"

        g_path = os.path.join(visual_features_dir, "pooler_output", f"{sid}.npz")
        l_path = os.path.join(visual_features_dir, "last_hidden_state", f"{sid}.npz")
        if os.path.exists(g_path) and os.path.exists(l_path):
            global_np = self._load_npz_first_array(g_path)
            local_np = self._load_npz_first_array(l_path)

            if global_np.ndim == 1:
                global_np = global_np[None, :]
            if local_np.ndim == 2:
                local_np = local_np[None, :, :]

            num_views = min(global_np.shape[0], local_np.shape[0], 4)
            for view_idx in range(num_views):
                global_views[view_idx] = global_np[view_idx]
                local_views[view_idx] = local_np[view_idx]
            for suffix in ["_0", "_1", "_2", "_3"][num_views:]:
                missing_view_keys.append((suffix, "aggregate_file_missing_view"))
            return local_views, global_views, missing_view_keys, "aggregate"

        missing = []
        if not os.path.exists(g_path):
            missing.append("pooler_output")
        if not os.path.exists(l_path):
            missing.append("last_hidden_state")
        missing_view_keys.append(("", ",".join(missing)))
        return local_views, global_views, missing_view_keys, "missing"

    def summarize_image_feature_coverage(self, sample_ids, split_name="train", max_examples=5):
        visual_features_dir = self._resolve_visual_features_dir()
        per_sample_views = []
        example_missing_ids = []

        for sid in sample_ids:
            sid = int(sid)
            local_views, _, _, _ = self._load_views_for_sample(visual_features_dir, sid)
            views_found = sum(view is not None for view in local_views)
            per_sample_views.append(views_found)
            if views_found < 4 and len(example_missing_ids) < max_examples:
                example_missing_ids.append(sid)

        total_samples = len(per_sample_views)
        if total_samples == 0:
            return {
                "split_name": split_name,
                "full_samples": 0, "partial_samples": 0,
                "missing_samples": 0, "avg_views": 0.0,
                "example_missing_ids": [],
            }

        full_samples = sum(v == 4 for v in per_sample_views)
        missing_samples = sum(v == 0 for v in per_sample_views)
        partial_samples = total_samples - full_samples - missing_samples
        avg_views = float(sum(per_sample_views)) / total_samples
        return {
            "split_name": split_name,
            "full_samples": full_samples,
            "partial_samples": partial_samples,
            "missing_samples": missing_samples,
            "avg_views": avg_views,
            "example_missing_ids": example_missing_ids,
        }

    def _get_image_features_by_id(self, sample_id):
        B = len(sample_id)
        all_local = torch.zeros(B, 4, 257, 1024, device=device)
        all_global = torch.zeros(B, 4, 1024, device=device)
        view_mask = torch.zeros(B, 4, device=device)

        visual_features_dir = self._resolve_visual_features_dir()

        for i, sid in enumerate(sample_id):
            sid = int(sid)
            local_views, global_views, missing_view_keys, source_format = self._load_views_for_sample(
                visual_features_dir, sid
            )

            if any(view is not None for view in local_views):
                if (
                    source_format == "aggregate"
                    and sid not in self._missing_feature_logged_samples
                    and len(self._missing_feature_logged_samples) < self._missing_feature_log_limit
                ):
                    logger.warning(
                        "Image features loaded from aggregate file for sample_id=%s with %d views.",
                        sid, sum(view is not None for view in local_views),
                    )
                    self._missing_feature_logged_samples.add(sid)
                for v_idx, (g_np, l_np) in enumerate(zip(global_views[:4], local_views[:4])):
                    if g_np is None or l_np is None:
                        continue
                    all_global[i, v_idx] = torch.from_numpy(g_np)
                    all_local[i, v_idx] = torch.from_numpy(l_np)
                    view_mask[i, v_idx] = 1.0
                continue
            for v_idx, suffix in enumerate(["_0", "_1", "_2", "_3"]):
                g_path = os.path.join(visual_features_dir, "pooler_output", f"{sid}{suffix}.npz")
                l_path = os.path.join(visual_features_dir, "last_hidden_state", f"{sid}{suffix}.npz")

                if not (os.path.exists(g_path) and os.path.exists(l_path)):
                    if (
                        sid not in self._missing_feature_logged_samples
                        and len(self._missing_feature_logged_samples) < self._missing_feature_log_limit
                    ):
                        missing_parts = []
                        if not os.path.exists(g_path):
                            missing_parts.append("pooler_output")
                        if not os.path.exists(l_path):
                            missing_parts.append("last_hidden_state")
                        logger.warning(
                            "Missing image features for sample_id=%s, view=%s: %s",
                            sid, suffix, ",".join(missing_parts),
                        )
                        self._missing_feature_logged_samples.add(sid)
                    continue

                g_np = np.load(g_path)
                l_np = np.load(l_path)
                all_global[i, v_idx] = torch.from_numpy(g_np[g_np.files[0]])
                all_local[i, v_idx] = torch.from_numpy(l_np[l_np.files[0]])
                view_mask[i, v_idx] = 1.0

        return all_local, all_global, view_mask

    # ────────────────────────────────────────────────────────────────────────
    # image token 合并（与旧版完全相同）
    # ────────────────────────────────────────────────────────────────────────

    def _merge_input_ids_with_image_features(
        self, image_features, inputs_embeds, input_ids, attention_mask, labels,
        return_image_mask=False,
    ):
        num_images, num_image_patches, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        left_padding = not torch.sum(
            input_ids[:, -1] == torch.tensor(self.pad_token_id)
        )
        special_image_token_mask = input_ids == self.image_token_index
        num_special_image_tokens = torch.sum(special_image_token_mask, dim=-1)
        max_embed_dim = (
            num_special_image_tokens.max() * (num_image_patches - 1)
        ) + sequence_length
        batch_indices, non_image_indices = torch.where(
            input_ids != self.image_token_index
        )
        new_token_positions = (
            torch.cumsum(
                (special_image_token_mask * (num_image_patches - 1) + 1), -1
            )
            - 1
        )
        nb_image_pad = max_embed_dim - 1 - new_token_positions[:, -1]
        if left_padding:
            new_token_positions += nb_image_pad[:, None]
        text_to_overwrite = new_token_positions[batch_indices, non_image_indices]

        final_embedding = torch.zeros(
            batch_size, max_embed_dim, embed_dim,
            dtype=inputs_embeds.dtype, device=inputs_embeds.device,
        )
        final_attention_mask = torch.zeros(
            batch_size, max_embed_dim,
            dtype=attention_mask.dtype, device=inputs_embeds.device,
        )
        if labels is not None:
            final_labels = torch.full(
                (batch_size, max_embed_dim), self.ignore_index,
                dtype=input_ids.dtype, device=input_ids.device,
            )
        target_device = inputs_embeds.device
        batch_indices = batch_indices.to(target_device)
        non_image_indices = non_image_indices.to(target_device)
        text_to_overwrite = text_to_overwrite.to(target_device)
        attention_mask = attention_mask.to(target_device)

        final_embedding[batch_indices, text_to_overwrite] = inputs_embeds[
            batch_indices, non_image_indices
        ]
        final_attention_mask[batch_indices, text_to_overwrite] = attention_mask[
            batch_indices, non_image_indices
        ]
        if labels is not None:
            final_labels[batch_indices, text_to_overwrite] = labels[
                batch_indices, non_image_indices
            ]

        image_to_overwrite = torch.all(final_embedding == 0, dim=-1)
        image_to_overwrite &= (
            image_to_overwrite.cumsum(-1) - 1
            >= nb_image_pad[:, None].to(target_device)
        )
        if image_to_overwrite.sum() != image_features.shape[:-1].numel():
            raise ValueError(
                f"image token count mismatch: mask={torch.sum(special_image_token_mask)}, "
                f"features={num_images}"
            )
        final_embedding[image_to_overwrite] = (
            image_features.contiguous().reshape(-1, embed_dim).to(target_device)
        )
        final_attention_mask |= image_to_overwrite
        position_ids = (final_attention_mask.cumsum(-1) - 1).masked_fill_(
            (final_attention_mask == 0), 1
        )

        batch_indices, pad_indices = torch.where(input_ids == self.pad_token_id)
        indices_to_mask = new_token_positions[batch_indices, pad_indices]
        final_embedding[batch_indices, indices_to_mask] = 0

        merged = (
            final_embedding,
            final_attention_mask,
            final_labels if labels is not None else None,
            position_ids,
        )
        # image_to_overwrite marks the actual inserted image positions after
        # padding and expansion. A prompt string index is not sufficient.
        if return_image_mask:
            return (*merged, image_to_overwrite)
        return merged

    # ────────────────────────────────────────────────────────────────────────
    # 批内相关性正则项；这不是严格的互信息估计。
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _corr_loss(z1, z2):
        z1 = (z1 - z1.mean(0)) / (z1.std(0, unbiased=False) + 1e-6)
        z2 = (z2 - z2.mean(0)) / (z2.std(0, unbiased=False) + 1e-6)
        corr = torch.mm(z1.T, z2) / max(z1.shape[0], 1)
        # 除以矩阵元素数，使输出与 hidden_dim 无关，稳定在 [0, 1] 量级
        D = z1.shape[1]
        return torch.norm(corr, p="fro") / D

    def _token_corr_loss(self, tokens, token_mask, query):
        """Average the original correlation penalty over valid mediator tokens.

        This keeps the three-token training path free of a pooled mediator.
        """
        losses = []
        for index in range(tokens.size(1)):
            valid = token_mask[:, index]
            if int(valid.sum().item()) >= 2:
                losses.append(self._corr_loss(tokens[valid, index], query[valid]))
        if not losses:
            return query.new_zeros(())
        return torch.stack(losses).mean()

    # ────────────────────────────────────────────────────────────────────────
    # 模块间正交约束（新增）
    # ────────────────────────────────────────────────────────────────────────

    @staticmethod
    def _module_independence_loss(module_reprs: Dict[str, torch.Tensor]) -> torch.Tensor:
        """
        鼓励不同模块表示之间跨样本不相关（Barlow Twins 风格）。

        对每对模块 (i, j)，在 batch 维度上计算它们的交叉相关矩阵，
        惩罚 off-diagonal 元素，使模块间携带的信息尽量不重叠。

        module_reprs: {name: [B, module_dim]}
        """
        reprs = [module_reprs[n] for n in MODULE_NAMES]
        B = reprs[0].shape[0]
        if B < 2:
            return torch.tensor(0.0, device=reprs[0].device)

        # 每个模块在 batch 维度上做 z-score 标准化：[B, D]
        normed = []
        for r in reprs:
            r = r - r.mean(dim=0, keepdim=True)
            r = r / (r.std(dim=0, unbiased=False, keepdim=True) + 1e-6)
            normed.append(r)

        # 计算所有模块对的交叉相关矩阵，惩罚 off-diagonal
        loss = torch.tensor(0.0, device=reprs[0].device)
        n_pairs = 0
        for i in range(len(normed)):
            for j in range(i + 1, len(normed)):
                # cross_corr: [D_i, D_j]，每个元素是特征维度的跨样本相关
                cross_corr = torch.mm(normed[i].T, normed[j]) / B
                loss = loss + cross_corr.pow(2).mean()
                n_pairs += 1

        return loss / max(n_pairs, 1)

    # ────────────────────────────────────────────────────────────────────────
    # 核心中介构造（供主预测和模型级干预共用）
    # ────────────────────────────────────────────────────────────────────────

    def _enhance_tab_module(self, name, tab_module, visual_context_module):
        """Use image evidence to calibrate one tabular pathology module."""
        gate_input = torch.cat([tab_module, visual_context_module], dim=-1)
        gate = torch.sigmoid(self.tabular_module_gates[name](gate_input))
        return self.tabular_module_norms[name](
            tab_module + gate * visual_context_module
        )

    def _fuse_dual_mediators(self, m_i, m_q):
        """Fuse visual-led mediator M_i and tabular-led mediator M_q into final M."""
        dual_input = torch.cat([m_i, m_q, m_i * m_q], dim=-1)
        dual_gate = torch.sigmoid(self.dual_mediator_gate(dual_input))
        dual_residual = self.dual_mediator_residual(dual_input)
        return self.dual_mediator_norm(
            dual_gate * m_i + (1.0 - dual_gate) * m_q + dual_residual
        )

    def _fuse_struct_mediators(self, m_tab, m_i):
        """Fuse tabular modular mediator M_tab and visual mediator M_i into M_struct."""
        fusion_input = torch.cat([m_tab, m_i, m_tab * m_i], dim=-1)
        gate = torch.sigmoid(self.struct_mediator_gate(fusion_input))
        residual = self.struct_mediator_residual(fusion_input)
        return self.struct_mediator_norm(
            gate * m_tab + (1.0 - gate) * m_i + residual
        )

    @staticmethod
    def _view_weights(values, coverage, valid_mask, scorer):
        """Missing views have exactly zero mass. Scores start at uniform weights.

        Image scoring uses only pure visual values and image availability;
        vessel scoring additionally uses that module's measurement coverage.
        """
        inputs = torch.cat([values, coverage.to(values.dtype).unsqueeze(-1)], dim=-1)
        scores = scorer(inputs).squeeze(-1).float()
        mask = valid_mask.bool()
        weights = torch.softmax(scores.masked_fill(~mask, -1e4), dim=1)
        weights = weights * mask.float()
        return (weights / weights.sum(dim=1, keepdim=True).clamp_min(1e-12)).to(values.dtype)

    def _build_modular_mediators(self, flat_tab, z_v_loc, z_v_glob,
                                 flat_observed=None, flat_view_fraction=None, flat_tortuosity_drop=None):
        """
        先做视觉/表格双向交互，再构造 8 个按表格指标分组的模块表示。

        Args:
            flat_tab: [B*V, 58] 原始表格特征
            z_v_loc: [B*V, vit_dim] 局部视觉表征
            z_v_glob: [B*V, vit_dim] 全局视觉表征

        Returns:
            med_modules: {name: [B*V, module_dim]} 表格医学模块中介
            m_i: [B*V, hidden_dim] 视觉中介
        """
        if self.v20_unified:
            if flat_observed is None:
                flat_observed = torch.ones_like(flat_tab, dtype=torch.bool)
            return self._v20_build_unified(flat_tab, z_v_loc, z_v_glob, flat_observed, flat_tortuosity_drop)
        if self.v20_mode != "full":
            # Neither comparison consumes vascular encoders, Bi-MHA, or their supervision.
            zeros = flat_tab.new_zeros(flat_tab.size(0), self.hidden_dim)
            chunks = zeros.chunk(len(MODULE_NAMES), dim=-1)
            return dict(zip(MODULE_NAMES, chunks)), zeros, zeros
        # 1) 表格模块特征映射；旧聚合 f_t 不再为 v5 计算。
        _, tab_modules = self.modular_encoder(flat_tab, return_aggregate=False)
        if flat_observed is None:
            flat_observed = torch.ones_like(flat_tab, dtype=torch.bool)
        if flat_view_fraction is None:
            flat_view_fraction = flat_tab.new_ones(flat_tab.size(0), 1)
        module_confidence = {}
        for name, indices in self.feature_modules.items():
            observed = flat_observed[:, indices].to(dtype=tab_modules[name].dtype)
            fraction = observed.mean(dim=-1, keepdim=True)
            context = torch.cat([fraction, flat_view_fraction.to(fraction)], dim=-1)
            confidence = fraction + (1.0 - fraction) * torch.sigmoid(
                self.module_confidence_gates[name](context))
            tab_modules[name] = (
                tab_modules[name]
                + self.module_missing_embeddings[name](1.0 - observed)
            )
            # Apply reliability once, after interaction and mediator projection.
            module_confidence[name] = confidence

        # Artificial module unavailability is distinct from natural missing cells.
        # Replace the complete encoded module before either Bi-MHA direction.
        if flat_tortuosity_drop is not None:
            drop = flat_tortuosity_drop.reshape(-1, 1).to(device=flat_tab.device, dtype=torch.bool)
            token = self.tortuosity_unavailable_embedding.to(tab_modules["tortuosity"])
            tab_modules["tortuosity"] = torch.where(drop, token.unsqueeze(0), tab_modules["tortuosity"])
            module_confidence["tortuosity"] = torch.where(
                drop, torch.ones_like(module_confidence["tortuosity"]), module_confidence["tortuosity"])

        # 2) 视觉特征映射：F_i -> f_il, f_ig
        f_il = self.visual_local_mapper(z_v_loc)
        f_ig = self.visual_global_mapper(z_v_glob)
        pure_visual = self.visual_base_proj(torch.cat([f_ig, f_il], dim=-1))
        visual_tokens = torch.stack([f_ig, f_il], dim=1)  # [B*V, 2, hidden_dim]

        tab_tokens = torch.stack(
            [self.tab_module_to_hidden[n](tab_modules[n]) for n in MODULE_NAMES],
            dim=1,
        )  # [B*V, N_MODULES, hidden_dim]

        # 3) 双向跨模态引导。消融时只旁路对应方向，不改模块维度。
        disable_t2v = (
            self.ab_disable_bidir_guidance or self.ab_disable_tab_to_visual
            or self.ab_mediator_source in {"visual_only", "tabular_only"}
        )
        disable_v2t = (
            self.ab_disable_bidir_guidance
            or self.ab_disable_visual_to_tab
            or self.ab_mediator_source in {"tabular_only", "visual_only"}
        )
        if disable_t2v:
            tab_to_visual_tokens = torch.zeros_like(visual_tokens)
            visual_tokens_enh = visual_tokens
        else:
            tab_to_visual_tokens, _ = self.tab_to_visual_attn(
                visual_tokens, tab_tokens, tab_tokens
            )
            visual_tokens_enh = self.cross_guided_norm_v(
                visual_tokens + tab_to_visual_tokens
            )
        f_ig_enh = visual_tokens_enh[:, 0, :]
        f_il_enh = visual_tokens_enh[:, 1, :]
        f_q_to_v = tab_to_visual_tokens.mean(dim=1)

        if disable_v2t:
            visual_to_tab_modules = {
                name: torch.zeros_like(tab_modules[name]) for name in MODULE_NAMES
            }
        else:
            visual_to_tab_tokens, _ = self.visual_to_tab_attn(
                tab_tokens, visual_tokens, visual_tokens
            )
            visual_to_tab_tokens = self.cross_guided_norm_q(
                tab_tokens + visual_to_tab_tokens
            )
            visual_to_tab_modules = {
                name: self.hidden_to_tab_module[name](visual_to_tab_tokens[:, i, :])
                for i, name in enumerate(MODULE_NAMES)
            }

        # 4) 表格引导的视觉中介 M_i
        visual_pair = torch.cat([f_ig_enh, f_il_enh], dim=-1)
        visual_base = self.visual_base_proj(visual_pair)
        visual_fusion_input = torch.cat([f_ig_enh, f_il_enh, f_q_to_v], dim=-1)
        visual_delta = self.visual_mediator_integrator(visual_fusion_input)
        visual_gate = torch.sigmoid(self.visual_mediator_gate(visual_fusion_input))
        m_i = self.visual_mediator_norm(visual_base + visual_gate * visual_delta)

        if self.pure_visual_path:
            gate = torch.sigmoid(self.pure_visual_gate_logit)
            # The pure term has no table dependencies. Cross-modal evidence is gated.
            m_i = self.visual_mediator_norm(pure_visual + gate * (m_i - pure_visual))

        # 5) 表格模块中介 M_k：只由对应医学表格模块生成，避免把 M_i 重复注入每个 M_k。
        tab_modules_enh = {}
        med_modules = {}
        for name in MODULE_NAMES:
            if disable_v2t:
                tab_modules_enh[name] = tab_modules[name]
            else:
                tab_modules_enh[name] = self._enhance_tab_module(
                    name, tab_modules[name], visual_to_tab_modules[name]
                )
            if self.ab_disable_module_mediator_proj:
                med_modules[name] = tab_modules_enh[name]
            else:
                med_modules[name] = self.module_mediator_projs[name](tab_modules_enh[name])
            if self.raw_tab_residual:
                raw = (tab_modules[name] if self.ab_disable_module_mediator_proj
                       else self.module_mediator_projs[name](tab_modules[name]))
                med_modules[name] = med_modules[name] + self.raw_tab_residual * raw
            med_modules[name] = med_modules[name] * module_confidence[name]

        # Patient-level pooling happens below. No legacy view-level fusion or
        # unused slot projections are evaluated in the three-token path.
        return med_modules, m_i, pure_visual

    # ────────────────────────────────────────────────────────────────────────
    # 前向传播
    # ────────────────────────────────────────────────────────────────────────

    def set_reference_lambda(self, value: float):
        """Set the fusion weight used for fixed-parameter inference."""
        value = float(value)
        if not np.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError("reference_lambda must be a finite value in [0, 1]")
        self.reference_lambda = value
        return self

    def set_reference_features(self, reference_queries: torch.Tensor):
        """Cache detached [J,D] queries encoded only from training patients."""
        refs = torch.as_tensor(reference_queries)
        if refs.ndim != 2 or refs.size(0) < 1 or refs.size(1) != self.hidden_dim:
            raise ValueError(
                f"reference_queries must have shape [J,{self.hidden_dim}], J>=1"
            )
        if not torch.isfinite(refs).all():
            raise ValueError("reference_queries contains non-finite values")
        target = self.feature_proj.weight
        self.reference_queries = refs.detach().clone().to(
            device=target.device, dtype=target.dtype
        )
        return self

    def clear_reference_features(self):
        self.reference_queries = self.reference_queries.new_empty(0, self.hidden_dim)
        return self

    def encode_patient_state(
        self,
        sample_id: torch.IntTensor,
        tabular_features: torch.FloatTensor,
        tabular_observed_mask: Optional[torch.BoolTensor] = None,
        text_prompt: Optional[str] = None,
        image_features_override=None,
        fixed_language_state: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Encode image, 58 features and prompt into f plus 3 named mediators.

        M token order: image-dominant, eight-module vascular aggregate, semantic.
        The image and vascular mediators already interact through original Bi-MHA,
        so these token labels identify their construction paths, not pure sources.
        fixed_language_state may be reused only if image features and prompt are
        identical and the change is restricted to tabular inputs.
        image_features_override is (img_local, img_global, v_mask) with the exact
        shapes returned by _get_image_features_by_id.
        """
        if tabular_features is None or tabular_features.ndim != 3:
            raise ValueError("tabular_features must have shape [B,V,58]")
        aux_device = self.feature_proj.weight.device
        aux_dtype = self.feature_proj.weight.dtype
        tabular_features = tabular_features.to(device=aux_device, dtype=aux_dtype)
        B, V, T_dim = tabular_features.shape
        if tabular_observed_mask is None:
            raise ValueError("v19 prediction requires tabular_observed_mask")
        if tabular_observed_mask.shape != tabular_features.shape:
            raise ValueError("tabular_observed_mask must match tabular_features [B,V,58]")
        tabular_observed_mask = tabular_observed_mask.to(
            device=aux_device, dtype=torch.bool)
        if T_dim != 58:
            raise ValueError(f"expected 58 vascular features, got {T_dim}")
        if image_features_override is None and fixed_language_state is not None \
                and "image_features" in fixed_language_state:
            # Reuse the same image evidence for named tabular interventions.
            img_local, img_global, v_mask = fixed_language_state["image_features"]
        elif image_features_override is None:
            if sample_id is None:
                raise ValueError("sample_id is required without image_features_override")
            img_local, img_global, v_mask = self._get_image_features_by_id(sample_id)
        else:
            if not isinstance(image_features_override, (tuple, list)) or len(image_features_override) != 3:
                raise ValueError("image_features_override must be (img_local,img_global,v_mask)")
            img_local, img_global, v_mask = image_features_override
        img_local = torch.as_tensor(img_local, device=aux_device, dtype=aux_dtype)
        img_global = torch.as_tensor(img_global, device=aux_device, dtype=aux_dtype)
        v_mask = torch.as_tensor(v_mask, device=aux_device, dtype=aux_dtype)
        if img_local.shape != (B, V, 257, 1024):
            raise ValueError(f"img_local shape {tuple(img_local.shape)} != {(B,V,257,1024)}")
        if img_global.shape != (B, V, 1024):
            raise ValueError(f"img_global shape {tuple(img_global.shape)} != {(B,V,1024)}")
        if v_mask.shape != (B, V):
            raise ValueError(f"v_mask shape {tuple(v_mask.shape)} != {(B,V)}")
        if fixed_language_state is not None and image_features_override is not None:
            # A changed image must be re-encoded by LLaMA, even if tabular values also change.
            raise ValueError("fixed_language_state cannot be used with image_features_override")

        flat_local = img_local.reshape(B * V, 257, 1024)
        flat_tab = tabular_features.reshape(B * V, T_dim)
        flat_observed = tabular_observed_mask.reshape(B * V, T_dim)
        view_fraction = (v_mask.float().mean(dim=1, keepdim=True)
                         .expand(B, V).reshape(B * V, 1))
        if fixed_language_state is not None and "visual_cache" in fixed_language_state:
            if fixed_language_state.get("visual_global_source", "patch_mean") != self.visual_global_source:
                raise ValueError("Cached language state uses another visual global source")
            z_v_loc, z_v_glob, alpha_map = fixed_language_state["visual_cache"]
        else:
            z_v_loc, alpha_map = self.local_vascular_attn(flat_local)
            z_v_glob = _visual_global_features(flat_local, img_global, self.visual_global_source)
        # Draw once per training patient, sharing the decision across all views.
        tortuosity_training_drop = torch.zeros(B, dtype=torch.bool, device=aux_device)
        if self.training and self.tortuosity_module_drop_prob > 0:
            tortuosity_training_drop = torch.rand(B, device=aux_device) < self.tortuosity_module_drop_prob
        flat_tortuosity_drop = tortuosity_training_drop[:, None].expand(B, V).reshape(-1)
        med_modules, m_i, pure_visual = self._build_modular_mediators(
            flat_tab, z_v_loc, z_v_glob, flat_observed, view_fraction, flat_tortuosity_drop
        )
        v_mask_exp = v_mask.unsqueeze(-1)
        view_count = v_mask.sum(dim=1, keepdim=True).clamp_min(1.0)
        m_i_reshaped = m_i.reshape(B, V, -1)
        pure_visual_views = pure_visual.reshape(B, V, -1)
        image_view_weights = self._view_weights(
            pure_visual_views, v_mask, v_mask, self.image_view_scorer)
        pure_image_patient = (pure_visual_views * image_view_weights.unsqueeze(-1)).sum(dim=1)
        m_i_patient = (m_i_reshaped * image_view_weights.unsqueeze(-1)).sum(dim=1)
        med_modules_patient = {}
        module_view_weights = {}
        for name in MODULE_NAMES:
            mod_view = med_modules[name].reshape(B, V, -1)
            coverage = tabular_observed_mask[:, :, self.feature_modules[name]].float().mean(dim=-1)
            if name == "tortuosity":
                coverage = coverage.masked_fill(tortuosity_training_drop[:, None], 0.0)
            weights = self._view_weights(mod_view, coverage, v_mask, self.module_view_scorers[name])
            module_view_weights[name] = weights
            med_modules_patient[name] = (mod_view * weights.unsqueeze(-1)).sum(dim=1)
        if self.v20_unified:
            unified_views = torch.cat([med_modules[name] for name in MODULE_NAMES], dim=-1).reshape(B, V, -1)
            coverage = tabular_observed_mask.float().mean(dim=-1)
            weights = self._view_weights(unified_views, coverage, v_mask, self.v20_unified_scorer)
            m_tab_patient = (unified_views * weights.unsqueeze(-1)).sum(dim=1)
        elif self.v20_mode != "full":
            m_tab_patient = tabular_features.new_zeros(B, self.hidden_dim)
        else:
            m_tab_patient = self.module_aggregate(
                torch.cat([med_modules_patient[name] for name in MODULE_NAMES], dim=-1)
            )
        m_tab_single_view = None
        if self.ab_degenerate:
            m_tab_single_view = self.degenerate_tab_encoder(flat_tab)
            m_tab_patient = (
                m_tab_single_view.reshape(B, V, -1) * v_mask_exp
            ).sum(dim=1) / view_count

        if fixed_language_state is None:
            if text_prompt is None:
                text_prompt = getattr(
                    self.train_config,
                    "default_text_prompt",
                    "[Img]<|image_feature|>[/Img][MRG]Analyze step by step the "
                    "fundus vascular structure, perfusion, arterial and venous "
                    "caliber, optic disc cues, and infer stroke risk.",
                )
            avg_img_global = (img_global * v_mask_exp).sum(dim=1) / view_count
            selected = avg_img_global.reshape(-1, 1, 1024)
            image_features_proj = self.multi_modal_projector(selected)
            prompt_ids = self.tokenizer.encode(
                self.tokenizer.bos_token + text_prompt, add_special_tokens=False
            )
            input_ids = torch.tensor(
                [prompt_ids] * B, device=image_features_proj.device
            )
            attention_mask = torch.ones_like(input_ids)
            inputs_embeds = self.language_model.get_input_embeddings()(input_ids)
            inputs_embeds, attention_mask, _, _, image_token_mask = (
                self._merge_input_ids_with_image_features(
                    image_features_proj, inputs_embeds, input_ids, attention_mask, None,
                    return_image_mask=True,
                )
            )
            llama_out = self.language_model(
                attention_mask=attention_mask,
                inputs_embeds=inputs_embeds,
                use_cache=False,
                output_hidden_states=True,
                return_dict=True,
            )
            last_hidden = llama_out.hidden_states[-1]
            mask = attention_mask.unsqueeze(-1).float()
            f_global = (last_hidden * mask).sum(dim=1) / mask.sum(dim=1)
            f = self.feature_proj(
                f_global.to(device=aux_device, dtype=aux_dtype)
            )
            if self.query_mode == "image_residual":
                # The visual projector contributes exactly one token per patient.
                # Read its *merged* position in the causal LLaMA hidden states,
                # rather than averaging fixed prompt words as visual evidence.
                image_counts = image_token_mask.sum(dim=1)
                if not torch.all(image_counts == 1):
                    raise ValueError(
                        "image_residual query expects exactly one projected "
                        "image token per patient"
                    )
                image_hidden = (
                    last_hidden * image_token_mask.to(last_hidden.device).unsqueeze(-1)
                ).sum(dim=1)
                f = self.query_image_adapter(f, image_hidden)
            semantic_mask = _semantic_readout_mask(attention_mask, image_token_mask, self.semantic_readout_scope)
            language_cache = {
                "semantic_attention_mask": semantic_mask,
                "semantic_readout_scope": self.semantic_readout_scope,
                "visual_global_source": self.visual_global_source,
                "last_hidden": last_hidden,
                "attention_mask": attention_mask,
                "f": f,
                "image_features": (img_local, img_global, v_mask),
                "visual_cache": (z_v_loc, z_v_glob, alpha_map),
            }
        else:
            language_cache = fixed_language_state
            last_hidden = language_cache["last_hidden"]
            attention_mask = language_cache["attention_mask"]
            if language_cache.get("semantic_readout_scope", "all_tokens") != self.semantic_readout_scope:
                raise ValueError("Cached language state uses another semantic scope")
            semantic_mask = language_cache.get("semantic_attention_mask", attention_mask)
            f = language_cache["f"]
            if f.shape != (B, self.hidden_dim) or last_hidden.size(0) != B:
                raise ValueError("fixed_language_state batch does not match current patient")
            f = f.to(device=aux_device, dtype=aux_dtype)

        if self.v20_mode == "direct_fusion":
            direct_views = self.v20_direct_tab(torch.cat(
                [tabular_features, tabular_observed_mask.to(tabular_features)], dim=-1))
            direct_tab = (direct_views * v_mask_exp).sum(dim=1) / view_count
            f = self.v20_direct_fusion(torch.cat([f, direct_tab], dim=-1))
        m_struct = None
        m_patient = None
        if self.mediator_token_mode == "single":
            # Controlled ablation: the original cotstrong fused mediator is one K/V.
            if self.ab_mediator_source == "visual_only":
                m_struct = m_i_patient
            elif self.ab_mediator_source == "tabular_only":
                m_struct = m_tab_patient
            else:
                m_struct = self._fuse_struct_mediators(m_tab_patient, m_i_patient)
            if self.ab_disable_cot_mediator:
                m_patient = m_struct
                m_cot = torch.zeros_like(m_struct)
            else:
                m_patient, m_cot = self.cot_semantic_mediator(
                    last_hidden, attention_mask, m_struct.to(last_hidden.device)
                )
            M = m_patient.to(device=aux_device, dtype=aux_dtype).unsqueeze(1)
            token_mask = torch.ones(B, 1, device=aux_device, dtype=torch.bool)
            regularizer_tokens, regularizer_mask = M, token_mask
        else:
            # No image/table/semantic collapse before the prediction head.
            if self.ab_disable_cot_mediator:
                m_cot = torch.zeros_like(m_tab_patient)
            else:
                m_cot = self.cot_semantic_mediator.extract_semantic(
                    last_hidden, semantic_mask
                )
            token_inputs = (
                m_i_patient.to(aux_device),
                m_tab_patient.to(aux_device),
                m_cot.to(aux_device),
            )
            token_mask = torch.ones(B, 3, device=aux_device, dtype=torch.bool)
            token_mask[:, 0] = (v_mask.sum(dim=1) > 0) & (
                self.ab_mediator_source != "tabular_only"
            )
            token_mask[:, 1] = self.ab_mediator_source != "visual_only"
            token_mask[:, 2] = not self.ab_disable_cot_mediator
            token_mask = self._v20_token_mask(token_mask)
            M = torch.stack(
                [
                    self.mediator_token_projs[k](token_inputs[k].to(dtype=aux_dtype))
                    for k in range(3)
                ],
                dim=1,
            )
            M = M + self.mediator_type_embedding.weight.to(dtype=M.dtype).unsqueeze(0)
            M = M.masked_fill(~token_mask.unsqueeze(-1), 0.0)
            regularizer_tokens, regularizer_mask = M, token_mask
            if self.mediator_token_mode == "pooled_three":
                # Controlled token-count ablation: all upstream three-token
                # encoders and regularizers stay identical; only K/V is pooled.
                M = (M * token_mask.unsqueeze(-1)).sum(dim=1, keepdim=True) / (
                    token_mask.sum(dim=1, keepdim=True).clamp_min(1).unsqueeze(-1))
                token_mask = torch.ones(B, 1, device=aux_device, dtype=torch.bool)
        return {
            "f": f,
            "M": M,
            "token_mask": token_mask,
            "regularizer_tokens": regularizer_tokens,
            "regularizer_mask": regularizer_mask,
            "language_cache": language_cache,
            "m_i_patient": m_i_patient,
            "tortuosity_training_drop": tortuosity_training_drop,
            "pure_image_patient": pure_image_patient,
            "pure_visual_views": pure_visual_views,
            "image_view_weights": image_view_weights,
            "module_view_weights": module_view_weights,
            "m_tab_patient": m_tab_patient,
            "m_cot": m_cot,
            "m_patient": m_patient,
            "m_struct": m_struct,
            "m_i_reshaped": m_i_reshaped,
            "med_modules": med_modules,
            "med_modules_patient": med_modules_patient,
            "m_tab_single_view": m_tab_single_view,
            "flat_tab": flat_tab,
            "v_mask": v_mask,
            "tabular_observed_mask": tabular_observed_mask,
            "module_observed_fraction": {
                name: (tabular_observed_mask[:, :, indices].float()
                       * v_mask.unsqueeze(-1)).sum(dim=(1, 2))
                      / (v_mask.sum(dim=1).clamp_min(1) * len(indices))
                for name, indices in self.feature_modules.items()
            },
            "alpha_map": alpha_map,
        }

    def predict_from_states(
        self,
        f: torch.Tensor,
        M: torch.Tensor,
        reference_queries: Optional[torch.Tensor] = None,
        reference_lambda: Optional[float] = None,
        token_mask: Optional[torch.Tensor] = None,
        return_attention: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Shared head for current and reference queries; logits are averaged first.

        reference_queries accepts [J,D] or [B,J,D] (patient-specific leave-one-out).
        logit_mean mode uses softmax((1-lambda)*z_cur + lambda*mean_j(z_ref_j)).
        feature_mean mode is an ablation using h(mean_j(ref_j),M) instead.
        """
        if f.ndim != 2 or f.size(1) != self.hidden_dim:
            raise ValueError("f must have shape [B,D]")
        if M.ndim != 3 or M.size(0) != f.size(0) or M.size(2) != self.hidden_dim:
            raise ValueError("M must have shape [B,T,D]")
        T = M.size(1)
        if T not in (1, 3):
            raise ValueError("the controlled mediator ablation supports T=1 or T=3")
        if token_mask is None:
            token_mask = torch.ones(
                f.size(0), T, device=M.device, dtype=torch.bool
            )
        else:
            token_mask = torch.as_tensor(
                token_mask, device=M.device, dtype=torch.bool
            )
            if token_mask.shape != (f.size(0), T):
                raise ValueError("token_mask must have shape [B,T]")
        if not token_mask.any(dim=1).all():
            raise ValueError("at least one mediator token must be valid per patient")
        lam = self.effective_reference_lambda() if reference_lambda is None else float(reference_lambda)
        lam_number = float(lam.detach()) if isinstance(lam, torch.Tensor) else lam
        if not np.isfinite(lam_number) or not 0.0 <= lam_number <= 1.0:
            raise ValueError("reference_lambda must be in [0,1]")
        if self.ab_disable_front_door:
            cur_hidden = f
            attn_cur = None
        else:
            cur_result = self.front_door_attn(
                f.unsqueeze(1), M, key_padding_mask=~token_mask,
                return_attention=return_attention,
            )
            if return_attention:
                cur_hidden, attn_cur = cur_result
            else:
                cur_hidden, attn_cur = cur_result, None
        logits_cur = self.classifier(cur_hidden)

        logits_ref = None
        reference_margin_std = None
        n_refs = 0
        refs = self.reference_queries if reference_queries is None else reference_queries
        if lam_number > 0.0:
            if refs is None or refs.numel() == 0:
                raise ValueError(
                    "reference_lambda>0 requires training-patient reference_queries"
                )
            refs = refs.detach().to(device=f.device, dtype=f.dtype)
            if refs.ndim == 2:
                if refs.size(1) != self.hidden_dim:
                    raise ValueError("reference_queries must have shape [J,D]")
                n_refs = refs.size(0)
                refs = refs.unsqueeze(0).expand(f.size(0), -1, -1)
            elif refs.ndim == 3:
                if refs.size(0) != f.size(0) or refs.size(2) != self.hidden_dim:
                    raise ValueError("reference_queries must have shape [B,J,D]")
                n_refs = refs.size(1)
            else:
                raise ValueError("reference_queries must be [J,D] or [B,J,D]")
            if n_refs < 1:
                raise ValueError("at least one reference patient is required")
            B = f.size(0)
            if self.reference_aggregation == "feature_mean":
                # Controlled ablation: nonlinear head is applied after averaging queries.
                ref_query = refs.mean(dim=1)
                if self.ab_disable_front_door:
                    ref_hidden = ref_query
                else:
                    ref_hidden = self.front_door_attn(
                        ref_query.unsqueeze(1), M, key_padding_mask=~token_mask,
                    )
                logits_ref = self.classifier(ref_hidden)
            else:
                ref_sum = torch.zeros_like(logits_cur)
                reference_margins = []
                for start in range(0, n_refs, self.reference_chunk_size):
                    chunk = refs[:, start:start + self.reference_chunk_size]
                    C = chunk.size(1)
                    ref_query = chunk.reshape(B * C, self.hidden_dim)
                    if self.ab_disable_front_door:
                        ref_hidden = ref_query
                    else:
                        ref_mediator = (
                            M.unsqueeze(1).expand(-1, C, -1, -1)
                            .reshape(B * C, T, self.hidden_dim)
                        )
                        ref_mask = (
                            token_mask.unsqueeze(1).expand(-1, C, -1)
                            .reshape(B * C, T)
                        )
                        ref_hidden = self.front_door_attn(
                            ref_query.unsqueeze(1), ref_mediator,
                            key_padding_mask=~ref_mask,
                        )
                    chunk_logits = self.classifier(ref_hidden).reshape(B, C, 2)
                    ref_sum = ref_sum + chunk_logits.sum(dim=1)
                    reference_margins.append(
                        (chunk_logits[..., 1] - chunk_logits[..., 0]).detach()
                    )
                logits_ref = ref_sum / n_refs
                reference_margin_std = torch.cat(reference_margins, dim=1).std(
                    dim=1, unbiased=False
                )
            logits = (1.0 - lam) * logits_cur + lam * logits_ref
        else:
            logits = logits_cur
        return {
            "logits": logits,
            "logits_cur": logits_cur,
            "logits_ref": logits_ref,
            "reference_margin_std": reference_margin_std,
            "probs": torch.softmax(logits, dim=-1),
            "F_fd": cur_hidden,
            "attention_cur": attn_cur,
            "n_refs": n_refs,
            "reference_lambda": lam,
            "reference_aggregation": self.reference_aggregation,
        }

    def forward(
        self,
        sample_id: torch.IntTensor = None,
        tabular_features: torch.FloatTensor = None,
        tabular_observed_mask: Optional[torch.BoolTensor] = None,
        labels: torch.LongTensor = None,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values=None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = True,
        text_prompt: Optional[str] = None,
        reference_queries: Optional[torch.Tensor] = None,
        reference_lambda: Optional[float] = None,
        image_features_override=None,
    ):
        state = self.encode_patient_state(
            sample_id, tabular_features, text_prompt=text_prompt,
            tabular_observed_mask=tabular_observed_mask,
            image_features_override=image_features_override,
        )
        prediction_mask = state["token_mask"]
        if self.training and self.tab_token_drop_prob > 0:
            if state["M"].size(1) != 3:
                raise ValueError("Tab token dropout requires three mediator tokens")
            prediction_mask = prediction_mask.clone()
            drop = torch.rand(prediction_mask.size(0), device=prediction_mask.device) < self.tab_token_drop_prob
            drop &= prediction_mask[:, [0, 2]].any(dim=1)
            prediction_mask[:, 1] &= ~drop
        pred = self.predict_from_states(
            state["f"], state["M"], reference_queries=reference_queries,
            reference_lambda=reference_lambda, token_mask=prediction_mask,
        )
        logits = pred["logits"]
        B = logits.size(0)
        loss = None
        loss_dict = {}
        if labels is not None:
            labels = labels.to(logits.device)
            class_weights = self.loss_class_weights
            if class_weights is None:
                class_counts = torch.bincount(labels, minlength=2).float()
                nonzero = class_counts > 0
                if torch.any(nonzero):
                    class_weights = torch.ones_like(class_counts)
                    class_weights[nonzero] = (
                        class_counts.sum() / (nonzero.sum() * class_counts[nonzero])
                    )
                else:
                    class_weights = None
            if class_weights is not None:
                class_weights = class_weights.to(logits.device)
            label_smoothing = float(
                getattr(self.train_config, "label_smoothing", 0.0) or 0.0
            )
            l_cls = F.cross_entropy(
                logits, labels, weight=class_weights,
                label_smoothing=label_smoothing,
            )
            loss_dict["L_cls"] = l_cls
            loss_dict["L_NWGM"] = l_cls
            l_obs = F.cross_entropy(
                pred["logits_cur"], labels, weight=class_weights,
                label_smoothing=label_smoothing,
            )
            loss_dict["L_obs"] = l_obs
            l_ref = logits.new_zeros(())
            if self.reference_cls_weight > 0:
                if pred["logits_ref"] is None:
                    raise RuntimeError("Reference supervision needs active train references")
                l_ref = F.cross_entropy(pred["logits_ref"], labels,
                                        weight=class_weights, label_smoothing=label_smoothing)
            loss_dict["L_ref"] = l_ref
            loss_dict["lambda_value"] = logits.new_tensor(float(
                self.effective_reference_lambda().detach()) if self.learn_reference_lambda
                else self.reference_lambda)
            effective_rho = (
                float(self.front_door_attn.query_residual_logit.sigmoid().detach())
                if self.front_door_attn.query_gate_enabled else 1.0
            ) if self.query_value_residual else 0.0
            loss_dict["query_residual_rho"] = logits.new_tensor(effective_rho)

            flat_tab = state["flat_tab"]
            med_modules = state["med_modules"]
            v_mask = state["v_mask"]
            observed = _reconstruction_observed_mask(
                tabular_features, tabular_observed_mask, v_mask
            )
            flat_observed = observed.reshape(-1, observed.size(-1)).to(flat_tab.device)

            def masked_reconstruction(prediction, target, mask):
                # Imputed entries are model inputs, not observed reconstruction targets.
                squared_error = F.mse_loss(prediction, target, reduction="none")
                mask = mask.to(device=squared_error.device, dtype=squared_error.dtype)
                return (squared_error * mask).sum() / mask.sum().clamp_min(1)

            l_med = logits.new_zeros(())
            module_losses = {}
            if self.alpha == 0:
                pass
            elif self.v20_unified:
                unified = torch.cat([med_modules[name] for name in MODULE_NAMES], dim=-1)
                pred_all = self.v20_unified_decoder(unified)
                # Match Full: sum of per-module observed-cell MSEs, not a 58-cell mean.
                for name, indices in self.feature_modules.items():
                    value = masked_reconstruction(pred_all[:, indices], flat_tab[:, indices], flat_observed[:, indices])
                    module_losses[f"L_med_{name}"] = value
                    l_med = l_med + value
            elif self.ab_degenerate:
                pred_all = self.degenerate_decoder(state["m_tab_single_view"])
                l_med = masked_reconstruction(pred_all, flat_tab, flat_observed)
                module_losses["L_med_all"] = l_med
            elif self.ab_single_decoder:
                concat_med = torch.cat(
                    [med_modules[n] for n in self.module_names], dim=-1
                )
                pred_all = self.single_module_decoder(concat_med)
                l_med = masked_reconstruction(pred_all, flat_tab, flat_observed)
                module_losses["L_med_all"] = l_med
            else:
                for name, indices in self.feature_modules.items():
                    pred_module = self.module_decoders[name](med_modules[name])
                    target = flat_tab[:, indices]
                    module_losses[f"L_med_{name}"] = masked_reconstruction(
                        pred_module, target, flat_observed[:, indices]
                    )
                    l_med = l_med + module_losses[f"L_med_{name}"]
            loss_dict["L_med"] = l_med
            loss_dict.update(module_losses)

            f = state["f"]
            if B > 1:
                l_corr = self._token_corr_loss(
                    state["regularizer_tokens"], state["regularizer_mask"], f)
            else:
                l_corr = logits.new_zeros(())
            loss_dict["L_corr"] = l_corr

            # Optional experiment: the same two-layer head reads the actual
            # mediator tokens from a fixed zero query. Default is off because
            # mediator-only supervision can drive the model MR toward one.
            l_fd = logits.new_zeros(())
            l_m_cls = logits.new_zeros(())
            if self.auxiliary_mode == "token_query":
                logits_med = self.predict_from_states(
                    torch.zeros_like(f), state["M"], reference_lambda=0.0,
                    token_mask=state["token_mask"],
                )["logits"]
                l_fd = F.kl_div(
                    F.log_softmax(logits_med, dim=-1),
                    F.softmax(logits.detach(), dim=-1),
                    reduction="batchmean", log_target=False,
                )
                l_m_cls = F.cross_entropy(
                    logits_med, labels, weight=class_weights,
                    label_smoothing=label_smoothing,
                )
            if self.auxiliary_mode == "token_query":
                loss_dict["L_fd"] = l_fd
                loss_dict["L_m_cls"] = l_m_cls
            if self.ab_degenerate or self.v20_unified or self.delta == 0:
                l_indep = logits.new_zeros(())
            else:
                l_indep = self._module_independence_loss(med_modules)
            loss_dict["L_indep"] = l_indep
            loss = (
                l_cls + self.observed_cls_weight * l_obs + self.reference_cls_weight * l_ref
                + self.alpha * l_med + self.beta * l_corr
                + self.gamma * l_fd
                + self.mediator_cls_weight * l_m_cls
                + self.delta * l_indep
            )
            if self.image_aux_weight > 0:
                image_parameter = self.image_aux_classifier.weight
                image_logits = self.image_aux_classifier(state["pure_image_patient"].to(image_parameter))
                l_image_aux = F.cross_entropy(image_logits, labels, weight=class_weights,
                                             label_smoothing=label_smoothing)
                loss = loss + self.image_aux_weight * l_image_aux
                loss_dict["L_image_aux"] = l_image_aux
            loss_dict["tortuosity_training_drop_fraction"] = state["tortuosity_training_drop"].float().mean()
            loss_dict["loss"] = loss
        return {
            "loss": loss,
            "logits": logits,
            "logits_cur": pred["logits_cur"],
            "logits_ref": pred["logits_ref"],
            "reference_margin_std": pred["reference_margin_std"],
            "probs": pred["probs"],
            "F_query": state["f"],
            "M_tokens": state["M"],
            "M_token_mask": state["token_mask"],
            "M_v": state["m_i_reshaped"],
            "image_view_weights": state["image_view_weights"],
            "module_view_weights": state["module_view_weights"],
            "pure_image_logits": (self.image_aux_classifier(state["pure_image_patient"].to(self.image_aux_classifier.weight))
                                  if self.image_aux_weight > 0 else None),
            "M_struct": state["m_struct"],
            "M_cot": state["m_cot"],
            "M_patient": state["m_patient"],
            "F_fd": pred["F_fd"],
            "loss_dict": loss_dict,
            "med_modules": {
                n: state["med_modules"][n].reshape(
                    B, state["v_mask"].size(1), -1
                ) for n in MODULE_NAMES
            },
        }

    @torch.no_grad()
    def model_path_effects(
        self,
        current_state: Dict[str, torch.Tensor],
        baseline_state: Dict[str, torch.Tensor],
        target_class: int = 1,
        reference_queries: Optional[torch.Tensor] = None,
        reference_lambda: Optional[float] = None,
        freeze_semantic: bool = False,
    ) -> Dict[str, torch.Tensor]:
        """Algebraic model-path decomposition on the sole final predictor.

        TE=s(f1,M1)-s(f0,M0), DE=s(f1,M0)-s(f0,M0),
        IE=s(f1,M1)-s(f1,M0). A high MR can be architecture-driven,
        especially lambda=1; these values do not identify clinical effects.
        A baseline may have batch size 1 and is then broadcast to current batch.
        """
        if self.training:
            raise RuntimeError("call model.eval() before intervention analysis")
        f1, M1, mask1 = (
            current_state["f"], current_state["M"], current_state["token_mask"]
        )
        f0, M0, mask0 = (
            baseline_state["f"], baseline_state["M"], baseline_state["token_mask"]
        )
        if f0.size(0) == 1 and f1.size(0) != 1:
            f0 = f0.expand(f1.size(0), -1)
            M0 = M0.expand(f1.size(0), -1, -1)
            mask0 = mask0.expand(f1.size(0), -1)
        if f0.shape != f1.shape or M0.shape != M1.shape:
            raise ValueError("current and baseline state shapes must match")
        if freeze_semantic:
            if M1.size(1) != 3:
                raise ValueError("freeze_semantic requires three mediator tokens")
            # All three counterfactual states then share the baseline semantic token.
            M1 = M1.clone()
            mask1 = mask1.clone()
            M1[:, 2] = M0[:, 2]
            mask1[:, 2] = mask0[:, 2]
        def score(f, M, mask):
            result = self.predict_from_states(
                f, M, reference_queries=reference_queries,
                reference_lambda=reference_lambda, token_mask=mask,
            )
            margin = result["logits"][:, target_class] - result["logits"][:, 1 - target_class]
            return result["probs"][:, target_class], margin
        (p11, a11) = score(f1, M1, mask1)
        (p10, a10) = score(f1, M0, mask0)
        (p00, a00) = score(f0, M0, mask0)
        te = p11 - p00
        de = p10 - p00
        ie = p11 - p10
        return {
            "TE_model": te,
            "DE_model": de,
            "IE_model": ie,
            "p11": p11,
            "p10": p10,
            "p00": p00,
            "logit_margin_11": a11,
            "logit_margin_10": a10,
            "logit_margin_00": a00,
            "TE_logit": a11 - a00,
            "DE_logit": a10 - a00,
            "IE_logit": a11 - a10,
            "identity_error": te - de - ie,
            "semantic_frozen": bool(freeze_semantic),
        }

    @torch.no_grad()
    def tabular_intervention_response(
        self,
        sample_id: torch.IntTensor,
        tabular_features: torch.FloatTensor,
        replacement: torch.Tensor,
        *,
        tabular_observed_mask: Optional[torch.BoolTensor] = None,
        module_name: Optional[str] = None,
        feature_index: Optional[int] = None,
        target_class: int = 1,
        text_prompt: Optional[str] = None,
        reference_queries: Optional[torch.Tensor] = None,
        reference_lambda: Optional[float] = None,
    ) -> Dict[str, torch.Tensor]:
        """Named tabular intervention; all Bi-MHA/mediator stages are recomputed.

        replacement is in the same standardized units as tabular_features.
        For a module it must broadcast to [B,V,module_size]; for a feature to
        [B,V,1]. The original LLaMA language state is reused because tabular
        inputs do not enter that encoder in cotstrong.
        """
        if self.training:
            raise RuntimeError("call model.eval() before intervention analysis")
        if tabular_observed_mask is None:
            raise ValueError("v19 tabular intervention requires the original observed-cell mask")
        if (module_name is None) == (feature_index is None):
            raise ValueError("specify exactly one of module_name or feature_index")
        if module_name is not None:
            if module_name not in self.feature_modules:
                raise ValueError(f"unknown vascular module: {module_name}")
            indices = self.feature_modules[module_name]
        else:
            if not 0 <= int(feature_index) < 58:
                raise ValueError("feature_index must be in [0,57]")
            indices = [int(feature_index)]
        baseline = self.encode_patient_state(
            sample_id, tabular_features, tabular_observed_mask=tabular_observed_mask,
            text_prompt=text_prompt
        )
        changed = tabular_features.clone()
        values = torch.as_tensor(
            replacement, device=changed.device, dtype=changed.dtype
        )
        eligible = (tabular_observed_mask.to(device=changed.device, dtype=torch.bool)
                    & baseline["v_mask"].to(device=changed.device, dtype=torch.bool).unsqueeze(-1))
        changed[:, :, indices] = torch.where(
            eligible[:, :, indices],
            torch.broadcast_to(values, changed[:, :, indices].shape),
            changed[:, :, indices],
        )
        intervened = self.encode_patient_state(
            sample_id, changed, tabular_observed_mask=tabular_observed_mask,
            text_prompt=text_prompt,
            fixed_language_state=baseline["language_cache"],
        )
        original_pred = self.predict_from_states(
            baseline["f"], baseline["M"],
            reference_queries=reference_queries,
            reference_lambda=reference_lambda,
            token_mask=baseline["token_mask"],
        )
        changed_pred = self.predict_from_states(
            baseline["f"], intervened["M"],
            reference_queries=reference_queries,
            reference_lambda=reference_lambda,
            token_mask=intervened["token_mask"],
        )
        return {
            "baseline_probability": original_pred["probs"][:, target_class],
            "intervened_probability": changed_pred["probs"][:, target_class],
            "response": (
                original_pred["probs"][:, target_class]
                - changed_pred["probs"][:, target_class]
            ),
            "feature_indices": indices,
            "module_name": module_name,
        }

    # Legacy single-vector compute_* APIs are intentionally absent here.
    # New explanation scripts must call model_path_effects,
    # tabular_intervention_response, or predict_from_states explicitly.

    # ────────────────────────────────────────────────────────────────────────
    # 保存 / 加载
    # ────────────────────────────────────────────────────────────────────────

    def save_pretrained(self, save_directory, model_name="stroke_causal_refnwgm_v19", epoch=None):
        os.makedirs(save_directory, exist_ok=True)
        save_params = {
            name: param
            for name, param in self.named_parameters()
            if param.requires_grad and any(k in name for k in [
                "v20_", "projector", "lora",
                "modular_encoder", "local_vascular_attn",
                "module_missing_embeddings", "module_confidence_gates",
                "image_view_scorer", "module_view_scorers", "pure_visual_gate_logit",
                "tortuosity_unavailable_embedding",
                "visual_mediator_proj", "cross_modal_attn", "cross_modal_norm",
                "visual_local_mapper", "visual_global_mapper",
                "tab_module_to_hidden", "hidden_to_tab_module",
                "tab_to_visual_attn", "visual_to_tab_attn",
                "cross_guided_norm_v", "cross_guided_norm_q",
                "visual_base_proj", "visual_mediator_integrator",
                "visual_mediator_gate", "visual_mediator_norm",
                "tabular_module_gates", "tabular_module_norms",
                "dual_mediator_residual", "dual_mediator_gate",
                "dual_mediator_norm",
                "module_mediator_projs", "module_aggregate",
                "struct_mediator_residual", "struct_mediator_gate", "struct_mediator_norm",
                "module_out_projs", "m_t_proj", "slot_align",
                "feature_proj", "front_door_attn", "cot_semantic_mediator", "classifier",
                "query_image_adapter",
                "mediator_token_projs", "mediator_type_embedding",
                "module_decoders",
                "single_module_decoder",
                "degenerate_tab_encoder", "degenerate_joint_proj", "degenerate_decoder",
            ])
        }
        save_params["reference_lambda_logit"] = self.reference_lambda_logit.detach().cpu()
        save_params["front_door_attn.query_residual_logit"] = self.front_door_attn.query_residual_logit.detach().cpu()
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        path = os.path.join(save_directory, f"{model_name}{suffix}.pth")
        required_new = ["mediator_token_projs.", "mediator_type_embedding.",
                        "module_missing_embeddings.", "module_confidence_gates."]
        required_new.extend(["image_view_scorer.", "module_view_scorers."])
        if self.tortuosity_module_drop_prob > 0:
            required_new.append("tortuosity_unavailable_embedding")
        if self.pure_visual_path:
            required_new.append("pure_visual_gate_logit")
        if self.image_aux_weight > 0:
            required_new.append("image_aux_classifier.")
        if self.query_mode == "image_residual":
            required_new.append("query_image_adapter.")
        if self.ab_single_decoder:
            required_new.append("single_module_decoder.")
        for prefix in required_new:
            if not any(name.startswith(prefix) for name in save_params):
                raise RuntimeError(f"v19 checkpoint missing new module {prefix}")
        # The added prompt-token rows are frozen by k-bit preparation. They
        # must nevertheless be restored for a separately loaded predictor.
        embeddings = self.language_model.get_input_embeddings().weight
        save_params["__v5_base_vocab_size__"] = torch.tensor(self.base_vocab_size)
        save_params["__v5_added_embedding_rows__"] = (
            embeddings[self.base_vocab_size:].detach().cpu().clone()
        )
        save_params["__v19_tortuosity_drop_prob__"] = torch.tensor(self.tortuosity_module_drop_prob)
        save_params["__v19_pure_visual_path__"] = torch.tensor(int(self.pure_visual_path), dtype=torch.int8)
        save_params["__v19_query_mode__"] = torch.tensor(
            1 if self.query_mode == "image_residual" else 0, dtype=torch.int8
        )
        save_params["__v26_settings__"] = self.v26_settings()
        torch.save(save_params, path)
        print(f"Model saved to {path}")

    def load_pretrained(self, save_directory, model_name="stroke_causal_refnwgm_v19", epoch=None):
        suffix = f"_epoch{epoch}" if epoch is not None else ""
        path = os.path.join(save_directory, f"{model_name}{suffix}.pth")
        state = torch.load(path, map_location="cpu", weights_only=True)
        if state.pop("__v26_settings__", None) != self.v26_settings():
            raise RuntimeError("Checkpoint architecture/config differs from this v20 variant")
        saved_drop = state.pop("__v19_tortuosity_drop_prob__", None)
        if saved_drop is None or abs(float(saved_drop) - self.tortuosity_module_drop_prob) > 1e-6:
            raise RuntimeError("Checkpoint tortuosity dropout setting mismatch")
        saved_pure = state.pop("__v19_pure_visual_path__", None)
        if saved_pure is None or bool(int(saved_pure)) != self.pure_visual_path:
            raise RuntimeError("Checkpoint pure-visual mode mismatch")
        saved_query_mode = state.pop("__v19_query_mode__", None)
        expected_query_mode = 1 if self.query_mode == "image_residual" else 0
        if saved_query_mode is None or int(saved_query_mode) != expected_query_mode:
            raise RuntimeError(
                f"{path} query_mode mismatch; expected {self.query_mode}"
            )
        base_vocab = state.pop("__v5_base_vocab_size__", None)
        added_rows = state.pop("__v5_added_embedding_rows__", None)
        if base_vocab is None or added_rows is None:
            raise RuntimeError(f"{path} lacks the frozen added-token embeddings")
        embeddings = self.language_model.get_input_embeddings().weight
        if (int(base_vocab) != self.base_vocab_size
                or tuple(added_rows.shape) != tuple(embeddings[self.base_vocab_size:].shape)):
            raise RuntimeError(f"{path} added-token embedding shape/vocabulary mismatch")
        required = [
            "reference_lambda_logit", "front_door_attn.query_residual_logit",
            "mediator_token_projs.", "mediator_type_embedding.",
            "front_door_attn.", "classifier.",
            "module_missing_embeddings.", "module_confidence_gates.",
        ]
        required.extend(["image_view_scorer.", "module_view_scorers."])
        if self.v20_unified:
            required.extend(["v20_unified_encoder.", "v20_unified_decoder.", "v20_unified_scorer."])
        if self.v20_mode == "direct_fusion":
            required.extend(["v20_direct_tab.", "v20_direct_fusion."])
        if self.tortuosity_module_drop_prob > 0:
            required.append("tortuosity_unavailable_embedding")
        if self.pure_visual_path:
            required.append("pure_visual_gate_logit")
        if self.image_aux_weight > 0:
            required.append("image_aux_classifier.")
        if self.query_mode == "image_residual":
            required.append("query_image_adapter.")
        if self.ab_single_decoder:
            required.append("single_module_decoder.")
        for prefix in required:
            if not any(name.startswith(prefix) for name in state):
                raise RuntimeError(
                    f"{path} lacks {prefix}; this is not a complete refnwgm_v19 checkpoint"
                )
        incompatible = self.load_state_dict(state, strict=False)
        missing_required = [
            name for name in incompatible.missing_keys
            if any(name.startswith(prefix) for prefix in required)
        ]
        if missing_required:
            raise RuntimeError(f"v19 checkpoint could not load keys: {missing_required}")
        missing_trainable = [
            name for name, param in self.named_parameters()
            if param.requires_grad and name not in state
        ]
        if missing_trainable:
            raise RuntimeError(
                f"v19 checkpoint omitted trainable parameters: {missing_trainable[:8]}"
            )
        if incompatible.unexpected_keys:
            logger.warning("Unexpected v6 checkpoint keys: %s", incompatible.unexpected_keys)
        with torch.no_grad():
            embeddings[self.base_vocab_size:].copy_(
                added_rows.to(device=embeddings.device, dtype=embeddings.dtype)
            )
        print(f"Model loaded from {path}")
