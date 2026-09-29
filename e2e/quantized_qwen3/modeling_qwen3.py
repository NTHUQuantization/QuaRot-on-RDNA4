import quarot
from transformers import Qwen3Config
from transformers.models.qwen3.modeling_qwen3 import (
    Qwen3Attention, Qwen3ForCausalLM, Qwen3MLP, apply_rotary_pos_emb)
try:
    from e2e.quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)
except ImportError:
    from .quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)

class QuarotQwen3Config(Qwen3Config):
    model_type = "qwen3_quarot"

class QuarotFP16Qwen3Attention(QuarotAttentionMixin, Qwen3Attention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(False)

class QuarotQwen3Attention(QuarotAttentionMixin, Qwen3Attention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(True)

class QuarotQwen3MLP(QuarotMLPMixin, Qwen3MLP):
    def __init__(self, config):
        super().__init__(config)
        self._init_quarot_mlp()

class QuarotFP16Qwen3ForCausalLM(QuarotCausalLMMixin, Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "float16"
        self._init_quarot_model(QuarotFP16Qwen3Attention)

class QuarotQwen3ForCausalLM(QuarotCausalLMMixin, Qwen3ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "int4"
        self._init_quarot_model(
            QuarotQwen3Attention, QuarotQwen3MLP,
            quarot.nn.FusedRMSNormQuant)
