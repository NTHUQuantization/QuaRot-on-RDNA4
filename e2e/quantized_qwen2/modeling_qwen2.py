import quarot
from transformers import Qwen2Config
from transformers.models.qwen2.modeling_qwen2 import (
    Qwen2Attention, Qwen2ForCausalLM, Qwen2MLP, apply_rotary_pos_emb)
try:
    from e2e.quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)
except ImportError:
    from .quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)

class QuarotQwen2Config(Qwen2Config):
    model_type = "qwen2_quarot"

class QuarotFP16Qwen2Attention(QuarotAttentionMixin, Qwen2Attention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(False)

class QuarotQwen2Attention(QuarotAttentionMixin, Qwen2Attention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(True)

class QuarotQwen2MLP(QuarotMLPMixin, Qwen2MLP):
    def __init__(self, config):
        super().__init__(config)
        self._init_quarot_mlp()

class QuarotFP16Qwen2ForCausalLM(QuarotCausalLMMixin, Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "float16"
        self._init_quarot_model(QuarotFP16Qwen2Attention)

class QuarotQwen2ForCausalLM(QuarotCausalLMMixin, Qwen2ForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "int4"
        self._init_quarot_model(
            QuarotQwen2Attention, QuarotQwen2MLP,
            quarot.nn.FusedRMSNormQuant)
