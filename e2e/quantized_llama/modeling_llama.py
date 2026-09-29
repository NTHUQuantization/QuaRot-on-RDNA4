import quarot
from transformers import LlamaConfig
from transformers.models.llama.modeling_llama import (
    LlamaAttention, LlamaForCausalLM, LlamaMLP, apply_rotary_pos_emb)
try:
    from e2e.quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)
except ImportError:
    from .quantized_common import (
        QuarotAttentionMixin, QuarotCausalLMMixin, QuarotMLPMixin)

class QuarotLlamaConfig(LlamaConfig):
    model_type = "llama_quarot"

class QuarotFP16LlamaAttention(QuarotAttentionMixin, LlamaAttention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(False)

class QuarotLlamaAttention(QuarotAttentionMixin, LlamaAttention):
    _quarot_apply_rotary = staticmethod(apply_rotary_pos_emb)
    def __init__(self, config, layer_idx):
        super().__init__(config, layer_idx)
        self._init_quarot_attention(True)

class QuarotLlamaMLP(QuarotMLPMixin, LlamaMLP):
    def __init__(self, config):
        super().__init__(config)
        self._init_quarot_mlp()

class QuarotFP16LlamaForCausalLM(QuarotCausalLMMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "float16"
        self._init_quarot_model(QuarotFP16LlamaAttention)

class QuarotLlamaForCausalLM(QuarotCausalLMMixin, LlamaForCausalLM):
    def __init__(self, config):
        super().__init__(config)
        self.cache_dtype = "int4"
        self._init_quarot_model(
            QuarotLlamaAttention, QuarotLlamaMLP,
            quarot.nn.FusedRMSNormQuant)
