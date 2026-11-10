"""Experimental hybrid model that never constructs resident routed-expert matrices.

Reference components come from the pinned mlx-lm Nemotron-H model. The caller
must supply a validated per-layer streamed-expert implementation and explicitly
load matching dense weights. There is no production checkpoint loader yet.
"""
import mlx.core as mx
import mlx.nn as nn
from mlx_lm.models.base import create_attention_mask, create_ssm_mask
from mlx_lm.models.cache import ArraysCache, KVCache
from mlx_lm.models.nemotron_h import NemotronHMamba2Mixer, NemotronHAttention, NemotronHMLP, MoEGate

class StreamingMoE(nn.Module):
    def __init__(self, args, experts):
        super().__init__()
        self.gate = MoEGate(args)
        self.experts = experts
        self.shared = NemotronHMLP(args, intermediate_size=args.moe_shared_expert_intermediate_size)
        self.latent = args.moe_latent_size is not None
        if self.latent:
            self.fc1_latent_proj = nn.Linear(args.hidden_size, args.moe_latent_size, bias=args.mlp_bias)
            self.fc2_latent_proj = nn.Linear(args.moe_latent_size, args.hidden_size, bias=args.mlp_bias)

    def __call__(self, x):
        indices, weights = self.gate(x)
        routed_input = self.fc1_latent_proj(x) if self.latent else x
        routed = self.experts(routed_input, indices)
        routed = (routed * weights[..., None]).sum(axis=-2).astype(x.dtype)
        if self.latent:
            routed = self.fc2_latent_proj(routed)
        return routed + self.shared(x)

class HybridBlock(nn.Module):
    def __init__(self, args, kind, experts=None):
        super().__init__()
        self.kind = kind
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        if kind == 'M':
            self.mixer = NemotronHMamba2Mixer(args)
        elif kind == '*':
            self.mixer = NemotronHAttention(args)
        elif kind == 'E':
            if experts is None:
                raise ValueError('MoE layer requires an expert store')
            self.mixer = StreamingMoE(args, experts)
        else:
            raise ValueError(f'unsupported layer type: {kind}')

    def __call__(self, x, mask=None, cache=None):
        y = self.norm(x)
        return x + (self.mixer(y, mask=mask, cache=cache) if self.kind in ('M', '*') else self.mixer(y))

class StreamingHybrid(nn.Module):
    def __init__(self, args, experts_for_layer):
        super().__init__()
        pattern = args.hybrid_override_pattern
        if not pattern or 'M' not in pattern or '*' not in pattern:
            raise ValueError('hybrid must include Mamba and attention')
        self.embeddings = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [HybridBlock(args, kind, experts_for_layer(i) if kind == 'E' else None)
                       for i, kind in enumerate(pattern)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.layer_norm_epsilon)
        self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)
        self._cache_kinds = [kind for kind in pattern if kind in ('M', '*')]
        self._mamba_index = self._cache_kinds.index('M')
        self._attention_index = self._cache_kinds.index('*')

    def make_cache(self):
        return [ArraysCache(size=2) if kind == 'M' else KVCache() for kind in self._cache_kinds]

    def __call__(self, tokens, cache=None):
        cache = self.make_cache() if cache is None else cache
        if len(cache) != len(self._cache_kinds):
            raise ValueError('cache layout does not match hybrid layers')
        x = self.embeddings(tokens)
        amask = create_attention_mask(x, cache[self._attention_index])
        mmask = create_ssm_mask(x, cache[self._mamba_index])
        index = 0
        for layer in self.layers:
            state = cache[index] if layer.kind in ('M', '*') else None
            index += layer.kind in ('M', '*')
            x = layer(x, mask=amask if layer.kind == '*' else mmask, cache=state)
        return self.lm_head(self.norm_f(x))
