"""Pure, non-destructive configuration transforms for harness model catalogs.

These functions do not write user files. Callers must back up and atomically
replace any existing configuration, with the user's explicit action.
"""
from __future__ import annotations

import copy

from veystrix.context_budget import suggested_client_window
from veystrix.model_registry import catalog_context


def model_context(model):
    """Configured window a harness should advertise, per family.

    Flash-Next is bounded by checkpoint metadata; Nemotron 3.5 Lightning is configured at
    1,048,576 tokens regardless of the checkpoint's max_position_embeddings, because its
    runtime window is not derived from that field. See veystrix/model_registry.py."""
    return catalog_context(model)


def _local_base(port):
    if not isinstance(port, int) or not 1024 <= port <= 65535:
        raise ValueError("invalid local port")
    return f"http://127.0.0.1:{port}/v1"


def qwen_config(settings, models, port=8080):
    """Merge registry entries into Qwen Code's OpenAI provider list."""
    if not isinstance(settings, dict):
        raise ValueError("Qwen settings must be a JSON object")
    result = copy.deepcopy(settings)
    providers = result.setdefault("modelProviders", {})
    if not isinstance(providers, dict):
        raise ValueError("modelProviders must be an object")
    entries = providers.setdefault("openai", [])
    if not isinstance(entries, list) or not all(isinstance(e, dict) for e in entries):
        raise ValueError("openai provider entries must be an array of objects")
    base = _local_base(port)
    ids = {m.id for m in models}
    # Replace only entries bearing an exact installed ID and the Veystrix loopback URL.
    # Other models, providers, and the selected default remain untouched.
    preserved = [e for e in entries if not (e.get("id") in ids and e.get("baseUrl") == base)]
    collisions = {e.get("id") for e in preserved} & ids
    if collisions:
        raise ValueError("an existing non-Veystrix provider uses an installed model ID")
    for m in models:
        preserved.append({"id": m.id, "name": "Veystrix · " + m.display_name, "baseUrl": base,
                          "envKey": "VEYSTRIX_API_KEY",
                          "generationConfig": {"timeout": 7200000, "streamIdleTimeoutMs": 7200000,
                                               "maxRetries": 0, "contextWindowSize": model_context(m)}})
    providers["openai"] = preserved
    return result


def opencode_config(settings, models, port=8080):
    """Merge installed models into the Veystrix provider of an OpenCode config object.

    OpenCode v1 config: ``providers.<id>`` (the old singular ``provider`` key is deprecated).
    Model keys are the engine's installed IDs so requests resolve exactly. ``limit.output``
    equals the window: OpenCode then lets generation run to the end of the context and the
    engine clamps it to the room actually left, which is uncapped output in practice."""
    if not isinstance(settings, dict):
        raise ValueError("OpenCode settings must be a JSON object")
    result = copy.deepcopy(settings)
    providers = result.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("providers must be an object")
    base = _local_base(port)
    existing = providers.get("veystrix")
    if existing is not None:
        if not isinstance(existing, dict):
            raise ValueError("invalid Veystrix provider")
        settings_block = existing.get("settings")
        if isinstance(settings_block, dict) and settings_block.get("baseURL") not in (None, base):
            raise ValueError("existing Veystrix provider points to a different endpoint")
    provider = copy.deepcopy(existing) if isinstance(existing, dict) else {}
    models_block = {}
    for m in models:
        window = catalog_context(m)
        models_block[m.id] = {
            "name": "Veystrix · " + m.display_name,
            "limit": {"context": window, "output": window},
            "capabilities": {"tools": True, "input": ["text"]},
        }
    settings_out = provider.get("settings")
    settings_out = dict(settings_out) if isinstance(settings_out, dict) else {}
    settings_out.update({"baseURL": base, "apiKey": "local"})
    provider.update({"name": provider.get("name") or "Veystrix (local)",
                     "package": "@opencode/ai/providers/openai-compatible",
                     "settings": settings_out,
                     "models": models_block})
    providers["veystrix"] = provider
    return result


def kimi_config(config, models, port=8080):
    """Merge installed models into the Kimi Code provider that already points at this engine.

    A provider is ours when its ``base_url`` equals the engine's, whatever the user named it:
    its model entries get the family-configured ``max_context_size`` (user-added flags and
    names are kept) and missing inventory models are added under it. Only when no provider
    points here is one created, named ``veystrix``. No ``max_output_size`` is written
    anywhere: Kimi's ``openai`` provider type does not read it, so generation stays uncapped
    and the engine clamps it to the room actually left in the window. ``default_model`` is
    never touched: the user's selection remains theirs."""
    if not isinstance(config, dict):
        raise ValueError("Kimi config must be a table")
    result = copy.deepcopy(config)
    providers = result.setdefault("providers", {})
    if not isinstance(providers, dict):
        raise ValueError("providers must be a table")
    base = _local_base(port)
    matches = [k for k, p in providers.items() if isinstance(p, dict) and p.get("base_url") == base]
    if len(matches) > 1:
        raise ValueError("multiple providers point at this engine; merge them by hand")
    if matches:
        provider = providers[matches[0]]
        provider.setdefault("type", "openai")     # the user's name, key and other fields stay
    else:
        existing = providers.get("veystrix")
        if existing is not None:
            if not isinstance(existing, dict):
                raise ValueError("invalid Veystrix provider")
            if existing.get("base_url") not in (None, base):
                raise ValueError("existing Veystrix provider points to a different endpoint")
        provider = copy.deepcopy(existing) if isinstance(existing, dict) else {}
        provider.update({"type": "openai", "base_url": base, "api_key": "local"})
        providers["veystrix"] = provider
    model_table = result.setdefault("models", {})
    if not isinstance(model_table, dict):
        raise ValueError("models must be a table")
    for m in models:
        owner = matches[0] if matches else "veystrix"
        entry = next((t for t in model_table.values()
                      if isinstance(t, dict) and t.get("provider") == owner and t.get("model") == m.id), None)
        if entry is None:
            model_table[m.id] = {"provider": owner, "model": m.id,
                                 "max_context_size": catalog_context(m),
                                 "capabilities": ["tool_use"],
                                 "display_name": "Veystrix · " + m.display_name}
        else:
            entry["max_context_size"] = catalog_context(m)
    return result


def deepseek_config(patch, models, port=8080):
    """Merge into the named Veystrix provider in a dsh cordis patch list."""
    if not isinstance(patch, list) or not all(isinstance(p, dict) for p in patch):
        raise ValueError("DeepSeek patch must be a YAML array of objects")
    result = copy.deepcopy(patch)
    base = _local_base(port)
    matches = [p for p in result if p.get("id") == "llm-pi-ai"]
    if len(matches) != 1:
        raise ValueError("expected exactly one llm-pi-ai plugin entry")
    plugin = matches[0]
    config = plugin.setdefault("config", {})
    if not isinstance(config, dict): raise ValueError("invalid plugin config")
    providers = config.setdefault("providers", {})
    if not isinstance(providers, dict): raise ValueError("invalid provider config")
    existing = providers.get("veystrix")
    if existing is not None and not isinstance(existing, dict): raise ValueError("invalid Veystrix provider")
    if existing and existing.get("baseURL") not in (None, base):
        raise ValueError("existing Veystrix provider points to a different endpoint")
    provider = copy.deepcopy(existing or {})
    old_models = provider.get("models", [])
    if not isinstance(old_models, list) or not all(isinstance(m, dict) for m in old_models):
        raise ValueError("invalid Veystrix model list")
    installed_ids = {m.id for m in models}
    preserved_models = [m for m in old_models if m.get("id") not in installed_ids]
    provider.update({"displayName": "Veystrix", "api": "openai-completions", "baseURL": base,
                     "apiKeyEnv": "VEYSTRIX_API_KEY", "streamIdleTimeoutMs": 7200000,
                     "timeoutMs": 7200000, "retryPolicy": {"mode": "normal", "maxRetries": 0},
                     "models": preserved_models + [{"id": m.id, "name": m.display_name,
                                 "contextWindow": suggested_client_window(model_context(m)),
                                 "maxTokens": model_context(m), "input": ["text"],
                                 "reasoningEfforts": {"off": "none", "low": "low", "medium": "medium",
                                                      "high": "high", "xhigh": "xhigh", "max": "max"}}
                                for m in models]})
    providers["veystrix"] = provider
    return result
