"""Pure, non-destructive configuration transforms for harness model catalogs.

These functions do not write user files. Callers must back up and atomically
replace any existing configuration, with the user's explicit action.
"""
from __future__ import annotations

import copy


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
                                               "maxRetries": 0, "contextWindowSize": 32768}})
    providers["openai"] = preserved
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
                     "models": preserved_models + [{"id": m.id, "name": m.display_name, "contextWindow": 32768,
                                 "maxTokens": 1024, "input": ["text"],
                                 "reasoningEfforts": {"off": "none", "low": "low", "medium": "medium", "high": "high"}}
                                for m in models]})
    providers["veystrix"] = provider
    return result
