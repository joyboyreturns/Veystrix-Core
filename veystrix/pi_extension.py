"""The Veystrix provider extension for the pi coding agent, as installable TypeScript.

``python -m veystrix.harness_configure --apply`` writes ``~/.pi/agent/extensions/veystrix.ts``
from ``EXTENSION_SOURCE`` (idempotent: an unchanged file is left alone and a changed file gets
a timestamped backup). The source lives here, not in ``integrations/``, so the packaged app can
install it: the engine bundle ships only the ``veystrix`` and ``models`` packages.

The extension discovers models from the engine's ``GET /v1/models``: ``context_window`` is used
as-is, and ``max_output_tokens`` (the server's own output ceiling) becomes pi's ``maxTokens``.
pi then never asks for more output than fits, and the engine clamps whatever the client asks to
the room actually left in the window — uncapped generation in practice.
"""

EXTENSION_SOURCE = '''\
import type { ExtensionAPI } from "@earendil-works/pi-coding-agent";

// Installed by `python -m veystrix.harness_configure` (canonical source: veystrix/pi_extension.py).
// Override either default with environment variables when the engine runs elsewhere or needs a key.
const BASE = process.env.VEYSTRIX_BASE_URL ?? "http://127.0.0.1:8080/v1";
// The engine accepts any value on loopback unless it was started with a required key.
const API_KEY = process.env.VEYSTRIX_API_KEY ?? "local";

type ServerModel = {
  id: string;
  name?: string;
  family?: string;
  context_window?: number;
  max_output_tokens?: number;
};

async function discover(signal?: AbortSignal) {
  const response = await fetch(`${BASE}/models`, {
    signal,
    headers: { Authorization: `Bearer ${API_KEY}` },
  });

  if (!response.ok) {
    throw new Error(`Veystrix returned HTTP ${response.status}`);
  }

  const payload = await response.json() as {
    data?: ServerModel[];
  };

  if (!Array.isArray(payload.data)) {
    throw new Error("Invalid Veystrix model catalog");
  }

  return payload.data
    .filter(m => typeof m.id === "string" && m.id.length > 0)
    .map(m => {
      // The engine clamps any requested output cap to the room left in the window, so the
      // server's own ceiling is "unlimited output": never the bogus 4096 default.
      const contextWindow = m.context_window ?? 16384;
      const maxTokens = Math.max(1, m.max_output_tokens ?? contextWindow);
      const qwen = m.family === "flashnext" || m.id.startsWith("veystrix-flashnext");
      const nemotron = m.family === "nemotron35-lightning" || m.id.startsWith("veystrix-nemotron35-lightning");

      return {
        id: m.id,
        name: m.name ?? m.id,
        reasoning: qwen || nemotron,
        thinkingLevelMap: nemotron
          ? { minimal: null, low: null, medium: "medium", high: null, xhigh: null, max: null }
          : { minimal: null, xhigh: "xhigh", max: null },
        compat: {
          supportsDeveloperRole: false,
          thinkingFormat: "qwen" as const,
          // Nemotron sends only medium, avoiding a Qwen-specific server default.
          supportsReasoningEffort: qwen || nemotron,
          supportsThinkingTokenBudget: false,
          sendSessionAffinityHeaders: true,
          // This Pi header format uses X-Session-Id, which the router forwards.
          sessionAffinityFormat: "openrouter" as const,
        },
        input: ["text"] as ("text" | "image")[],
        cost: {
          input: 0,
          output: 0,
          cacheRead: 0,
          cacheWrite: 0
        },
        contextWindow,
        maxTokens
      };
    });
}

export default async function (pi: ExtensionAPI) {
  let models: Awaited<ReturnType<typeof discover>> = [];

  try {
    models = await discover(AbortSignal.timeout(3000));
  } catch {
    console.warn("Veystrix unavailable; no models discovered.");
  }

  pi.registerProvider("veystrix", {
    name: "Veystrix Local",
    baseUrl: BASE,
    apiKey: API_KEY,
    api: "openai-completions",
    models,
    async refreshModels({ signal }) {
      return discover(signal);
    }
  });
}
'''
