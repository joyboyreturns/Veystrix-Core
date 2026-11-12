import Foundation

public enum Harness: String, CaseIterable, Identifiable, Sendable {
    case hermes = "Hermes Agent"
    case qwenCode = "Qwen Code"
    case deepSeek = "DeepSeek Harness"
    case openCode = "OpenCode"
    case pi = "pi"
    case kimiCode = "Kimi Code"
    case openAI = "Other OpenAI-compatible client"
    public var id: String { rawValue }
}

public struct HarnessSnippet: Equatable, Sendable {
    public var harness: Harness
    public var whereToPutIt: String
    public var content: String
    public var notes: [String]
}

/// Copy-paste configuration for supported agents. Pure text generation: nothing is written to another app's files
/// and nothing is sent anywhere. The model id and context numbers should come from the running engine's
/// `GET /v1/models`; the defaults below match what the engine serves for the Flash-Next runtime.
public enum HarnessProfiles {
    /// The id the engine serves for the Flash-Next runtime (`veystrix-<model>`).
    public static let defaultModelID = "veystrix-flashnext"
    public static let defaultKeyEnv = "VEYSTRIX_API_KEY"

    public static func snippet(for harness: Harness, port: Int, modelID: String = defaultModelID,
                               suggestedClientContext: Int? = nil, modelContext: Int? = nil) -> HarnessSnippet {
        let base = "http://127.0.0.1:\(port)/v1"
        // JSONEncoder supplies quotes and escaping for arbitrary model IDs.
        let jsonModelID = String(data: try! JSONEncoder().encode(modelID), encoding: .utf8)!
        let resolvedModelContext = modelContext ?? suggestedClientContext.map { max(1, $0 - 4096) }
        switch harness {
        case .hermes:
            let hermesContext = max(resolvedModelContext ?? 262144, 65536)
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "A dedicated Hermes profile (for example ~/.hermes/profiles/veystrix/config.yaml). Run: HERMES_HOME=~/.hermes/profiles/veystrix hermes",
                content: """
                model:
                  default: "\(modelID)"
                  provider: "custom"
                  base_url: "\(base)"
                  # key from the environment: export \(defaultKeyEnv)=<your Veystrix key> (the server accepts any value on loopback unless a key is required)
                  key_env: "\(defaultKeyEnv)"
                  context_length: \(hermesContext)
                fallback_providers: []
                """,
                notes: [
                    "Use a dedicated profile so your other providers are untouched.",
                    "fallback_providers is empty so Hermes never falls back to a cloud provider.",
                    "Set the auxiliary tasks (title generation, compression, vision, web) to this endpoint or turn them off; each extra request evicts the live cache.",
                    "context_length must be at least 64,000 for Hermes and no larger than the engine's configured limit.",
                ])
        case .qwenCode:
            let qwenContext = resolvedModelContext ?? 262144
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "~/.qwen/settings.json (merge with your existing settings; keep other providers)",
                content: """
                {
                  "modelProviders": {"openai": [{
                    "id": \(jsonModelID), "name": "Veystrix (local)",
                    "baseUrl": "\(base)", "envKey": "\(defaultKeyEnv)",
                    "generationConfig": {"timeout": 7200000, "streamIdleTimeoutMs": 7200000, "maxRetries": 0,
                                         "contextWindowSize": \(qwenContext)}}]},
                  "security": {"auth": {"selectedType": "openai"}},
                  "model": {"name": \(jsonModelID)},
                  "memory": {"enableManagedAutoMemory": false, "enableManagedAutoDream": false, "enableAutoSkill": false},
                  "ui": {"enableFollowupSuggestions": false}
                }
                """,
                notes: [
                    "maxRetries is 0: a retry after a timeout would restart a prefill the engine may still be running.",
                    "No output token cap is configured, so generation runs uncapped up to remaining context.",
                    "The background memory and suggestion switches are off so no second model request evicts the live cache.",
                    "Qwen Code's tool schemas are large; see the engine's Qwen Code integration notes for the tool exclusion list.",
                ])
        case .deepSeek:
            let dshWindow = suggestedClientContext ?? resolvedModelContext.map { $0 + 4096 } ?? 266240
            let dshMaxTokens = resolvedModelContext ?? max(1, dshWindow - 4096)
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "A patch file for dsh, for example veystrix.patch.yml (dsh headless --patch veystrix.patch.yml)",
                content: """
                - id: llm-pi-ai
                  name: "@deepseek-ai/dsh-llm-pi-ai"
                  config:
                    providers:
                      veystrix:
                        displayName: Veystrix
                        apiKeyEnv: \(defaultKeyEnv)
                        api: openai-completions
                        baseURL: \(base)
                        streamIdleTimeoutMs: 7200000
                        timeoutMs: 7200000
                        models:
                          - id: \(modelID)
                            name: Veystrix (local)
                            contextWindow: \(dshWindow)
                            maxTokens: \(dshMaxTokens)
                            input: [text]
                            reasoningEfforts:
                              off: none
                              low: low
                              medium: medium
                              high: high
                              xhigh: xhigh
                              max: max
                - id: agent-default-model
                  name: "@deepseek-ai/dsh-agent-default-model"
                  config: {provider: veystrix, model: \(modelID)}
                """,
                notes: [
                    "contextWindow includes the harness's fixed 4096-token safety reserve so available output never collapses to 1 token near the limit.",
                    "maxTokens equals the full server context window so remaining context governs output length instead of a 1024-token ceiling.",
                    "Trim the harness's tool set as in the engine's DSH patch files to keep requests small.",
                ])
        case .openCode:
            let ocWindow = resolvedModelContext ?? 262144
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "~/.config/opencode/opencode.json (merge with your existing config; keep other providers)",
                content: """
                {
                  "providers": {
                    "veystrix": {
                      "name": "Veystrix (local)",
                      "package": "@opencode/ai/providers/openai-compatible",
                      "settings": {"baseURL": "\(base)", "apiKey": "local"},
                      "models": {
                        \(jsonModelID): {
                          "name": "Veystrix (local)",
                          "limit": {"context": \(ocWindow), "output": \(ocWindow)},
                          "capabilities": {"tools": true, "input": ["text"]}
                        }
                      }
                    }
                  }
                }
                """,
                notes: [
                    "Restart OpenCode after changing the file so it reloads the model list.",
                    "limit.output equals the context window: generation runs to the end of the context and the engine clamps it to the room actually left, which is uncapped output in practice.",
                    "Any apiKey value works while the engine runs on loopback without a required key.",
                    "Refresh Harness Models writes this provider for you (with a backup); this card shows the same values.",
                ])
        case .pi:
            let piWindow = resolvedModelContext ?? 262144
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "~/.pi/agent/extensions/veystrix.ts (installed and updated by Refresh Harness Models)",
                content: """
                // Provider "veystrix", registered by ~/.pi/agent/extensions/veystrix.ts
                baseUrl: \(base)                // models discovered from GET /v1/models
                contextWindow: \(piWindow)      // the engine's window for this model
                maxTokens: engine's max_output_tokens (uncapped: clamped to remaining context)
                Optional overrides:
                  export VEYSTRIX_BASE_URL=\(base)
                  export VEYSTRIX_API_KEY=<your Veystrix key>
                """,
                notes: [
                    "Refresh Harness Models installs the extension (with a backup); restart pi or run the extension's refreshModels to pick up a changed model list.",
                    "maxTokens comes from the engine's own output ceiling instead of a 4096-token default, so generation runs uncapped up to remaining context.",
                ])
        case .kimiCode:
            let kimiContext = resolvedModelContext ?? 262144
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "~/.kimi-code/config.toml (merge with your existing config; default_model is never touched)",
                content: """
                [providers.veystrix]
                type = "openai"
                base_url = "\(base)"
                api_key = "local"

                [models.\(jsonModelID)]
                provider = "veystrix"
                model = \(jsonModelID)
                max_context_size = \(kimiContext)
                capabilities = ["tool_use"]
                """,
                notes: [
                    "max_context_size is the engine's window for this model; no max_output_size is written because Kimi's openai provider type does not read it, so generation runs uncapped.",
                    "Refresh Harness Models writes this file even when Kimi Code is not installed yet; run it again if a Kimi Code login rewrites the file.",
                    "Any api_key value works while the engine runs on loopback without a required key.",
                    "Use /skill:veystrix-swarm from Kimi Code to run a native architect and executor workflow on one local engine.",
                ])
        case .openAI:
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "Environment variables or the client's provider settings",
                content: """
                OPENAI_BASE_URL=\(base)
                OPENAI_API_KEY=<value of \(defaultKeyEnv), or any text if no key is required>
                model: \(modelID)
                endpoints: POST /v1/chat/completions (streaming, tools, reasoning), POST /v1/responses, POST /v1/messages
                """,
                notes: [
                    "Browser pages are refused by the engine unless you allow their origin; command-line clients and SDKs are unaffected.",
                    "One request runs at a time. Do not point several agents at one engine.",
                ])
        }
    }
}

/// Connection state shown next to each harness card.
public enum ConnectionState: Equatable, Sendable {
    case engineStopped, reachable(modelID: String?, contextWindow: Int?), unauthorized, unreachable
    public var label: String {
        switch self {
        case .engineStopped: return "Engine stopped"
        case .reachable: return "Endpoint reachable"
        case .unauthorized: return "Key rejected"
        case .unreachable: return "Not reachable"
        }
    }
}
