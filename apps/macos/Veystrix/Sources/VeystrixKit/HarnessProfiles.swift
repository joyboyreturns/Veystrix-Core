import Foundation

public enum Harness: String, CaseIterable, Identifiable, Sendable {
    case hermes = "Hermes Agent"
    case qwenCode = "Qwen Code"
    case deepSeek = "DeepSeek Harness"
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
                               suggestedClientContext: Int? = nil) -> HarnessSnippet {
        let base = "http://127.0.0.1:\(port)/v1"
        switch harness {
        case .hermes:
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
                  context_length: \(suggestedClientContext ?? 16384)
                fallback_providers: []
                """,
                notes: [
                    "Use a dedicated profile so your other providers are untouched.",
                    "fallback_providers is empty so Hermes never falls back to a cloud provider.",
                    "Set the auxiliary tasks (title generation, compression, vision, web) to this endpoint or turn them off; each extra request evicts the live cache.",
                    "context_length must be no larger than the engine's configured limit. Check GET /v1/models once the engine is running.",
                ])
        case .qwenCode:
            return HarnessSnippet(
                harness: harness,
                whereToPutIt: "~/.qwen/settings.json (merge with your existing settings; keep other providers)",
                content: """
                {
                  "modelProviders": {"openai": [{
                    "id": "veystrix-flashnext", "name": "Veystrix (local)",
                    "baseUrl": "\(base)", "envKey": "\(defaultKeyEnv)",
                    "generationConfig": {"timeout": 3600000, "streamIdleTimeoutMs": 900000, "maxRetries": 0,
                                         "contextWindowSize": \(suggestedClientContext ?? 45056)}}]},
                  "security": {"auth": {"selectedType": "openai"}},
                  "model": {"name": "veystrix-flashnext"},
                  "memory": {"enableManagedAutoMemory": false, "enableManagedAutoDream": false, "enableAutoSkill": false},
                  "ui": {"enableFollowupSuggestions": false}
                }
                """,
                notes: [
                    "maxRetries is 0: a retry after a timeout would restart a prefill the engine may still be running.",
                    "The background memory and suggestion switches are off so no second model request evicts the live cache.",
                    "Qwen Code's tool schemas are large; see the engine's Qwen Code integration notes for the tool exclusion list.",
                ])
        case .deepSeek:
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
                        models:
                          - id: \(modelID)
                            name: Veystrix (local)
                            contextWindow: \(suggestedClientContext ?? 20480)
                            maxTokens: 1024
                - id: agent-default-model
                  name: "@deepseek-ai/dsh-agent-default-model"
                  config: {provider: veystrix, model: \(modelID)}
                """,
                notes: [
                    "contextWindow is the engine's window plus the harness's fixed 4096-token reserve; the engine's /v1/models reports `suggested_client_context_window`.",
                    "Trim the harness's tool set as in the engine's DSH patch files to keep requests small.",
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
