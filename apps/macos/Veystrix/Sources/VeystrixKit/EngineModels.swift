import Foundation

public enum EngineState: Equatable, Sendable {
    case stopped, starting, running, stopping
    case error(String)

    public var label: String {
        switch self {
        case .stopped: return "Stopped"
        case .starting: return "Starting"
        case .running: return "Running"
        case .stopping: return "Stopping"
        case .error: return "Error"
        }
    }
    public var isActive: Bool { self == .starting || self == .running || self == .stopping }
}

/// Subset of the engine's `GET /health` JSON. Every field is optional so a newer or older engine still decodes.
public struct HealthReport: Decodable, Equatable, Sendable {
    public var status: String?
    public var ready: Bool?
    public var model: String?
    public var installedModelID: String?
    public var busy: Bool?
    public var contextLimit: Int?
    public var maxColdPrefill: Int?
    public var flashNextKVCache: String?
    public var fault: String?
    public var memoryPolicy: MemoryPolicy?
    public var cache: CacheState?
    public var queue: Queue?
    public var activity: Activity?
    public var worker: Worker?

    public struct Activity: Decodable, Equatable, Sendable {
        public var active: Bool?
        public var stage: String?
        public var promptTokens: Int?
        public var prefillDone: Int?
        public var prefillTotal: Int?
        public var decodeTokens: Int?
        public var decodeTokensPerSecond: Double?
        enum CodingKeys: String, CodingKey {
            case active, stage
            case promptTokens = "prompt_tokens", prefillDone = "prefill_done", prefillTotal = "prefill_total"
            case decodeTokens = "decode_tokens", decodeTokensPerSecond = "decode_tokens_per_second"
        }
    }
    public struct Worker: Decodable, Equatable, Sendable {
        public var stage: String?
    }

    public struct MemoryPolicy: Decodable, Equatable, Sendable {
        public var canAdmit: Bool?
        public var stopReason: String?
        public var currentSwapGrowthMb: Double?
        public var peakSwapGrowthMb: Double?
        public var swapCeilingMib: Double?
        enum CodingKeys: String, CodingKey {
            case canAdmit = "can_admit", stopReason = "stop_reason", currentSwapGrowthMb = "current_swap_growth_mb"
            case peakSwapGrowthMb = "peak_swap_growth_mb", swapCeilingMib = "swap_ceiling_mib"
        }
    }
    /// The engine's cache state has no stable schema; keep the numbers we can recognise.
    public struct CacheState: Decodable, Equatable, Sendable {
        public var values: [String: Double] = [:]
        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let dict = try? c.decode([String: JSONValue].self) {
                for (k, v) in dict { if let n = v.number { values[k] = n } }
            }
        }
    }
    public struct Queue: Decodable, Equatable, Sendable {
        public var values: [String: Double] = [:]
        public init(from decoder: Decoder) throws {
            let c = try decoder.singleValueContainer()
            if let dict = try? c.decode([String: JSONValue].self) {
                for (k, v) in dict { if let n = v.number { values[k] = n } }
            }
        }
    }
    enum CodingKeys: String, CodingKey {
        case status, ready, model, busy, fault, cache, queue, activity, worker
        case installedModelID = "installed_model_id"
        case flashNextKVCache = "flashnext_kv_cache"
        case contextLimit = "context_limit", maxColdPrefill = "max_cold_prefill", memoryPolicy = "memory_policy"
    }

    public var isHealthy: Bool { (ready ?? false) && (fault == nil) }
}

/// Minimal JSON value, so unknown shapes decode without failing.
public enum JSONValue: Decodable, Equatable, Sendable {
    case number(Double), string(String), bool(Bool), null, other
    public init(from decoder: Decoder) throws {
        let c = try decoder.singleValueContainer()
        if c.decodeNil() { self = .null }
        else if let b = try? c.decode(Bool.self) { self = .bool(b) }
        else if let n = try? c.decode(Double.self) { self = .number(n) }
        else if let s = try? c.decode(String.self) { self = .string(s) }
        else { self = .other }
    }
    public var number: Double? { if case .number(let n) = self { return n }; return nil }
}

/// `GET /v1/models` (OpenAI shape) as served by the engine.
public struct ModelsReport: Decodable, Equatable, Sendable {
    public struct Entry: Decodable, Equatable, Sendable {
        public var id: String
        public var contextWindow: Int?
        public var suggestedClientContextWindow: Int?
        public var contextLimitEnforced: Bool?
        public var maxOutputTokens: Int?
        enum CodingKeys: String, CodingKey {
            case id
            case contextWindow = "context_window"
            case suggestedClientContextWindow = "suggested_client_context_window"
            case contextLimitEnforced = "context_limit_enforced"
            case maxOutputTokens = "max_output_tokens"
        }
    }
    public var data: [Entry]
}

/// Metadata-only installed checkpoint inventory. `loaded` is true only for the exact active installation.
public struct InstalledModelsReport: Decodable, Sendable {
    public struct Entry: Decodable, Identifiable, Sendable {
        public let id: String
        public let family: String
        public let displayName: String
        public let checkpoint: String
        public let architecture: String
        public let loaded: Bool
        public let verification: String
        enum CodingKeys: String, CodingKey {
            case id, family, checkpoint, architecture, loaded, verification
            case displayName = "display_name"
        }
    }
    public let data: [Entry]
}
