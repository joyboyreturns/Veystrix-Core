import Foundation

/// Wire-compatible reasoning levels for `/v1/chat/completions`.
public enum ChatReasoningEffort: String, CaseIterable, Codable, Identifiable, Sendable {
    case off = "none"
    case low = "low"
    case medium = "medium"
    case high = "high"
    case xhigh = "xhigh"

    public var id: String { rawValue }

    public var label: String {
        switch self {
        case .off: return "Off"
        case .low: return "Low"
        case .medium: return "Medium"
        case .high: return "High"
        case .xhigh: return "XHigh"
        }
    }
}

public struct ChatUsage: Equatable, Codable, Sendable {
    public var promptTokens: Int
    public var completionTokens: Int
    public var totalTokens: Int
    public var cachedTokens: Int

    public init(promptTokens: Int, completionTokens: Int, totalTokens: Int, cachedTokens: Int = 0) {
        self.promptTokens = promptTokens
        self.completionTokens = completionTokens
        self.totalTokens = totalTokens
        self.cachedTokens = cachedTokens
    }
}

public enum ChatStreamEvent: Equatable, Sendable {
    case reasoning(String)
    case content(String)
    case finishReason(String)
    case usage(ChatUsage)
    case error(String)
    case done
}

public struct ChatWireMessage: Equatable, Codable, Sendable {
    public var role: String
    public var content: String
    public var reasoningContent: String?

    public init(role: String, content: String, reasoningContent: String? = nil) {
        self.role = role
        self.content = content
        self.reasoningContent = reasoningContent
    }
}

public enum ChatRequestBuilder {
    public enum ValidationError: Error, Equatable, CustomStringConvertible, Sendable {
        case invalidExplicitLimit(String)
        public var description: String {
            switch self {
            case .invalidExplicitLimit(let text):
                return "Explicit debug max output tokens must be a positive integer (got \(text.isEmpty ? "empty" : text))."
            }
        }
    }

    public static func parseExplicitLimit(enabled: Bool, rawText: String) throws -> Int? {
        guard enabled else { return nil }
        let trimmed = rawText.trimmingCharacters(in: .whitespacesAndNewlines)
        guard let value = Int(trimmed), value >= 1 else {
            throw ValidationError.invalidExplicitLimit(trimmed)
        }
        return value
    }

    public static func wireMessages(systemPrompt: String, messages: [ChatWireMessage]) -> [[String: Any]] {
        var wire: [[String: Any]] = []
        let trimmedSystem = systemPrompt.trimmingCharacters(in: .whitespacesAndNewlines)
        if !trimmedSystem.isEmpty {
            wire.append(["role": "system", "content": systemPrompt])
        }
        for message in messages {
            var item: [String: Any] = ["role": message.role, "content": message.content]
            if message.role == "assistant", let reasoning = message.reasoningContent, !reasoning.isEmpty {
                item["reasoning_content"] = reasoning
            }
            wire.append(item)
        }
        return wire
    }

    /// Build a `/v1/chat/completions` JSON payload. When `maxOutputTokens` is `nil` (Unlimited),
    /// no `max_tokens` or `max_completion_tokens` field is emitted so the engine uses uncapped generation.
    public static func payload(model: String,
                               messages: [[String: Any]],
                               reasoning: ChatReasoningEffort,
                               maxOutputTokens: Int?,
                               stream: Bool = true) -> [String: Any] {
        var body: [String: Any] = [
            "model": model,
            "messages": messages,
            "stream": stream,
            "reasoning_effort": reasoning.rawValue,
        ]
        if stream {
            body["stream_options"] = ["include_usage": true]
        }
        if let limit = maxOutputTokens, limit > 0 {
            body["max_completion_tokens"] = limit
        }
        return body
    }

    public static func makeURLRequest(port: Int,
                                      key: String?,
                                      sessionID: String,
                                      bodyData: Data,
                                      timeoutInterval: TimeInterval = 7200) -> URLRequest {
        var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
        request.httpMethod = "POST"
        request.timeoutInterval = timeoutInterval
        request.setValue("application/json", forHTTPHeaderField: "Content-Type")
        if let key, !key.isEmpty {
            request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization")
        }
        request.setValue(sessionID, forHTTPHeaderField: "X-Session-Id")
        request.httpBody = bodyData
        return request
    }
}

public enum ChatSSEParser {
    public static func parse(line: String) -> [ChatStreamEvent] {
        guard line.hasPrefix("data:") else { return [] }
        let raw = line.dropFirst(5).trimmingCharacters(in: .whitespacesAndNewlines)
        if raw.isEmpty { return [] }
        if raw == "[DONE]" { return [.done] }
        guard let data = raw.data(using: .utf8),
              let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            return []
        }
        var events: [ChatStreamEvent] = []
        if let err = object["error"] as? [String: Any],
           let message = err["message"] as? String, !message.isEmpty {
            events.append(.error(message))
        }
        if let choices = object["choices"] as? [[String: Any]], let first = choices.first {
            if let delta = first["delta"] as? [String: Any] {
                if let reasoning = delta["reasoning_content"] as? String, !reasoning.isEmpty {
                    events.append(.reasoning(reasoning))
                }
                if let content = delta["content"] as? String, !content.isEmpty {
                    events.append(.content(content))
                }
            }
            if let finish = first["finish_reason"] as? String, !finish.isEmpty {
                events.append(.finishReason(finish))
            }
        }
        if let usage = object["usage"] as? [String: Any] {
            let prompt = (usage["prompt_tokens"] as? NSNumber)?.intValue ?? 0
            let completion = (usage["completion_tokens"] as? NSNumber)?.intValue ?? 0
            let total = (usage["total_tokens"] as? NSNumber)?.intValue ?? (prompt + completion)
            let details = usage["prompt_tokens_details"] as? [String: Any]
            let cached = (details?["cached_tokens"] as? NSNumber)?.intValue
                ?? (usage["prompt_cache_hit_tokens"] as? NSNumber)?.intValue
                ?? 0
            events.append(.usage(ChatUsage(promptTokens: prompt, completionTokens: completion, totalTokens: total, cachedTokens: cached)))
        }
        return events
    }
}

public enum ChatHTTPErrorParser {
    public static func message(from data: Data, statusCode: Int) -> String {
        if let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
           let error = object["error"] as? [String: Any],
           let detail = error["message"] as? String, !detail.isEmpty {
            if let code = error["code"] as? String, !code.isEmpty {
                return "HTTP \(statusCode) (\(code)): \(detail)"
            }
            return "HTTP \(statusCode): \(detail)"
        }
        if let text = String(data: data, encoding: .utf8)?.trimmingCharacters(in: .whitespacesAndNewlines), !text.isEmpty {
            return "HTTP \(statusCode): \(text)"
        }
        return "Engine rejected the chat request (HTTP \(statusCode)). Check the engine logs."
    }
}
