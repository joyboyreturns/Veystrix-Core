import Foundation

/// Removes secrets and personal paths from text before it is shown in a diagnostics export.
/// It is deliberately conservative: when in doubt, it redacts.
public struct Sanitizer: Sendable {
    public var homeDirectory: String
    public var extraSecrets: [String]

    public init(homeDirectory: String = NSHomeDirectory(), extraSecrets: [String] = []) {
        self.homeDirectory = homeDirectory
        self.extraSecrets = extraSecrets.filter { $0.count >= 4 }
    }

    // Order matters: specific token shapes first, then generic key=value forms.
    private static let patterns: [(NSRegularExpression, String)] = {
        let raw: [(String, String)] = [
            (#"(?i)\bBearer\s+[A-Za-z0-9._~+/=-]{8,}"#, "Bearer [REDACTED]"),
            (#"\bsk-[A-Za-z0-9_-]{16,}"#, "[REDACTED-KEY]"),
            (#"\bsk-ant-[A-Za-z0-9_-]{16,}"#, "[REDACTED-KEY]"),
            (#"\bgh[pousr]_[A-Za-z0-9]{20,}"#, "[REDACTED-KEY]"),
            (#"\bgithub_pat_[A-Za-z0-9_]{20,}"#, "[REDACTED-KEY]"),
            (#"\bhf_[A-Za-z0-9]{20,}"#, "[REDACTED-KEY]"),
            (#"\bAKIA[0-9A-Z]{16}\b"#, "[REDACTED-KEY]"),
            (#"\bAIza[0-9A-Za-z_-]{30,}"#, "[REDACTED-KEY]"),
            (#"\bxox[baprs]-[A-Za-z0-9-]{10,}"#, "[REDACTED-KEY]"),
            (#"\beyJ[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{10,}\.[A-Za-z0-9_-]{5,}"#, "[REDACTED-JWT]"),
            (#"(?i)\b([A-Z0-9_]*(?:API[_-]?KEY|SECRET|TOKEN|PASSWORD|PASSWD)[A-Z0-9_]*)\s*[=:]\s*["']?[^\s"',;]+"#, "$1=[REDACTED]"),
            (#"(?i)"(api[_-]?key|authorization|token|secret|password)"\s*:\s*"[^"]*""#, #""$1":"[REDACTED]""#),
            // prompt or message text that a verbose log might echo
            (#"(?i)"(content|prompt|text|input)"\s*:\s*"(?:[^"\\]|\\.){0,20000}""#, #""$1":"[PROMPT TEXT REMOVED]""#),
            (#"-----BEGIN [A-Z ]*PRIVATE KEY-----[\s\S]*?-----END [A-Z ]*PRIVATE KEY-----"#, "[REDACTED-PRIVATE-KEY]"),
        ]
        return raw.compactMap { pair in (try? NSRegularExpression(pattern: pair.0)).map { ($0, pair.1) } }
    }()

    public func clean(_ text: String) -> String {
        var s = text
        for secret in extraSecrets { s = s.replacingOccurrences(of: secret, with: "[REDACTED]") }
        for (regex, template) in Self.patterns {
            s = regex.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: template)
        }
        if !homeDirectory.isEmpty && homeDirectory != "/" {
            s = s.replacingOccurrences(of: homeDirectory, with: "~")
        }
        // any other user's home directory
        if let regex = try? NSRegularExpression(pattern: #"/Users/[^/\s"']+"#) {
            s = regex.stringByReplacingMatches(in: s, range: NSRange(s.startIndex..., in: s), withTemplate: "/Users/[user]")
        }
        return s
    }
}
