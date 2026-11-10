import Foundation

/// Thread-safe in-memory log ring plus an on-disk copy under ~/Library/Logs/Veystrix.
/// Lines are stored raw in the ring for the on-screen viewer only; anything exported goes through `Sanitizer`.
public final class LogStore: @unchecked Sendable {
    public struct Line: Identifiable, Equatable, Sendable {
        public enum Source: String, Sendable { case app, engine }
        public let id: Int
        public let date: Date
        public let source: Source
        public let text: String
    }

    private let lock = NSLock()
    private var lines: [Line] = []
    private var counter = 0
    private let capacity: Int
    private let fileURL: URL?
    private let maxFileBytes: Int
    private var handle: FileHandle?
    public var onChange: (@Sendable () -> Void)?

    public init(capacity: Int = 5000, fileURL: URL? = nil, maxFileBytes: Int = 5_000_000) {
        self.capacity = capacity
        self.fileURL = fileURL
        self.maxFileBytes = maxFileBytes
    }

    public func append(_ text: String, source: Line.Source) {
        let parts = text.split(whereSeparator: \.isNewline).map(String.init)
        guard !parts.isEmpty else { return }
        lock.lock()
        for part in parts {
            counter += 1
            lines.append(Line(id: counter, date: Date(), source: source, text: part))
        }
        if lines.count > capacity { lines.removeFirst(lines.count - capacity) }
        writeToDisk(parts, source: source)
        lock.unlock()
        onChange?()
    }

    public func snapshot() -> [Line] { lock.lock(); defer { lock.unlock() }; return lines }
    public func tail(_ n: Int) -> [Line] { lock.lock(); defer { lock.unlock() }; return Array(lines.suffix(n)) }
    public func clear() { lock.lock(); lines.removeAll(); lock.unlock(); onChange?() }

    // MARK: disk (owner-only, rotated once at maxFileBytes)
    private func writeToDisk(_ parts: [String], source: Line.Source) {
        guard let fileURL else { return }
        do {
            if handle == nil {
                try FileManager.default.createDirectory(at: fileURL.deletingLastPathComponent(), withIntermediateDirectories: true,
                                                        attributes: [.posixPermissions: 0o700])
                if !FileManager.default.fileExists(atPath: fileURL.path) {
                    FileManager.default.createFile(atPath: fileURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
                }
                handle = try FileHandle(forWritingTo: fileURL)
                try handle?.seekToEnd()
            }
            if let size = try handle?.offset(), Int(size) > maxFileBytes {
                try handle?.close(); handle = nil
                let old = fileURL.appendingPathExtension("1")
                try? FileManager.default.removeItem(at: old)
                try FileManager.default.moveItem(at: fileURL, to: old)
                FileManager.default.createFile(atPath: fileURL.path, contents: nil, attributes: [.posixPermissions: 0o600])
                handle = try FileHandle(forWritingTo: fileURL)
            }
            let stamp = ISO8601DateFormatter().string(from: Date())
            let blob = parts.map { "\(stamp) [\(source.rawValue)] \($0)\n" }.joined()
            try handle?.write(contentsOf: Data(blob.utf8))
        } catch {
            handle = nil   // logging must never take the app down
        }
    }
}

/// Recognises engine output that means a safety guard stopped or refused something, so the Dashboard can show it.
public enum SafetyStopDetector {
    static let markers = ["memory_admission_refused", "Nothing was loaded", "refused", "guard stop", "swap ceiling",
                          "memory guard", "safety stop", "Memory pressure", "watchdog"]
    public static func isSafetyStop(_ line: String) -> Bool {
        let lower = line.lowercased()
        return markers.contains { lower.contains($0.lowercased()) }
    }
}
