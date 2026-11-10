import Foundation

/// Lifetime usage statistics. The engine appends one JSON line per finished request (counts and timings only, never
/// text) to `usage/requests.jsonl`; the app adds "loaded time" per model while an engine is ready. Both live in
/// Application Support, so totals survive engine restarts, app restarts and updates. Nothing is ever deleted except by
/// the explicit "Reset statistics" action.

public struct UsageRecord: Decodable, Equatable, Sendable {
    public var time: Date
    public var model: String
    public var prompt: Int
    public var cached: Int
    public var uncached: Int
    public var completion: Int
    public var wallSeconds: Double
    public var prefillSeconds: Double
    public var decodeSeconds: Double
    public var warm: Bool

    enum CodingKeys: String, CodingKey {
        case ts, model, prompt, cached, uncached, completion, warm
        case wallSeconds = "wall_s", prefillSeconds = "prefill_s", decodeSeconds = "decode_s"
    }

    public init(time: Date, model: String, prompt: Int, cached: Int, completion: Int, wallSeconds: Double = 0,
                prefillSeconds: Double = 0, decodeSeconds: Double = 0, warm: Bool = false) {
        self.time = time; self.model = model; self.prompt = prompt; self.cached = cached
        self.uncached = max(0, prompt - cached); self.completion = completion
        self.wallSeconds = wallSeconds; self.prefillSeconds = prefillSeconds; self.decodeSeconds = decodeSeconds
        self.warm = warm
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        time = Date(timeIntervalSince1970: try c.decode(Double.self, forKey: .ts))
        model = try c.decodeIfPresent(String.self, forKey: .model) ?? "unknown"
        prompt = max(0, try c.decodeIfPresent(Int.self, forKey: .prompt) ?? 0)
        cached = max(0, try c.decodeIfPresent(Int.self, forKey: .cached) ?? 0)
        uncached = max(0, try c.decodeIfPresent(Int.self, forKey: .uncached) ?? max(0, prompt - cached))
        completion = max(0, try c.decodeIfPresent(Int.self, forKey: .completion) ?? 0)
        wallSeconds = max(0, try c.decodeIfPresent(Double.self, forKey: .wallSeconds) ?? 0)
        prefillSeconds = max(0, try c.decodeIfPresent(Double.self, forKey: .prefillSeconds) ?? 0)
        decodeSeconds = max(0, try c.decodeIfPresent(Double.self, forKey: .decodeSeconds) ?? 0)
        warm = try c.decodeIfPresent(Bool.self, forKey: .warm) ?? false
    }
}

public enum UsageRange: String, CaseIterable, Identifiable, Sendable {
    case allTime = "All time", today = "Today", week = "Last 7 days", month = "Last 30 days"
    public var id: String { rawValue }

    /// First instant included in the range; nil means "since the beginning".
    public func start(now: Date, calendar: Calendar = .current) -> Date? {
        let today = calendar.startOfDay(for: now)
        switch self {
        case .allTime: return nil
        case .today: return today
        case .week: return calendar.date(byAdding: .day, value: -6, to: today)
        case .month: return calendar.date(byAdding: .day, value: -29, to: today)
        }
    }
}

public struct ModelUsage: Identifiable, Equatable, Sendable {
    public var model: String
    public var id: String { model }
    public var requests = 0
    public var inputTokens = 0          // everything the client sent, cached or not
    public var cachedTokens = 0         // part of the input served from the cache
    public var uncachedTokens = 0       // part of the input that had to be computed
    public var outputTokens = 0
    public var activeSeconds = 0.0      // time spent answering requests
    public var prefillSeconds = 0.0
    public var decodeSeconds = 0.0
    public var loadedSeconds = 0.0      // time the model was loaded and ready
    // Rates use only requests that reported timings, so a record without timings cannot inflate them.
    var decodeTimedTokens = 0
    var prefillTimedTokens = 0

    public init(model: String) { self.model = model }

    public var totalTokens: Int { inputTokens + outputTokens }
    public var cacheHitRate: Double? { inputTokens > 0 ? Double(cachedTokens) / Double(inputTokens) : nil }
    public var decodeTokensPerSecond: Double? { decodeSeconds > 0 && decodeTimedTokens > 0 ? Double(decodeTimedTokens) / decodeSeconds : nil }
    public var prefillTokensPerSecond: Double? { prefillSeconds > 0 && prefillTimedTokens > 0 ? Double(prefillTimedTokens) / prefillSeconds : nil }
    public var isEmpty: Bool { requests == 0 && loadedSeconds == 0 }

    mutating func add(_ r: UsageRecord) {
        requests += 1; inputTokens += r.prompt; cachedTokens += min(r.cached, r.prompt); uncachedTokens += r.uncached
        outputTokens += r.completion; activeSeconds += r.wallSeconds
        prefillSeconds += r.prefillSeconds; decodeSeconds += r.decodeSeconds
        if r.decodeSeconds > 0 { decodeTimedTokens += r.completion }
        if r.prefillSeconds > 0 { prefillTimedTokens += r.uncached }
    }
    mutating func merge(_ o: ModelUsage) {
        requests += o.requests; inputTokens += o.inputTokens; cachedTokens += o.cachedTokens
        uncachedTokens += o.uncachedTokens; outputTokens += o.outputTokens; activeSeconds += o.activeSeconds
        prefillSeconds += o.prefillSeconds; decodeSeconds += o.decodeSeconds; loadedSeconds += o.loadedSeconds
        decodeTimedTokens += o.decodeTimedTokens; prefillTimedTokens += o.prefillTimedTokens
    }
}

public struct UsageSummary: Equatable, Sendable {
    public var models: [ModelUsage]          // most active first
    public var total: ModelUsage
    public var since: Date?                  // first record or loaded-time day inside the range
    public var warmupRequestsExcluded: Int
}

public enum UsageStats {
    public static func displayName(model: String) -> String {
        let m = model.lowercased()
        if m.contains("nemotron") { return "Nemotron 3.5 Lightning" }
        if m.contains("flashnext") || m.contains("flash-next") || m.contains("qwen") { return "Qwen 3.8 Flash-Next" }
        return model
    }

    /// `loaded`: day key ("yyyy-MM-dd", local time) -> model -> seconds the model was loaded and ready.
    public static func summarize(records: [UsageRecord], loaded: [String: [String: Double]] = [:], range: UsageRange,
                                 includeWarmups: Bool = false, now: Date = Date(), calendar: Calendar = .current) -> UsageSummary {
        let start = range.start(now: now, calendar: calendar)
        var perModel: [String: ModelUsage] = [:]
        var excluded = 0
        var first: Date?
        for r in records {
            if let start, r.time < start { continue }
            if r.time > now.addingTimeInterval(86_400) { continue }       // clock skew guard: ignore far-future rows
            if r.warm && !includeWarmups { excluded += 1; continue }
            perModel[r.model, default: ModelUsage(model: r.model)].add(r)
            if first == nil || r.time < first! { first = r.time }
        }
        let startKey = start.map { dayKey($0, calendar: calendar) }
        for (day, models) in loaded {
            if let startKey, day < startKey { continue }
            if let date = date(fromDayKey: day, calendar: calendar) {
                if date > now.addingTimeInterval(86_400) { continue }
                if first == nil || date < first! { first = date }
            }
            for (model, seconds) in models where seconds > 0 {
                perModel[model, default: ModelUsage(model: model)].loadedSeconds += seconds
            }
        }
        var total = ModelUsage(model: "All models")
        let sorted = perModel.values.sorted {
            ($0.activeSeconds, $0.loadedSeconds, $0.model) > ($1.activeSeconds, $1.loadedSeconds, $1.model)
        }
        for u in sorted { total.merge(u) }
        return UsageSummary(models: sorted, total: total, since: first, warmupRequestsExcluded: excluded)
    }

    public static func dayKey(_ date: Date, calendar: Calendar = .current) -> String {
        let c = calendar.dateComponents([.year, .month, .day], from: date)
        return String(format: "%04d-%02d-%02d", c.year ?? 1970, c.month ?? 1, c.day ?? 1)
    }

    static func date(fromDayKey key: String, calendar: Calendar = .current) -> Date? {
        let parts = key.split(separator: "-").compactMap { Int($0) }
        guard parts.count == 3 else { return nil }
        return calendar.date(from: DateComponents(year: parts[0], month: parts[1], day: parts[2]))
    }

    // MARK: formatting
    public static func count(_ n: Int) -> String {
        let f = NumberFormatter(); f.numberStyle = .decimal; f.groupingSeparator = ","
        return f.string(from: NSNumber(value: n)) ?? String(n)
    }
    public static func compact(_ n: Int) -> String {
        switch n {
        case ..<10_000: return count(n)
        case ..<1_000_000: return String(format: "%.1fK", Double(n) / 1_000)
        case ..<1_000_000_000: return String(format: "%.2fM", Double(n) / 1_000_000)
        default: return String(format: "%.2fB", Double(n) / 1_000_000_000)
        }
    }
    public static func duration(_ seconds: Double) -> String {
        let s = Int(seconds.rounded())
        if s < 60 { return "\(s)s" }
        if s < 3600 { return "\(s / 60)m \(s % 60)s" }
        if s < 86_400 { return "\(s / 3600)h \((s % 3600) / 60)m" }
        return "\(s / 86_400)d \((s % 86_400) / 3600)h"
    }
}

/// Reads and resets the on-disk statistics. Value type: safe to use from a background task.
public struct UsageStore: Sendable {
    public let directory: URL
    public init(paths: Paths = Paths()) { directory = paths.applicationSupport.appendingPathComponent("usage", isDirectory: true) }
    public init(directory: URL) { self.directory = directory }

    public var requestsFile: URL { directory.appendingPathComponent("requests.jsonl") }
    public var loadedTimeFile: URL { directory.appendingPathComponent("loaded-time.json") }

    /// Current file plus every rotated archive (`requests.<n>.jsonl`), oldest first. Malformed lines are skipped.
    public func requestFiles() -> [URL] {
        let names = (try? FileManager.default.contentsOfDirectory(atPath: directory.path)) ?? []
        return names.filter { $0.hasPrefix("requests") && $0.hasSuffix(".jsonl") }.sorted().map { directory.appendingPathComponent($0) }
    }

    public func loadRecords() -> [UsageRecord] {
        let decoder = JSONDecoder()
        var out: [UsageRecord] = []
        for file in requestFiles() {
            guard let data = try? Data(contentsOf: file, options: .mappedIfSafe) else { continue }
            for line in data.split(separator: UInt8(ascii: "\n")) where !line.isEmpty {
                if let r = try? decoder.decode(UsageRecord.self, from: Data(line)) { out.append(r) }
            }
        }
        return out
    }

    public func loadLoadedTime() -> [String: [String: Double]] {
        guard let data = try? Data(contentsOf: loadedTimeFile),
              let file = try? JSONDecoder().decode(LoadedTimeFile.self, from: data) else { return [:] }
        return file.days
    }

    /// Cheap change detector so the page does not re-read large files when nothing was written.
    public func signature() -> String {
        var parts: [String] = []
        for url in requestFiles() + [loadedTimeFile] {
            let a = try? FileManager.default.attributesOfItem(atPath: url.path)
            parts.append("\(url.lastPathComponent):\((a?[.size] as? Int) ?? -1):\((a?[.modificationDate] as? Date)?.timeIntervalSince1970 ?? 0)")
        }
        return parts.joined(separator: "|")
    }

    /// Explicit user action only.
    public func reset() throws {
        for url in requestFiles() + [loadedTimeFile] where FileManager.default.fileExists(atPath: url.path) {
            try FileManager.default.removeItem(at: url)
        }
    }

    struct LoadedTimeFile: Codable { var days: [String: [String: Double]] }
}

/// Accumulates "model loaded and ready" seconds per day and model. Written by the app while the engine is healthy.
public final class LoadedTimeRecorder: @unchecked Sendable {
    private let store: UsageStore
    private let lock = NSLock()
    private var pending: [String: Double] = [:]          // model -> seconds not yet written
    private var lastSample: Date?
    private let flushEvery: Double
    private let maxGap: Double

    /// `maxGap`: a gap longer than this between two samples (app asleep, engine unreachable) is not counted.
    public init(store: UsageStore, flushEvery: Double = 20, maxGap: Double = 30) {
        self.store = store; self.flushEvery = flushEvery; self.maxGap = maxGap
    }

    public func sample(model: String?, now: Date = Date(), calendar: Calendar = .current) {
        lock.lock(); defer { lock.unlock() }
        defer { lastSample = now }
        guard let model, !model.isEmpty, let last = lastSample else { return }
        let gap = now.timeIntervalSince(last)
        guard gap > 0, gap <= maxGap else { return }
        pending[model, default: 0] += gap
        if pending.values.reduce(0, +) >= flushEvery { flushLocked(now: now, calendar: calendar) }
    }

    /// Call when the engine stops or becomes unreachable, so the next healthy sample does not count the gap.
    public func interrupt(now: Date = Date(), calendar: Calendar = .current) {
        lock.lock(); defer { lock.unlock() }
        flushLocked(now: now, calendar: calendar)
        lastSample = nil
    }

    public func flush(now: Date = Date(), calendar: Calendar = .current) {
        lock.lock(); defer { lock.unlock() }
        flushLocked(now: now, calendar: calendar)
    }

    private func flushLocked(now: Date, calendar: Calendar) {
        guard !pending.isEmpty else { return }
        var days = store.loadLoadedTime()
        let key = UsageStats.dayKey(now, calendar: calendar)
        for (model, seconds) in pending { days[key, default: [:]][model, default: 0] += seconds }
        do {
            try FileManager.default.createDirectory(at: store.directory, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            let data = try JSONEncoder().encode(UsageStore.LoadedTimeFile(days: days))
            try data.write(to: store.loadedTimeFile, options: .atomic)
            pending.removeAll()
        } catch {
            // Keep the pending seconds and retry at the next flush; statistics must never disturb the engine.
        }
    }
}
