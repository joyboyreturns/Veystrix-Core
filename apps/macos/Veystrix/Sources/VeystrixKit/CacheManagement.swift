import Foundation

public struct CacheStatus: Decodable, Sendable {
    public struct Model: Decodable, Sendable {
        public let bytes: Int64
        public let snapshots: Int
        public let hits: Int?
        public let misses: Int?
        public let restoredTokens: Int?
    }

    public let root: String
    public let usedBytes: Int64
    public let quotaBytes: Int64
    public let freeBytes: Int64
    public let reserveBytes: Int64
    public let snapshots: Int
    public let models: [String: Model]
    public let healthy: Bool
    public let lastCleanup: CacheReclaim?
}

public struct CacheReclaim: Decodable, Sendable {
    public let requestedBytes: Int64
    public let reclaimedBytes: Int64
    public let shortfallBytes: Int64
    public let snapshots: Int
}

public enum CacheManagement {
    public enum Action: String, Sendable {
        case status, preview, reclaim, clear, quota
    }

    public static func run<Result: Decodable & Sendable>(
        _ action: Action, gigabytes: Int? = nil, settings: AppSettings, source: LaunchPlan.Source?
    ) async throws -> Result {
        try await Task.detached(priority: .utility) {
            let python: URL
            let workingDirectory: URL
            switch source {
            case .bundled(let root):
                python = root.appendingPathComponent("python/bin/python3")
                workingDirectory = root.appendingPathComponent("lib")
            case .developer(let repo):
                python = repo.appendingPathComponent(".venv/bin/python")
                workingDirectory = repo
            case nil:
                throw CocoaError(.fileNoSuchFile, userInfo: [NSLocalizedDescriptionKey: "The Veystrix runtime is unavailable."])
            }
            let process = Process()
            process.executableURL = python
            process.currentDirectoryURL = workingDirectory
            var args = ["-m", "veystrix.universal_cache"]
            if !settings.cacheDirectory.isEmpty { args += ["--directory", settings.cacheDirectory] }
            args.append(action.rawValue)
            if let gigabytes { args.append(String(gigabytes)) }
            process.arguments = args
            var env = ProcessInfo.processInfo.environment.filter { ["HOME", "PATH", "TMPDIR", "LANG"].contains($0.key) }
            env["VEYSTRIX_CACHE_GB"] = String(settings.cacheQuotaGB)
            env["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = env
            let output = Pipe()
            process.standardOutput = output
            process.standardError = output
            try process.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                let detail = String(decoding: data, as: UTF8.self).split(separator: "\n").last.map(String.init)
                throw CocoaError(.fileReadUnknown, userInfo: [NSLocalizedDescriptionKey: detail ?? "Cache command failed."])
            }
            let decoder = JSONDecoder()
            decoder.keyDecodingStrategy = .convertFromSnakeCase
            return try decoder.decode(Result.self, from: data)
        }.value
    }
}
