import Foundation

public enum WorkflowMode: String, CaseIterable, Identifiable, Sendable {
    case swarm, overnight
    public var id: Self { self }
    public var label: String { self == .swarm ? "Swarm" : "Overnight" }
}

public struct WorkflowEntry: Decodable, Identifiable, Sendable {
    public let id: String
    public let objective: String
    public let state: String
}

public struct WorkflowStatus: Decodable, Sendable {
    public struct Configuration: Decodable, Sendable {
        public let model: String
        public let executorModel: String?
    }
    public struct WorkItem: Decodable, Identifiable, Sendable {
        public struct Result: Decodable, Sendable {
            public let summary: String?
            public let reason: String?
            public let workspace: String?
            public let files: [String]?
            public let unavailableDependencies: [String]?
        }
        public let id: String
        public let objective: String
        public let role: String
        public let state: String
        public let attempt: Int
        public let dependsOn: [String]
        public let result: Result?

        public var finished: Bool { ["completed", "blocked", "failed", "cancelled", "skipped"].contains(state) }
    }

    public let objective: String
    public let state: String
    public let control: String
    public let runnerActive: Bool
    public let settings: Configuration
    public let tasks: [WorkItem]

    public var finished: Bool { ["completed", "blocked", "failed", "cancelled"].contains(state) }
    public var canPause: Bool { !finished && (state == "ready" || runnerActive) && control == "run" }
    public var canResume: Bool { !finished && !runnerActive && control != "cancel" }
    public var canCancel: Bool { !finished && control != "cancel" }
}

public struct WorkflowLaunch: Sendable {
    public let repository: URL
    public let objective: String
    public let mode: WorkflowMode
    public let model: String
    public let executorModel: String?
    public let seconds: Int
    public let agents: Int
    public let allowEdits: Bool
    public let mockFixture: URL?

    public init(repository: URL, objective: String, mode: WorkflowMode, model: String = "mock",
                executorModel: String? = nil, seconds: Int = 3600, agents: Int = 3,
                allowEdits: Bool = false, mockFixture: URL? = nil) {
        self.repository = repository
        self.objective = objective
        self.mode = mode
        self.model = model
        self.executorModel = executorModel
        self.seconds = seconds
        self.agents = agents
        self.allowEdits = allowEdits
        self.mockFixture = mockFixture
    }
}

public struct WorkflowClient: Sendable {
    public enum Action: String, Sendable { case pause, resume, cancel }

    public enum Failure: LocalizedError {
        case runtimeMissing
        case invalidWorkflowID
        case command(String)
        public var errorDescription: String? {
            switch self {
            case .runtimeMissing: return "Veystrix’s workflow runtime is unavailable."
            case .invalidWorkflowID: return "The selected workflow has an invalid directory name."
            case .command(let message): return message
            }
        }
    }

    private let source: LaunchPlan.Source?
    public let root: URL
    private let port: Int
    private let apiKey: String?

    public init(source: LaunchPlan.Source?, root: URL, port: Int, apiKey: String?) {
        self.source = source
        self.root = root
        self.port = port
        self.apiKey = apiKey
    }

    public func start(_ launch: WorkflowLaunch) async throws -> String {
        let id = UUID().uuidString.lowercased()
        try FileManager.default.createDirectory(at: root, withIntermediateDirectories: true,
                                                attributes: [.posixPermissions: 0o700])
        let path = try directory(id).path
        var args = ["start", path, "--repo", launch.repository.path, "--goal", launch.objective,
                    "--mode", launch.mode.rawValue, "--model", launch.model, "--port", String(port),
                    "--seconds", String(launch.seconds), "--agents", String(launch.agents), "--detach"]
        if let executor = launch.executorModel { args += ["--executor-model", executor] }
        if let fixture = launch.mockFixture { args += ["--mock", fixture.path] }
        else if launch.model == "mock" { args += ["--demo"] }
        if launch.allowEdits { args += ["--allow-edits"] }
        _ = try await run(args, useAPIKey: launch.model != "mock")
        return id
    }

    public func history() async throws -> [WorkflowEntry] {
        try await decode([WorkflowEntry].self, args: ["list", root.path])
    }

    public func status(_ id: String) async throws -> WorkflowStatus {
        try await decode(WorkflowStatus.self, args: ["status", directory(id).path])
    }

    public func perform(_ action: Action, on id: String) async throws {
        var args = [action.rawValue, try directory(id).path]
        var useAPIKey = false
        if action == .resume {
            args.append("--detach")
            useAPIKey = try await status(id).settings.model != "mock"
        }
        _ = try await run(args, useAPIKey: useAPIKey)
    }

    public func report(_ id: String) async throws -> String {
        let data = try await run(["report", directory(id).path])
        return String(decoding: data, as: UTF8.self)
    }

    public func directory(_ id: String) throws -> URL {
        guard !id.isEmpty, id != ".", id != "..", !id.contains("/"), !id.hasPrefix(".") else {
            throw Failure.invalidWorkflowID
        }
        let path = root.appendingPathComponent(id, isDirectory: true)
        let resolved = path.resolvingSymlinksInPath().standardizedFileURL
        let parent = root.resolvingSymlinksInPath().standardizedFileURL
        guard resolved.deletingLastPathComponent().path == parent.path, resolved.lastPathComponent == id else {
            throw Failure.invalidWorkflowID
        }
        return path
    }

    private func decode<T: Decodable>(_ type: T.Type, args: [String]) async throws -> T {
        let data = try await run(args)
        let decoder = JSONDecoder()
        decoder.keyDecodingStrategy = .convertFromSnakeCase
        return try decoder.decode(type, from: data)
    }

    private func run(_ args: [String], useAPIKey: Bool = false) async throws -> Data {
        try await Task.detached(priority: .utility) {
            let executable: URL
            let directory: URL
            switch source {
            case .bundled(let root):
                executable = root.appendingPathComponent("python/bin/python3")
                directory = root.appendingPathComponent("lib")
            case .developer(let root):
                executable = root.appendingPathComponent(".venv/bin/python")
                directory = root
            case nil:
                throw Failure.runtimeMissing
            }
            guard FileManager.default.isExecutableFile(atPath: executable.path) else { throw Failure.runtimeMissing }
            let process = Process()
            process.executableURL = executable
            process.currentDirectoryURL = directory
            process.arguments = ["-m", "veystrix.workflows"] + args
            let parent = ProcessInfo.processInfo.environment
            var environment = parent.filter { ["HOME", "PATH", "TMPDIR", "LANG", "LC_ALL"].contains($0.key) }
            environment["PYTHONPATH"] = directory.path
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            environment["PYTHONNOUSERSITE"] = "1"
            if useAPIKey { environment["VEYSTRIX_SERVER_API_KEY"] = apiKey }
            process.environment = environment
            process.standardInput = FileHandle.nullDevice
            let output = Pipe()
            process.standardOutput = output
            process.standardError = output
            try process.run()
            let data = output.fileHandleForReading.readDataToEndOfFile()
            process.waitUntilExit()
            guard process.terminationStatus == 0 else {
                let detail = String(decoding: data.prefix(2000), as: UTF8.self).trimmingCharacters(in: .whitespacesAndNewlines)
                throw Failure.command(detail.isEmpty ? "Workflow command failed (status \(process.terminationStatus))." : detail)
            }
            return data
        }.value
    }
}
