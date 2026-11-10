import Foundation

/// How the engine process is started. Pure value type: building a plan starts nothing.
public struct LaunchPlan: Equatable, Sendable {
    public var executable: URL
    public var arguments: [String]
    public var environment: [String: String]
    public var workingDirectory: URL
    public var description: String

    public enum Source: Equatable, Sendable {
        /// The engine runtime bundled inside the app: Contents/Resources/engine
        case bundled(engineRoot: URL)
        /// A source checkout with its own .venv (development).
        case developer(repoRoot: URL)
    }

    public enum PlanError: Error, Equatable, CustomStringConvertible, Sendable {
        case invalidSettings([String])
        case noCheckpoint
        case checkpointNotUsable(String)
        case runtimeMissing(String)
        public var description: String {
            switch self {
            case .invalidSettings(let p): return p.joined(separator: " ")
            case .noCheckpoint: return "Choose a model folder first (Model > Choose Folder…)."
            case .checkpointNotUsable(let why): return "The model folder cannot be used: \(why)"
            case .runtimeMissing(let what): return "The engine runtime was not found: \(what)"
            }
        }
    }

    /// Variables passed to the engine. Nothing else is inherited, so secrets in the user's shell
    /// (cloud API keys, tokens) never reach the engine or its tools.
    static let inheritedKeys = ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "PATH", "SHELL"]

    public static func make(settings: AppSettings,
                            checkpoint: CheckpointReport?,
                            source: Source,
                            apiKey: String?,
                            parentEnvironment: [String: String] = ProcessInfo.processInfo.environment,
                            fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) },
                            isExecutable: (String) -> Bool = { FileManager.default.isExecutableFile(atPath: $0) }) throws -> LaunchPlan {
        let problems = settings.validationProblems()
        guard problems.isEmpty else { throw PlanError.invalidSettings(problems) }
        guard !settings.activeCheckpointPath.isEmpty, let checkpoint else { throw PlanError.noCheckpoint }
        guard checkpoint.isUsable else {
            throw PlanError.checkpointNotUsable(checkpoint.notes.first ?? "it is incomplete or incompatible")
        }

        let python: URL, cwd: URL, note: String
        switch source {
        case .bundled(let root):
            python = root.appendingPathComponent("python/bin/python3")
            cwd = root.appendingPathComponent("lib")
            note = "bundled runtime"
        case .developer(let repo):
            python = repo.appendingPathComponent(".venv/bin/python")
            cwd = repo
            note = "developer checkout"
        }
        guard fileExists(python.path) else { throw PlanError.runtimeMissing(python.path) }
        guard isExecutable(python.path) else { throw PlanError.runtimeMissing("Python is not executable: \(python.path)") }
        guard fileExists(cwd.path) else { throw PlanError.runtimeMissing("working directory: \(cwd.path)") }
        if case .bundled = source {
            for file in ["models/flashnext/overnight.pyc", "veystrix/cli.pyc", "veystrix/session.pyc"] {
                let path = cwd.appendingPathComponent(file).path
                guard fileExists(path) else { throw PlanError.runtimeMissing(path) }
            }
        }

        var args = ["-m", "models.flashnext.overnight",
                    "--host", "127.0.0.1",                       // loopback only, always
                    "--port", String(settings.port),
                    "--pool-gb", String(settings.decodePool.rawValue),
                    "--checkpoint", settings.activeCheckpointPath]
        if settings.selectedModel == .nemotron35 { args += ["--model", ModelFamily.nemotron35.rawValue] }
        if settings.supervisedRestarts { args.append("--overnight") }
        // The prefix cache is a Flash-Next feature; the Nemotron engine has no use for the folder.
        if settings.selectedModel == .flashnext, !settings.cacheDirectory.isEmpty { args += ["--prefix-dir", settings.cacheDirectory] }

        var env: [String: String] = [:]
        for key in inheritedKeys { if let v = parentEnvironment[key] { env[key] = v } }
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["VEYSTRIX_MIN_BATTERY_PERCENT"] = String(settings.minimumBatteryPercent)
        env["VEYSTRIX_REQUIRE_AC"] = settings.allowBattery ? "0" : "1"
        env["VEYSTRIX_ALLOW_LOW_POWER"] = settings.allowLowPowerMode ? "1" : "0"
        env["VEYSTRIX_PORT"] = String(settings.port)
        env["VEYSTRIX_CACHE_GB"] = String(settings.cacheQuotaGB)
        // When memory is short the engine purges the macOS file cache by itself (conditional, never during a request;
        // needs the narrow `sudo -n /usr/sbin/purge` permission, otherwise it records that and backs off).
        env["FLASHNEXT_AUTO_PURGE"] = "1"
        // Choosing Nemotron and pressing Start (after the load confirmation) is the approval for real inference.
        if settings.selectedModel == .nemotron35 { env["VEYSTRIX_NEMOTRON_AUTHORIZED"] = "1" }
        env["VEYSTRIX_API_KEY"] = apiKey ?? "local"
        if settings.requireAPIKey, let apiKey { env["VEYSTRIX_SERVER_API_KEY"] = apiKey }
        if case .bundled(let root) = source { env["PYTHONPATH"] = root.appendingPathComponent("lib").path }

        var executable = python
        var finalArgs = args
        if settings.keepMacAwake, fileExists("/usr/bin/caffeinate") {
            // Same as serve.sh: no idle sleep (-i), no disk idle sleep (-m), no system sleep on AC (-s).
            executable = URL(fileURLWithPath: "/usr/bin/caffeinate")
            finalArgs = ["-ims", python.path] + args
        }
        return LaunchPlan(executable: executable, arguments: finalArgs, environment: env, workingDirectory: cwd,
                          description: "Veystrix engine (\(note)) on 127.0.0.1:\(settings.port)")
    }
}
