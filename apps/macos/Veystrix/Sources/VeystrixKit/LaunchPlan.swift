import Foundation

/// How the engine process is started. Pure value type: building a plan starts nothing.
public struct LaunchPlan: Equatable, Sendable {
    public var executable: URL
    public var arguments: [String]
    public var environment: [String: String]
    public var workingDirectory: URL
    public var description: String
    public var keepMacAwake = false

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
    static let inheritedKeys = ["HOME", "USER", "LOGNAME", "TMPDIR", "LANG", "LC_ALL", "PATH", "SHELL", "VEYSTRIX_MODEL_ROOT"]

    public static func make(settings: AppSettings,
                            checkpoint: CheckpointReport?,
                            source: Source,
                            apiKey: String?,
                            parentEnvironment: [String: String] = ProcessInfo.processInfo.environment,
                            paths: Paths = Paths(),
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
            for file in ["models/flashnext/overnight.pyc", "veystrix/cli.pyc", "veystrix/session.pyc", "veystrix/model_router.pyc", "veystrix/engine_guard.pyc"] {
                let path = cwd.appendingPathComponent(file).path
                guard fileExists(path) else { throw PlanError.runtimeMissing(path) }
            }
        }

        // Nemotron's expert pool is sized from free memory when the setting is left at "none" (0 would disable it).
        let poolArgument = String(settings.decodePool.rawValue)
        var args = ["-m", "veystrix.model_router",
                    "--host", "127.0.0.1",                       // loopback only, always
                    "--port", String(settings.port),
                    "--pool-gb", poolArgument,
                    "--checkpoint", settings.activeCheckpointPath]
        args += ["--model", settings.selectedModel.rawValue]
        args += ["--flashnext-kv-cache", settings.flashNextKVCache.rawValue,
                 "--flashnext-memory-policy", "absolute-swap-only"]
        if settings.flashNextSpeed { args.append("--flashnext-speed") }
        if settings.supervisedRestarts { args.append("--overnight") }
        // The router forwards the shared cache location when either model starts.
        if !settings.cacheDirectory.isEmpty { args += ["--prefix-dir", settings.cacheDirectory] }

        var env: [String: String] = [:]
        for key in inheritedKeys { if let v = parentEnvironment[key] { env[key] = v } }
        env["PYTHONUNBUFFERED"] = "1"
        env["PYTHONDONTWRITEBYTECODE"] = "1"
        env["VEYSTRIX_MIN_BATTERY_PERCENT"] = String(settings.minimumBatteryPercent)
        env["VEYSTRIX_REQUIRE_AC"] = settings.allowBattery ? "0" : "1"
        env["VEYSTRIX_ALLOW_LOW_POWER"] = settings.allowLowPowerMode ? "1" : "0"
        env["FLASHNEXT_SWAP_CEILING_MIB"] = String(settings.swapLimitGiB * 1024)
        env["VEYSTRIX_PORT"] = String(settings.port)
        // Lifetime statistics: the engine appends one counts-only line per finished request here.
        env["VEYSTRIX_USAGE_LOG"] = UsageStore(paths: paths).requestsFile.path
        env["VEYSTRIX_CACHE_GB"] = String(settings.cacheQuotaGB)
        env["VEYSTRIX_NEMOTRON_CONTEXT"] = String(settings.nemotronContextTokens)
        env["VEYSTRIX_NEMOTRON_KV"] = settings.nemotronKVCache.rawValue
        for (prefix, options) in [("FLASHNEXT", settings.flashNextPerformance), ("VEYSTRIX_NEMOTRON", settings.nemotronPerformance)] {
            env["\(prefix)_PROMPT_LOOKUP"] = options.decoder == .promptLookup ? "1" : "0"
            env["\(prefix)_PREFILL_SEED_DECODE"] = options.seedExperts ? "1" : "0"
            env["\(prefix)_PREFIX_ASYNC_WRITE"] = options.backgroundSnapshots ? "1" : "0"
            env["\(prefix)_METAL_FMA_DEQUANT"] = options.affineFMA ? "1" : "0"
            env["\(prefix)_MLX_AUTO_BUFFER_CAPS"] = options.largerMetalBuffers ? "1" : "0"
        }
        env["VEYSTRIX_FLASHNEXT_NATIVE_MTP"] = settings.flashNextPerformance.decoder == .nativeMTP ? "on" : "off"
        env["VEYSTRIX_FLASHNEXT_OUTPUT_TOKENS"] = String(settings.flashNextPerformance.defaultOutputTokens)
        env["VEYSTRIX_FLASHNEXT_MTP_DEPTH"] = String(settings.flashNextPerformance.draftTokens)
        env["VEYSTRIX_NEMOTRON_NATIVE_MTP"] = settings.nemotronPerformance.decoder == .nativeMTP ? "1" : "0"
        env["VEYSTRIX_NEMOTRON_OUTPUT_TOKENS"] = String(settings.nemotronPerformance.defaultOutputTokens)
        env["VEYSTRIX_NEMOTRON_MTP_DEPTH"] = String(settings.nemotronPerformance.draftTokens)
        if !settings.nemotronMTPPath.isEmpty { env["VEYSTRIX_NEMOTRON_MTP_PATH"] = settings.nemotronMTPPath }
        // When memory is short the engine purges the macOS file cache by itself (conditional, never during a request;
        // needs the narrow `sudo -n /usr/sbin/purge` permission, otherwise it records that and backs off).
        env["FLASHNEXT_AUTO_PURGE"] = "1"
        // Choosing Nemotron and pressing Start (after the load confirmation) is the approval for real inference.
        if settings.selectedModel == .nemotron35 { env["VEYSTRIX_NEMOTRON_AUTHORIZED"] = "1" }
        env["VEYSTRIX_API_KEY"] = apiKey ?? "local"
        if settings.requireAPIKey, let apiKey { env["VEYSTRIX_SERVER_API_KEY"] = apiKey }
        if case .bundled(let root) = source { env["PYTHONPATH"] = root.appendingPathComponent("lib").path }

        return LaunchPlan(executable: python, arguments: args, environment: env, workingDirectory: cwd,
                          description: "Veystrix engine (\(note)) on 127.0.0.1:\(settings.port)",
                          keepMacAwake: settings.keepMacAwake && fileExists("/usr/bin/caffeinate"))
    }
}
