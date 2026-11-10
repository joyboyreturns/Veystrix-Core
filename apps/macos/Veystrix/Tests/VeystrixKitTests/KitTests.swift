import Testing
import Foundation
@testable import VeystrixKit

// MARK: helpers
private func makeCheckpoint(in dir: URL, modelType: String = "qwen4_exp", shards: [String] = ["model-00001-of-00002.safetensors", "model-00002-of-00002.safetensors"],
                            createShards: Bool = true, extraWeightMapValue: String? = nil) throws {
    try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true)
    try Data(#"{"model_type":"\#(modelType)"}"#.utf8).write(to: dir.appendingPathComponent("config.json"))
    try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer.json"))
    try Data("{}".utf8).write(to: dir.appendingPathComponent("tokenizer_config.json"))
    var map: [String: String] = [:]
    for (i, s) in shards.enumerated() { map["t\(i)"] = s }
    if let extra = extraWeightMapValue { map["bad"] = extra }
    let index = try JSONSerialization.data(withJSONObject: ["weight_map": map])
    try index.write(to: dir.appendingPathComponent("model.safetensors.index.json"))
    if createShards { for s in shards { try Data(repeating: 0, count: 1024).write(to: dir.appendingPathComponent(s)) } }
}

private func tempDir() -> URL {
    let u = FileManager.default.temporaryDirectory.appendingPathComponent("veystrix-test-\(UUID().uuidString)")
    try? FileManager.default.createDirectory(at: u, withIntermediateDirectories: true)
    return u
}

// MARK: checkpoint validation (never reads weights)
@Suite struct CheckpointTests {
    @Test func completeCheckpointIsCompatible() throws {
        let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d)
        let r = CheckpointValidator.validate(directory: d)
        #expect(r.verdict == .compatible); #expect(r.shardCount == 2); #expect(r.totalBytes == 2048); #expect(r.isUsable)
    }
    @Test func missingShardIsIncomplete() throws {
        let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d)
        try FileManager.default.removeItem(at: d.appendingPathComponent("model-00002-of-00002.safetensors"))
        let r = CheckpointValidator.validate(directory: d)
        #expect(r.verdict == .incomplete); #expect(r.missingFiles == ["model-00002-of-00002.safetensors"]); #expect(!r.isUsable)
    }
    @Test func unsupportedArchitectureIsIncompatible() throws {
        let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d, modelType: "llama")
        #expect(CheckpointValidator.validate(directory: d).verdict == .incompatible)
    }
    @Test func nestedTextConfigModelTypeIsAccepted() throws {
        let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d)
        try Data(#"{"model_type":"wrapper","text_config":{"model_type":"qwen4_exp_text"}}"#.utf8).write(to: d.appendingPathComponent("config.json"))
        #expect(CheckpointValidator.validate(directory: d).verdict == .compatible)
    }
    @Test func pathTraversalShardNamesAreRefused() throws {
        for bad in ["../evil.safetensors", "/abs/evil.safetensors", ".hidden.safetensors", "a\\b.safetensors", "notweights.bin"] {
            let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d, extraWeightMapValue: bad)
            #expect(CheckpointValidator.validate(directory: d).verdict == .incomplete, "\(bad)")
        }
    }
    @Test func missingFolderIsNotFound() {
        #expect(CheckpointValidator.validate(directory: URL(fileURLWithPath: "/nonexistent/\(UUID())")).verdict == .notFound)
    }
    @Test func validationDoesNotReadWeightFiles() throws {
        // A shard that cannot be read must not matter: only its size is taken.
        let d = tempDir().appendingPathComponent("ckpt"); try makeCheckpoint(in: d)
        let shard = d.appendingPathComponent("model-00001-of-00002.safetensors")
        try FileManager.default.setAttributes([.posixPermissions: 0o000], ofItemAtPath: shard.path)
        defer { try? FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: shard.path) }
        #expect(CheckpointValidator.validate(directory: d).verdict == .compatible)
    }
}

// MARK: settings
@Suite struct SettingsTests {
    @Test func defaultsAreSafe() {
        let s = AppSettings()
        #expect(s.port == 8080); #expect(s.decodePool == .none); #expect(s.confirmBeforeLoading)
        #expect(!s.checkForUpdatesAutomatically); #expect(s.validationProblems().isEmpty)
    }
    @Test func roundTripAndOwnerOnlyPermissions() throws {
        let paths = Paths(home: tempDir()); let store = SettingsStore(paths: paths)
        var s = AppSettings(); s.port = 9001; s.decodePool = .twoGB; s.checkpointPath = "/models/x"
        s.allowBattery = false; s.minimumBatteryPercent = 35; s.allowLowPowerMode = true
        try store.save(s)
        #expect(store.load() == s)
        let attrs = try FileManager.default.attributesOfItem(atPath: paths.settingsFile.path)
        #expect((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }
    @Test func oldAndPartialSettingsFilesStillLoad() throws {
        let paths = Paths(home: tempDir()); try paths.ensureDirectories()
        try Data(#"{"port": 9100, "futureField": true}"#.utf8).write(to: paths.settingsFile)
        let s = SettingsStore(paths: paths).load()
        #expect(s.port == 9100); #expect(s.decodePool == .none); #expect(s.cacheQuotaGB == 50)
        #expect(s.allowBattery); #expect(s.minimumBatteryPercent == 25); #expect(!s.allowLowPowerMode)
    }
    @Test func damagedSettingsAreKeptAndDefaultsUsed() throws {
        let paths = Paths(home: tempDir()); try paths.ensureDirectories()
        try Data("not json".utf8).write(to: paths.settingsFile)
        let s = SettingsStore(paths: paths).load()
        #expect(s == AppSettings())
        #expect(FileManager.default.fileExists(atPath: paths.settingsFile.appendingPathExtension("damaged").path))
    }
    @Test func invalidPortAndQuotaAreReported() {
        var s = AppSettings(); s.port = 80; s.cacheQuotaGB = 0
        #expect(s.validationProblems().count == 2)
    }
    @Test func settingsSurviveAnUpdateBecauseTheyLiveInApplicationSupport() {
        let p = Paths(home: URL(fileURLWithPath: "/Users/someone"))
        #expect(p.settingsFile.path == "/Users/someone/Library/Application Support/Veystrix/settings.json")
        #expect(p.defaultCacheDirectory.path == "/Users/someone/.flashnext/prefix-cache")   // existing caches are reused
    }
}

// MARK: launch plan
@Suite struct LaunchPlanTests {
    private func good(_ tmp: URL) throws -> (AppSettings, CheckpointReport) {
        let d = tmp.appendingPathComponent("ckpt"); try makeCheckpoint(in: d)
        var s = AppSettings(); s.checkpointPath = d.path
        return (s, CheckpointValidator.validate(directory: d))
    }
    private let exists: (String) -> Bool = { _ in true }

    @Test func tellsTheEngineWhereToWriteLifetimeStatistics() throws {
        let (s, c) = try good(tempDir())
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil,
                                       fileExists: exists, isExecutable: exists)
        #expect(plan.environment["VEYSTRIX_USAGE_LOG"]?.hasSuffix("Veystrix/usage/requests.jsonl") == true)
    }

    @Test func buildsLoopbackOnlyPlanWithoutStartingAnything() throws {
        let (s, c) = try good(tempDir())
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: "local",
                                       parentEnvironment: ["HOME": "/Users/x", "PATH": "/usr/bin", "OPENAI_API_KEY": "sk-secret", "ANTHROPIC_API_KEY": "sk-ant-secret", "HF_TOKEN": "hf_x"],
                                       fileExists: exists, isExecutable: exists)
        #expect(plan.arguments.contains("models.flashnext.overnight"))
        let i = try #require(plan.arguments.firstIndex(of: "--host")); #expect(plan.arguments[i + 1] == "127.0.0.1")
        #expect(plan.arguments.contains("--overnight")); #expect(plan.arguments.contains("--checkpoint"))
        #expect(plan.environment["OPENAI_API_KEY"] == nil); #expect(plan.environment["ANTHROPIC_API_KEY"] == nil)
        #expect(plan.environment["HF_TOKEN"] == nil); #expect(plan.environment["HOME"] == "/Users/x")
        #expect(plan.environment["VEYSTRIX_PORT"] == "8080")
        #expect(plan.environment["FLASHNEXT_AUTO_PURGE"] == "1")   // memory short -> the engine purges the file cache by itself
    }
    @Test func powerSettingsReachTheEngine() throws {
        var (s, c) = try good(tempDir())
        s.allowBattery = false; s.minimumBatteryPercent = 40; s.allowLowPowerMode = true
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil, fileExists: exists, isExecutable: exists)
        #expect(plan.environment["VEYSTRIX_REQUIRE_AC"] == "1")
        #expect(plan.environment["VEYSTRIX_MIN_BATTERY_PERCENT"] == "40")
        #expect(plan.environment["VEYSTRIX_ALLOW_LOW_POWER"] == "1")
    }
    @Test func keepAwakeWrapsWithCaffeinateLikeServeSh() throws {
        var (s, c) = try good(tempDir()); s.keepMacAwake = true
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil, fileExists: exists, isExecutable: exists)
        #expect(plan.executable.path == "/usr/bin/caffeinate"); #expect(plan.arguments.first == "-ims")
        s.keepMacAwake = false
        let plain = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil, fileExists: exists, isExecutable: exists)
        #expect(plain.executable.path == "/repo/.venv/bin/python")
    }
    @Test func requireKeyPassesServerKeyOnlyWhenEnabled() throws {
        var (s, c) = try good(tempDir())
        let off = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/r")), apiKey: "vx-abc", fileExists: exists, isExecutable: exists)
        #expect(off.environment["VEYSTRIX_SERVER_API_KEY"] == nil)
        s.requireAPIKey = true
        let on = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/r")), apiKey: "vx-abc", fileExists: exists, isExecutable: exists)
        #expect(on.environment["VEYSTRIX_SERVER_API_KEY"] == "vx-abc")
        #expect(!on.arguments.joined(separator: " ").contains("vx-abc"))      // never on the command line
    }
    @Test func refusesWithoutAUsableCheckpoint() throws {
        let s = AppSettings()
        #expect(throws: LaunchPlan.PlanError.noCheckpoint) {
            try LaunchPlan.make(settings: s, checkpoint: nil, source: .developer(repoRoot: URL(fileURLWithPath: "/r")), apiKey: nil, fileExists: exists, isExecutable: exists)
        }
        var s2 = AppSettings(); s2.checkpointPath = "/x"
        let bad = CheckpointReport(verdict: .incomplete, path: "/x", notes: ["missing shards"])
        #expect(throws: LaunchPlan.PlanError.checkpointNotUsable("missing shards")) {
            try LaunchPlan.make(settings: s2, checkpoint: bad, source: .developer(repoRoot: URL(fileURLWithPath: "/r")), apiKey: nil, fileExists: exists, isExecutable: exists)
        }
    }
    @Test func refusesWhenRuntimeIsMissing() throws {
        let (s, c) = try good(tempDir())
        #expect(throws: LaunchPlan.PlanError.self) {
            try LaunchPlan.make(settings: s, checkpoint: c, source: .bundled(engineRoot: URL(fileURLWithPath: "/nope")), apiKey: nil, fileExists: { _ in false })
        }
    }
    @Test func bundledPlanRejectsNonExecutableAndMissingResources() throws {
        let (s, c) = try good(tempDir())
        let source = LaunchPlan.Source.bundled(engineRoot: URL(fileURLWithPath: "/Applications/App With Spaces.app/Contents/Resources/engine"))
        #expect(throws: LaunchPlan.PlanError.self) {
            try LaunchPlan.make(settings: s, checkpoint: c, source: source, apiKey: nil, fileExists: exists, isExecutable: { _ in false })
        }
        #expect(throws: LaunchPlan.PlanError.self) {
            try LaunchPlan.make(settings: s, checkpoint: c, source: source, apiKey: nil,
                                fileExists: { !$0.hasSuffix("overnight.pyc") }, isExecutable: exists)
        }
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: source, apiKey: nil, fileExists: exists, isExecutable: exists)
        #expect(plan.environment["PYTHONPATH"] == "/Applications/App With Spaces.app/Contents/Resources/engine/lib")
    }
    @Test func customCacheDirectoryIsPassedAsPrefixDir() throws {
        var (s, c) = try good(tempDir()); s.cacheDirectory = "/Volumes/Fast/cache"
        let plan = try LaunchPlan.make(settings: s, checkpoint: c, source: .developer(repoRoot: URL(fileURLWithPath: "/r")), apiKey: nil, fileExists: exists, isExecutable: exists)
        let i = try #require(plan.arguments.firstIndex(of: "--prefix-dir")); #expect(plan.arguments[i + 1] == "/Volumes/Fast/cache")
    }
    @Test func locatorPrefersBundledRuntime() {
        let res = URL(fileURLWithPath: "/App/Contents/Resources")
        let src = EngineLocator.resolve(settings: AppSettings(), resourceURL: res, environment: [:], fileExists: { $0.hasSuffix("engine/python/bin/python3") })
        guard case .bundled(let root)? = src else { Issue.record("expected the bundled runtime"); return }
        #expect(root.standardizedFileURL.path == "/App/Contents/Resources/engine")
        #expect(EngineLocator.resolve(settings: AppSettings(), resourceURL: nil, environment: [:], fileExists: { _ in false }) == nil)
    }
}

// MARK: engine controller (scripted fakes: no process, no network, no model)
final class FakeProcess: EngineProcess, @unchecked Sendable {
    var running = true; var signals: [String] = []; var ignoreInterrupt = false
    let onExit: @Sendable (Int32) -> Void
    init(onExit: @escaping @Sendable (Int32) -> Void) { self.onExit = onExit }
    var isRunning: Bool { running }
    func interrupt() { signals.append("INT"); if !ignoreInterrupt { exit(0) } }
    func terminate() { signals.append("TERM"); exit(15) }
    func kill() { signals.append("KILL"); exit(9) }
    func exit(_ code: Int32) { running = false; onExit(code) }
}
final class FakeLauncher: EngineProcessLauncher, @unchecked Sendable {
    var launched = 0; var last: FakeProcess?; var ignoreInterrupt = false; var failWith: Error?
    func launch(_ plan: LaunchPlan, onOutput: @escaping @Sendable (String) -> Void, onExit: @escaping @Sendable (Int32) -> Void) throws -> EngineProcess {
        if let failWith { throw failWith }
        launched += 1
        let p = FakeProcess(onExit: onExit); p.ignoreInterrupt = ignoreInterrupt; last = p
        onOutput("engine: loading\n")
        return p
    }
}
struct ScriptedHealth: HealthChecking {
    let script: @Sendable () -> HealthClient.Outcome
    func check() async -> HealthClient.Outcome { script() }
    func models() async -> ModelsReport? { nil }
}

@Suite @MainActor struct EngineControllerTests {
    private func ready() throws -> (AppSettings, CheckpointReport, URL) {
        let tmp = tempDir(); let d = tmp.appendingPathComponent("c"); try makeCheckpoint(in: d)
        var s = AppSettings(); s.checkpointPath = d.path
        return (s, CheckpointValidator.validate(directory: d), tmp)
    }
    private func src() -> LaunchPlan.Source { .developer(repoRoot: URL(fileURLWithPath: "/usr")) }   // /usr/.venv missing -> overridden below

    @Test func creatingTheControllerStartsNothing() {
        let launcher = FakeLauncher()
        let c = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot })
        #expect(c.state == .stopped); #expect(launcher.launched == 0)
    }
    @Test func lowBatteryRefusesTheStartBeforeAnythingLaunches() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher()
        let low = MemorySnapshot(physicalBytes: 16 << 30, availableBytes: 8 << 30, swapUsedBytes: 0, swapTotalBytes: 0,
                                 onACPower: false, batteryPercent: 24, lowPowerMode: false, cacheVolumeFreeBytes: nil)
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { low })
        do {
            try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: true, source: self.src())
            Issue.record("the start should have been refused")
        } catch let error as EngineController.StartError {
            guard case .batteryTooLow(let message) = error else { Issue.record("wrong error \(error)"); return }
            #expect(message.contains("24%"))
        }
        #expect(launcher.launched == 0); #expect(ctl.state == .stopped)
    }
    @Test func startRequiresUserInitiated() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher()
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot })
        await #expect(throws: EngineController.StartError.notUserInitiated) {
            try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: false, source: self.src())
        }
        #expect(launcher.launched == 0); #expect(ctl.state == .stopped)
    }
    @Test func missingRuntimeOrCheckpointNeverLaunches() async throws {
        let launcher = FakeLauncher(); let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot })
        await #expect(throws: EngineController.StartError.self) {
            try await ctl.start(settings: AppSettings(), checkpoint: nil, apiKey: nil, userInitiated: true, source: nil)
        }
        #expect(launcher.launched == 0)
    }
    @Test func portInUseRefusesToStartASecondEngine() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher()
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot }, healthFactory: { _, _ in ScriptedHealth { .healthy(HealthReport()) } })
        // plan creation needs a runtime; point the developer source at a fake tree
        let repo = tempDir(); try FileManager.default.createDirectory(at: repo.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: repo.appendingPathComponent(".venv/bin/python").path, contents: Data(), attributes: [.posixPermissions: 0o755])
        await #expect(throws: EngineController.StartError.portInUse(8080)) {
            try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: true, source: .developer(repoRoot: repo))
        }
        #expect(launcher.launched == 0)
    }
    @Test func startThenHealthyThenGracefulStop() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher()
        let calls = Counter()
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot }, healthFactory: { _, _ in ScriptedHealth {
            // first call is the pre-start probe (nothing listening); later calls report a ready engine
            calls.next() == 0 ? .unreachable : .healthy(HealthReport(status: "ok", ready: true))
        } })
        ctl.pollInterval = 0.02
        let repo = tempDir(); try FileManager.default.createDirectory(at: repo.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: repo.appendingPathComponent(".venv/bin/python").path, contents: Data(), attributes: [.posixPermissions: 0o755])
        try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: true, source: .developer(repoRoot: repo))
        #expect(launcher.launched == 1)
        for _ in 0..<100 where ctl.state != .running { try await Task.sleep(nanoseconds: 20_000_000) }
        #expect(ctl.state == .running)
        await ctl.stop()
        #expect(ctl.state == .stopped); #expect(launcher.last?.signals == ["INT"])
    }
    @Test func unexpectedExitIsAnErrorWithTheSafetyMessage() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher()
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot }, healthFactory: { _, _ in ScriptedHealth { .unreachable } })
        ctl.pollInterval = 0.05
        let repo = tempDir(); try FileManager.default.createDirectory(at: repo.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: repo.appendingPathComponent(".venv/bin/python").path, contents: Data(), attributes: [.posixPermissions: 0o755])
        try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: true, source: .developer(repoRoot: repo))
        launcher.last?.exit(2)
        for _ in 0..<100 { if case .error = ctl.state { break }; try await Task.sleep(nanoseconds: 20_000_000) }
        guard case .error(let message) = ctl.state else { Issue.record("expected error, got \(ctl.state)"); return }
        #expect(message.contains("status 2"))
    }
    @Test func stopEscalatesWhenTheEngineIgnoresInterrupt() async throws {
        let (s, c, _) = try ready(); let launcher = FakeLauncher(); launcher.ignoreInterrupt = true
        let ctl = EngineController(logs: LogStore(), launcher: launcher, snapshot: { testACSnapshot }, healthFactory: { _, _ in ScriptedHealth { .unreachable } })
        ctl.gracefulStopSeconds = 0.2; ctl.pollInterval = 0.5
        let repo = tempDir(); try FileManager.default.createDirectory(at: repo.appendingPathComponent(".venv/bin"), withIntermediateDirectories: true)
        FileManager.default.createFile(atPath: repo.appendingPathComponent(".venv/bin/python").path, contents: Data(), attributes: [.posixPermissions: 0o755])
        try await ctl.start(settings: s, checkpoint: c, apiKey: nil, userInitiated: true, source: .developer(repoRoot: repo))
        await ctl.stop()
        #expect(launcher.last?.signals == ["INT", "TERM"])
        #expect(ctl.state == .stopped || { if case .error = ctl.state { return true }; return false }())
    }
}
final class Counter: @unchecked Sendable { private var n = 0; private let l = NSLock(); func next() -> Int { l.lock(); defer { l.unlock() }; defer { n += 1 }; return n } }

// MARK: sanitizer, logs, diagnostics
@Suite struct SanitizerTests {
    let s = Sanitizer(homeDirectory: "/Users/alice", extraSecrets: ["vx-localsecret1234"])
    @Test func removesTokensKeysAndPaths() {
        let text = """
        Authorization: Bearer abcdefghijklmnop1234
        OPENAI_API_KEY=sk-abcdefghijklmnopqrstuvwxyz012345
        token ghp_abcdefghijklmnopqrstuvwxyz0123456789
        hf_abcdefghijklmnopqrstuvwxyz0123456789
        {"api_key": "supersecretvalue", "x": 1}
        own key vx-localsecret1234 here
        path /Users/alice/Documents/project and /Users/bob/other
        """
        let out = s.clean(text)
        for leak in ["abcdefghijklmnop1234", "sk-abcdefghijklmnopqrstuvwxyz", "ghp_abcdef", "hf_abcdef", "supersecretvalue", "vx-localsecret1234", "alice", "bob"] {
            #expect(!out.contains(leak), "leaked: \(leak)")
        }
        #expect(out.contains("~/Documents/project")); #expect(out.contains("/Users/[user]/other"))
    }
    @Test func removesPromptTextFromJSON() {
        let out = s.clean(#"{"messages":[{"role":"user","content":"my private source code here"}]}"#)
        #expect(!out.contains("private source code"))
    }
    @Test func leavesOrdinaryLinesAlone() { #expect(s.clean("engine ready on port 8080") == "engine ready on port 8080") }
}

@Suite struct LogAndDiagnosticsTests {
    @Test func ringBufferKeepsOnlyTheNewestLines() {
        let l = LogStore(capacity: 3)
        for i in 1...5 { l.append("line \(i)", source: .app) }
        #expect(l.snapshot().map(\.text) == ["line 3", "line 4", "line 5"])
    }
    @Test func diskLogIsOwnerOnly() throws {
        let f = tempDir().appendingPathComponent("logs/engine.log"); let l = LogStore(fileURL: f)
        l.append("hello", source: .engine)
        let attrs = try FileManager.default.attributesOfItem(atPath: f.path)
        #expect((attrs[.posixPermissions] as? NSNumber)?.intValue == 0o600)
    }
    @Test func safetyMarkersAreRecognised() {
        #expect(SafetyStopDetector.isSafetyStop("memory_admission_refused: 4.9 GiB available"))
        #expect(SafetyStopDetector.isSafetyStop("Nothing was loaded. No guard was weakened"))
        #expect(!SafetyStopDetector.isSafetyStop("token 12 decoded"))
    }
    @Test func exportedDiagnosticsContainNoSecretsOrUserName() {
        var settings = AppSettings(); settings.checkpointPath = "/Users/alice/models/x"
        let input = DiagnosticsInput(appVersion: "3.8.1", appBuild: "1", settings: settings, engineState: "Running", checkpoint: nil, health: nil,
                                     memory: nil, lastSafetyStop: nil,
                                     logLines: ["[engine] Authorization: Bearer abcdefghijklmnop1234", "[engine] reading /Users/alice/models/x"],
                                     supervisorStatusJSON: #"{"path":"/Users/alice/.flashnext"}"#, updaterSummary: "off")
        let text = DiagnosticsExporter.build(input, sanitizer: Sanitizer(homeDirectory: "/Users/alice"))
        #expect(!text.contains("abcdefghijklmnop1234")); #expect(!text.contains("alice"))
        #expect(text.contains("sanitized"))
    }
    @Test func logsCanBeExcludedFromDiagnostics() {
        var s = AppSettings(); s.includeLogsInDiagnostics = false
        let input = DiagnosticsInput(appVersion: "1", appBuild: "1", settings: s, engineState: "Stopped", checkpoint: nil, health: nil, memory: nil,
                                     lastSafetyStop: nil, logLines: ["SECRET-LINE"], supervisorStatusJSON: nil, updaterSummary: "")
        #expect(!DiagnosticsExporter.build(input, sanitizer: Sanitizer()).contains("SECRET-LINE"))
    }
}

@Suite struct HealthDecodingTests {
    @Test func decodesTheEnginesHealthShapeAndToleratesUnknownFields() throws {
        let json = #"""
        {"status":"ok","ready":true,"worker":{"stalled":false},"model":"qwen","busy":false,"queue":{"depth":0,"running":1},
         "context_limit":16384,"max_cold_prefill":4096,"cache":{"entries":12,"ram_gb":0.5,"name":"x","ssd_cache_bytes":1200,"ssd_cache_budget_bytes":10000},
         "activity":{"active":true,"stage":"prefill","prompt_tokens":200,"prefill_done":50,"prefill_total":100,"decode_tokens":3,"decode_tokens_per_second":2.5},
         "memory_policy":{"can_admit":true,"stop_reason":null,"current_swap_growth_mb":12.5,"swap_ceiling_mib":3072}}
        """#
        let r = try JSONDecoder().decode(HealthReport.self, from: Data(json.utf8))
        #expect(r.isHealthy); #expect(r.contextLimit == 16384); #expect(r.cache?.values["entries"] == 12)
        #expect(r.memoryPolicy?.swapCeilingMib == 3072); #expect(r.queue?.values["depth"] == 0)
        #expect(r.activity?.prefillDone == 50); #expect(r.activity?.promptTokens == 200)
        #expect(r.activity?.decodeTokensPerSecond == 2.5); #expect(r.cache?.values["ssd_cache_bytes"] == 1200)
    }
    @Test func aFaultMakesItUnhealthy() throws {
        let r = try JSONDecoder().decode(HealthReport.self, from: Data(#"{"status":"unavailable","ready":false,"fault":"gpu"}"#.utf8))
        #expect(!r.isHealthy)
    }
    @Test func modelsReportDecodes() throws {
        let r = try JSONDecoder().decode(ModelsReport.self, from: Data(#"{"object":"list","data":[{"id":"m1","context_window":16384}],"models":[]}"#.utf8))
        #expect(r.data.first?.id == "m1"); #expect(r.data.first?.contextWindow == 16384)
    }
}

/// A fixed AC-power reading so controller tests do not depend on the battery state of the machine running them.
let testACSnapshot = MemorySnapshot(physicalBytes: 16 << 30, availableBytes: 8 << 30, swapUsedBytes: 0, swapTotalBytes: 0,
                                    onACPower: true, batteryPercent: 100, lowPowerMode: false, cacheVolumeFreeBytes: nil)

@Suite struct MemoryProbeTests {
    @Test func snapshotIsPlausible() {
        let s = MemoryProbe.snapshot()
        #expect(s.physicalBytes > 1_000_000_000)
    }
    @Test func advisoriesFlagBatteryAndLowMemory() {
        let s = MemorySnapshot(physicalBytes: 16 << 30, availableBytes: 3 << 30, swapUsedBytes: 3 << 30, swapTotalBytes: 4 << 30,
                               onACPower: false, batteryPercent: 50, lowPowerMode: true, cacheVolumeFreeBytes: 5 << 30)
        #expect(MemoryProbe.advisories(s, checkpointBytes: nil).count == 5)
    }
    private func power(ac: Bool?, _ pct: Int?) -> MemorySnapshot {
        MemorySnapshot(physicalBytes: 16 << 30, availableBytes: 8 << 30, swapUsedBytes: 0, swapTotalBytes: 0,
                       onACPower: ac, batteryPercent: pct, lowPowerMode: false, cacheVolumeFreeBytes: nil)
    }
    @Test func batteryAboveTheFloorMayStartAndBelowItMayNot() {
        #expect(MemoryProbe.powerRefusal(power(ac: false, 25)) == nil)
        #expect(MemoryProbe.powerRefusal(power(ac: false, 80)) == nil)
        let low = MemoryProbe.powerRefusal(power(ac: false, 24))
        #expect(low?.contains("24%") == true); #expect(low?.contains("25%") == true)
    }
    @Test func acPowerAlwaysPassesAndUnreadablePowerFailsClosed() {
        #expect(MemoryProbe.powerRefusal(power(ac: true, 3)) == nil)
        #expect(MemoryProbe.powerRefusal(power(ac: true, nil)) == nil)
        #expect(MemoryProbe.powerRefusal(power(ac: false, nil)) != nil)
        #expect(MemoryProbe.powerRefusal(power(ac: nil, nil)) != nil)
        #expect(MemoryProbe.powerRefusal(power(ac: false, 1), minPercent: 0) == nil)
    }
}

// MARK: Nemotron model selection
@Suite struct NemotronSelectionTests {
    private let exists: (String) -> Bool = { _ in true }
    private func nemotronFolder(quant: String = #"{"mode":"affine","bits":4,"group_size":64}"#, partial: Bool = false) throws -> URL {
        let d = tempDir().appendingPathComponent("nemo")
        try makeCheckpoint(in: d, modelType: "nemotron_h")
        try Data(#"{"model_type":"nemotron_h","quantization":\#(quant)}"#.utf8).write(to: d.appendingPathComponent("config.json"))
        if partial { try Data().write(to: d.appendingPathComponent("model-00001-of-00002.safetensors.incomplete")) }
        return d
    }

    @Test func defaultsKeepFlashNextAndIndependentPaths() {
        let s = AppSettings()
        #expect(s.selectedModel == .flashnext); #expect(s.nemotronCheckpointPath.isEmpty)
        var t = s; t.selectedModel = .nemotron35; t.activeCheckpointPath = "/n"
        #expect(t.nemotronCheckpointPath == "/n"); #expect(t.checkpointPath.isEmpty)
    }
    @Test func oldSettingsFileStillDecodes() throws {
        let s = try JSONDecoder().decode(AppSettings.self, from: Data(#"{"port":8081,"checkpointPath":"/q"}"#.utf8))
        #expect(s.selectedModel == .flashnext); #expect(s.checkpointPath == "/q")
        let odd = try JSONDecoder().decode(AppSettings.self, from: Data(#"{"selectedModel":"future-model"}"#.utf8))
        #expect(odd.selectedModel == .flashnext)
    }
    @Test func validatorIsFamilySpecific() throws {
        let d = try nemotronFolder()
        #expect(CheckpointValidator.validate(directory: d, family: .nemotron35).verdict == .compatible)
        #expect(CheckpointValidator.validate(directory: d, family: .flashnext).verdict == .incompatible)
        let q = tempDir().appendingPathComponent("q"); try makeCheckpoint(in: q)
        #expect(CheckpointValidator.validate(directory: q, family: .nemotron35).verdict == .incompatible)
    }
    @Test func familyIsDetectedFromTheFolder() throws {
        #expect(CheckpointValidator.detectFamily(directory: try nemotronFolder()) == .nemotron35)
        let q = tempDir().appendingPathComponent("q"); try makeCheckpoint(in: q)
        #expect(CheckpointValidator.detectFamily(directory: q) == .flashnext)
        let l = tempDir().appendingPathComponent("l"); try makeCheckpoint(in: l, modelType: "llama")
        #expect(CheckpointValidator.detectFamily(directory: l) == nil)
    }
    @Test func nvfp4AndPartialDownloadsAreRefused() throws {
        let nv = try nemotronFolder(quant: #"{"mode":"nvfp4","bits":4,"group_size":16}"#)
        #expect(CheckpointValidator.validate(directory: nv, family: .nemotron35).verdict == .incompatible)
        let partial = try nemotronFolder(partial: true)
        #expect(CheckpointValidator.validate(directory: partial, family: .nemotron35).verdict == .incomplete)
    }
    @Test func launchPlanSelectsNemotronAndAuthorizesIt() throws {
        let d = try nemotronFolder()
        var s = AppSettings(); s.selectedModel = .nemotron35; s.nemotronCheckpointPath = d.path; s.checkpointPath = "/qwen"
        s.cacheDirectory = "/cache"
        let plan = try LaunchPlan.make(settings: s, checkpoint: CheckpointValidator.validate(directory: d, family: .nemotron35),
                                       source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil, fileExists: exists, isExecutable: exists)
        let m = try #require(plan.arguments.firstIndex(of: "--model")); #expect(plan.arguments[m + 1] == "nemotron35-lightning")
        let c = try #require(plan.arguments.firstIndex(of: "--checkpoint")); #expect(plan.arguments[c + 1] == d.path)
        #expect(!plan.arguments.contains("--prefix-dir"))
        #expect(plan.environment["VEYSTRIX_NEMOTRON_AUTHORIZED"] == "1")
        #expect(plan.arguments.contains("127.0.0.1"))
    }
    @Test func flashNextPlanIsUnchanged() throws {
        let d = tempDir().appendingPathComponent("q"); try makeCheckpoint(in: d)
        var s = AppSettings(); s.checkpointPath = d.path; s.cacheDirectory = "/cache"
        let plan = try LaunchPlan.make(settings: s, checkpoint: CheckpointValidator.validate(directory: d),
                                       source: .developer(repoRoot: URL(fileURLWithPath: "/repo")), apiKey: nil, fileExists: exists, isExecutable: exists)
        #expect(!plan.arguments.contains("--model")); #expect(plan.arguments.contains("--prefix-dir"))
        #expect(plan.environment["VEYSTRIX_NEMOTRON_AUTHORIZED"] == nil)
    }
}

// MARK: lifetime usage statistics
@Suite struct UsageStatsTests {
    private let cal = Calendar(identifier: .gregorian)
    private var now: Date { DateComponents(calendar: cal, year: 2026, month: 11, day: 11, hour: 12).date! }
    private func at(_ daysAgo: Int) -> Date { cal.date(byAdding: .day, value: -daysAgo, to: now)! }
    private var records: [UsageRecord] { [
        UsageRecord(time: at(0), model: "veystrix-nemotron35-lightning", prompt: 1000, cached: 600, completion: 50, wallSeconds: 20, prefillSeconds: 8, decodeSeconds: 10),
        UsageRecord(time: at(0), model: "veystrix-flashnext", prompt: 500, cached: 0, completion: 100, wallSeconds: 30, prefillSeconds: 5, decodeSeconds: 20),
        UsageRecord(time: at(3), model: "veystrix-nemotron35-lightning", prompt: 200, cached: 200, completion: 10, wallSeconds: 4),
        UsageRecord(time: at(40), model: "veystrix-flashnext", prompt: 70, cached: 10, completion: 5, wallSeconds: 2),
        UsageRecord(time: at(0), model: "veystrix-flashnext", prompt: 999, cached: 0, completion: 999, wallSeconds: 99, warm: true),
    ] }

    @Test func lifetimeTotalsSplitByModelAndExcludeWarmups() {
        let loaded = ["2026-11-11": ["veystrix-nemotron35-lightning": 600.0], "2026-11-01": ["veystrix-flashnext": 1000.0]]
        let s = UsageStats.summarize(records: records, loaded: loaded, range: .allTime, now: now, calendar: cal)
        #expect(s.total.requests == 4); #expect(s.total.inputTokens == 1770); #expect(s.total.cachedTokens == 810)
        #expect(s.total.outputTokens == 165); #expect(s.total.totalTokens == 1935); #expect(s.total.loadedSeconds == 1600)
        #expect(s.warmupRequestsExcluded == 1); #expect(s.models.first?.model == "veystrix-flashnext")
    }
    @Test func periodsNarrowTheWindowAndRatesIgnoreUntimedRecords() {
        let week = UsageStats.summarize(records: records, range: .week, now: now, calendar: cal)
        #expect(week.total.requests == 3)
        let all = UsageStats.summarize(records: records, range: .allTime, now: now, calendar: cal)
        let nemotron = all.models.first { $0.model.contains("nemotron") }!
        #expect(abs(nemotron.decodeTokensPerSecond! - 5.0) < 1e-9)
        #expect(abs(nemotron.cacheHitRate! - 800.0 / 1200.0) < 1e-9)
    }
    @Test func storeReadsArchivesSkipsBadLinesAndLoadedTimeAccumulatesAcrossSessions() throws {
        let dir = tempDir(); let store = UsageStore(directory: dir)
        let line = #"{"ts":1700000000,"model":"veystrix-flashnext","prompt":10,"cached":4,"uncached":6,"completion":3,"wall_s":1.5}"#
        try (line + "\nnot json\n" + line + "\n").write(to: dir.appendingPathComponent("requests.1699999999.jsonl"), atomically: true, encoding: .utf8)
        try (line + "\n").write(to: dir.appendingPathComponent("requests.jsonl"), atomically: true, encoding: .utf8)
        #expect(store.loadRecords().count == 3)
        let t0 = Date(); let day = UsageStats.dayKey(t0)
        let first = LoadedTimeRecorder(store: store, flushEvery: 1)
        first.sample(model: "veystrix-flashnext", now: t0); first.sample(model: "veystrix-flashnext", now: t0.addingTimeInterval(10)); first.flush(now: t0)
        let second = LoadedTimeRecorder(store: store, flushEvery: 1)
        second.sample(model: "veystrix-flashnext", now: t0.addingTimeInterval(100)); second.sample(model: "veystrix-flashnext", now: t0.addingTimeInterval(105)); second.flush(now: t0)
        #expect(abs((store.loadLoadedTime()[day]?["veystrix-flashnext"] ?? 0) - 15) < 1e-6)
        try store.reset()
        #expect(store.loadRecords().isEmpty)
    }
}
