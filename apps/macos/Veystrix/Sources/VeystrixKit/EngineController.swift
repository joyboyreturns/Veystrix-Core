import Foundation
import Combine

/// Owns the engine's lifecycle. Nothing here runs at app launch: the engine starts only when the user calls
/// `start(userInitiated: true)`, and the controller refuses to start without that flag.
@MainActor
public final class EngineController: ObservableObject {
    @Published public private(set) var state: EngineState = .stopped
    @Published public private(set) var health: HealthReport?
    @Published public private(set) var models: ModelsReport?
    @Published public private(set) var memory: MemorySnapshot?
    @Published public private(set) var lastSafetyStop: String?
    @Published public private(set) var isExternallyManaged = false
    @Published public private(set) var startedAt: Date?
    @Published public private(set) var advisories: [String] = []

    public let logs: LogStore
    /// Adds lifetime "model loaded and ready" time to the statistics. Optional: tests and headless use leave it nil.
    public var loadedTimeRecorder: LoadedTimeRecorder?
    private let launcher: EngineProcessLauncher
    private let healthFactory: @Sendable (Int, String?) -> HealthChecking
    private var process: EngineProcess?
    private var monitor: Task<Void, Never>?
    private var startInProgress = false
    private var stopRequested = false
    private var lastExitStatus: Int32?

    /// Longest allowed time from launch to the first healthy response. Loading a 113 GB checkpoint takes minutes.
    private let snapshot: @Sendable () -> MemorySnapshot
    public var startupTimeout: TimeInterval = 30 * 60
    public var pollInterval: TimeInterval = 2
    public var gracefulStopSeconds: TimeInterval = 90

    public init(logs: LogStore,
                launcher: EngineProcessLauncher = SystemEngineLauncher(),
                snapshot: @escaping @Sendable () -> MemorySnapshot = { MemoryProbe.snapshot() },
                healthFactory: @escaping @Sendable (Int, String?) -> HealthChecking = { RealHealth(client: HealthClient(port: $0, apiKey: $1)) }) {
        self.snapshot = snapshot
        self.logs = logs
        self.launcher = launcher
        self.healthFactory = healthFactory
    }

    // MARK: start

    public enum StartError: Error, Equatable, CustomStringConvertible {
        case notUserInitiated, alreadyActive, portInUse(Int), plan(String), launch(String), batteryTooLow(String)
        public var description: String {
            switch self {
            case .notUserInitiated: return "The engine only starts when you press Start."
            case .alreadyActive: return "The engine is already running or starting."
            case .portInUse(let p): return "Port \(p) is already in use (possibly by another Veystrix engine). Stop it or choose another port in Settings."
            case .plan(let m): return m
            case .launch(let m): return "The engine could not be launched: \(m)"
            case .batteryTooLow(let m): return m
            }
        }
    }

    /// Start the engine. `userInitiated` must be true; the UI passes it only from an explicit button or menu action.
    public func start(settings: AppSettings, checkpoint: CheckpointReport?, apiKey: String?, userInitiated: Bool,
                      source: LaunchPlan.Source?) async throws {
        guard userInitiated else { throw StartError.notUserInitiated }
        guard canStart, !startInProgress else { throw StartError.alreadyActive }
        startInProgress = true
        defer { startInProgress = false }
        // Battery use is allowed; a battery below the floor refuses the start before anything is launched.
        let power = snapshot()
        if !settings.allowBattery && power.onACPower != true { throw StartError.batteryTooLow("Plug in the Mac or enable battery operation in Settings.") }
        if let refusal = MemoryProbe.powerRefusal(power, minPercent: settings.minimumBatteryPercent) { throw StartError.batteryTooLow(refusal) }
        if power.lowPowerMode && !settings.allowLowPowerMode { throw StartError.batteryTooLow("Low Power Mode is on. Turn it off or allow it in Veystrix Settings.") }
        let source = source ?? .developer(repoRoot: URL(fileURLWithPath: "/nonexistent"))
        let plan: LaunchPlan
        do {
            plan = try LaunchPlan.make(settings: settings, checkpoint: checkpoint, source: source, apiKey: apiKey)
        } catch {
            throw StartError.plan("\(error)")
        }
        // Refuse if something already answers on the port: never start a second model.
        let probe = healthFactory(settings.port, apiKey)
        if case .unreachable = await probe.check() {} else { throw StartError.portInUse(settings.port) }
        try Task.checkCancellation()

        state = .starting
        health = nil; models = nil; lastSafetyStop = nil; stopRequested = false; lastExitStatus = nil
        isExternallyManaged = false
        memory = snapshot()
        advisories = MemoryProbe.advisories(memory!, checkpointBytes: checkpoint?.totalBytes, minBatteryPercent: settings.minimumBatteryPercent, allowLowPowerMode: settings.allowLowPowerMode)
        logs.append("Starting: \(plan.description)", source: .app)
        do {
            process = try launcher.launch(plan, onOutput: { [weak self] text in
                self?.logs.append(text, source: .engine)
                if text.split(whereSeparator: \.isNewline).contains(where: { SafetyStopDetector.isSafetyStop(String($0)) }) {
                    let first = text.split(whereSeparator: \.isNewline).first { SafetyStopDetector.isSafetyStop(String($0)) }.map(String.init)
                    Task { @MainActor [weak self] in self?.lastSafetyStop = first }
                }
            }, onExit: { [weak self] status in
                Task { @MainActor [weak self] in self?.processExited(status, port: settings.port) }
            })
        } catch {
            state = .error("launch failed: \(error.localizedDescription)")
            throw StartError.launch(error.localizedDescription)
        }
        startedAt = Date()
        beginMonitoring(port: settings.port, apiKey: apiKey)
    }

    private var isErrorState: Bool { if case .error = state { return true }; return false }
    private var canStart: Bool { process == nil && !isExternallyManaged && (state == .stopped || isErrorState) }

    // MARK: monitor

    private func beginMonitoring(port: Int, apiKey: String?) {
        monitor?.cancel()
        let checker = healthFactory(port, apiKey)
        let begun = Date()
        monitor = Task { [weak self] in
            var lost = 0
            while !Task.isCancelled {
                guard let self else { return }
                let interval = self.pollInterval
                try? await Task.sleep(nanoseconds: UInt64(interval * 1_000_000_000))
                if Task.isCancelled { return }
                let outcome = await checker.check()
                if Task.isCancelled { return }
                self.apply(outcome: outcome, begun: begun, lost: &lost)
            }
        }
    }

    private func apply(outcome: HealthClient.Outcome, begun: Date, lost: inout Int) {
        memory = snapshot()
        switch outcome {
        case .healthy(let report):
            lost = 0
            health = report
            if state == .starting { state = .running; logs.append("Engine ready.", source: .app) }
            if state == .running { loadedTimeRecorder?.sample(model: report.model) }
        case .unhealthy(let report):
            lost = 0
            health = report
            if let reason = report.memoryPolicy?.stopReason, !reason.isEmpty { lastSafetyStop = reason }
            if report.status == "stopped" { lastSafetyStop = lastSafetyStop ?? "The engine closed work admission after a safety stop." }
        case .unauthorized:
            state = .error("The engine rejected the API key. Rotate the key in Settings or restart the engine.")
        case .unreachable:
            lost += 1
            loadedTimeRecorder?.interrupt()
            if state == .starting, Date().timeIntervalSince(begun) > startupTimeout {
                logs.append("Startup timed out after \(Int(startupTimeout / 60)) minutes; stopping the engine.", source: .app)
                state = .error("The engine did not become ready in \(Int(startupTimeout / 60)) minutes.")
                forceStop()
            } else if state == .running, lost >= 15 {
                state = .starting      // the supervisor is most likely restarting it
                health = nil
                logs.append("Lost contact with the engine; waiting for it to come back.", source: .app)
            }
        }
        if state == .running, models == nil { models = nil }
    }

    /// Fetch `/v1/models` (non-loading metadata) for the dashboard. Safe to call while running.
    public func refreshModels(port: Int, apiKey: String?) async {
        guard state == .running else { return }
        models = await healthFactory(port, apiKey).models()
    }

    private func processExited(_ status: Int32, port: Int) {
        lastExitStatus = status
        loadedTimeRecorder?.interrupt()
        monitor?.cancel(); monitor = nil
        process = nil
        health = nil
        if stopRequested || status == 0 {
            state = .stopped
            logs.append("Engine stopped.", source: .app)
        } else {
            // The model router uses status 98 for a failed bind (EADDRINUSE).
            var message = status == 98 ? StartError.portInUse(port).description
                : "The engine exited unexpectedly (status \(status))."
            if let stop = lastSafetyStop { message += " Last safety message: \(stop)" }
            state = .error(message)
            logs.append(message, source: .app)
        }
        startedAt = nil
    }

    // MARK: stop

    /// Graceful stop: SIGINT (like Ctrl-C), then SIGTERM, then SIGKILL. Returns when the process is gone.
    public func stop() async {
        guard let process else {
            if isExternallyManaged { logs.append("This engine was started outside the app; stop it where it was started.", source: .app) }
            else if isErrorState { state = .stopped }
            return
        }
        guard state != .stopping else { return }
        stopRequested = true
        state = .stopping
        monitor?.cancel()
        logs.append("Stopping the engine (graceful).", source: .app)
        process.interrupt()
        if await waitForExit(process, seconds: gracefulStopSeconds) { return }
        logs.append("Graceful stop timed out; sending SIGTERM.", source: .app)
        process.terminate()
        if await waitForExit(process, seconds: 15) { return }
        logs.append("Still running; sending SIGKILL.", source: .app)
        process.kill()
        _ = await waitForExit(process, seconds: 5)
    }

    private func waitForExit(_ p: EngineProcess, seconds: TimeInterval) async -> Bool {
        let deadline = Date().addingTimeInterval(seconds)
        while p.isRunning && Date() < deadline { try? await Task.sleep(nanoseconds: 100_000_000) }
        // allow the termination handler's main-actor hop to run
        try? await Task.sleep(nanoseconds: 50_000_000)
        return !p.isRunning
    }

    private func forceStop() { stopRequested = true; process?.terminate() }

    public func restart(settings: AppSettings, checkpoint: CheckpointReport?, apiKey: String?, source: LaunchPlan.Source?) async throws {
        await stop()
        try await start(settings: settings, checkpoint: checkpoint, apiKey: apiKey, userInitiated: true, source: source)
    }

    /// Called when the app quits: do not leave an orphaned model process holding memory.
    public func shutdownForQuit() {
        loadedTimeRecorder?.interrupt()          // write the loaded time not yet saved
        guard let process, process.isRunning else { return }
        stopRequested = true
        process.interrupt()
        let deadline = Date().addingTimeInterval(10)
        while process.isRunning && Date() < deadline { Thread.sleep(forTimeInterval: 0.1) }
        if process.isRunning { process.terminate(); Thread.sleep(forTimeInterval: 1) }
        if process.isRunning { process.kill() }
    }

    // MARK: detect an engine started elsewhere (read-only; never starts anything)

    public func detectExternalEngine(port: Int, apiKey: String?) async {
        guard state == .stopped, !startInProgress else { return }
        let outcome = await healthFactory(port, apiKey).check()
        guard state == .stopped, !startInProgress else { return }
        switch outcome {
        case .healthy(let report), .unhealthy(let report):
            isExternallyManaged = true
            health = report
            state = report.isHealthy ? .running : .starting
            logs.append("Found an engine already listening on port \(port) (started outside the app).", source: .app)
            beginMonitoring(port: port, apiKey: apiKey)
        default: break
        }
    }
}

/// Health access used by the controller. A protocol so tests can script responses.
public protocol HealthChecking: Sendable {
    func check() async -> HealthClient.Outcome
    func models() async -> ModelsReport?
}

public struct RealHealth: HealthChecking {
    let client: HealthClient
    public init(client: HealthClient) { self.client = client }
    public func check() async -> HealthClient.Outcome { await client.health() }
    public func models() async -> ModelsReport? { await client.models() }
}
