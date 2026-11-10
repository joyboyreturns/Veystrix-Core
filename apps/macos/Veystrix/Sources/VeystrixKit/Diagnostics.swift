import Foundation

/// Builds a sanitized, human-readable diagnostics report the user can attach to a bug report.
/// Never includes: API keys, prompts, session files, model weights, or the user's name (home paths become `~`).
public struct DiagnosticsInput: Sendable {
    public var appVersion: String
    public var appBuild: String
    public var settings: AppSettings
    public var engineState: String
    public var checkpoint: CheckpointReport?
    public var health: HealthReport?
    public var memory: MemorySnapshot?
    public var lastSafetyStop: String?
    public var logLines: [String]
    public var supervisorStatusJSON: String?
    public var updaterSummary: String

    public init(appVersion: String, appBuild: String, settings: AppSettings, engineState: String, checkpoint: CheckpointReport?,
                health: HealthReport?, memory: MemorySnapshot?, lastSafetyStop: String?, logLines: [String],
                supervisorStatusJSON: String?, updaterSummary: String) {
        self.appVersion = appVersion; self.appBuild = appBuild; self.settings = settings; self.engineState = engineState
        self.checkpoint = checkpoint; self.health = health; self.memory = memory; self.lastSafetyStop = lastSafetyStop
        self.logLines = logLines; self.supervisorStatusJSON = supervisorStatusJSON; self.updaterSummary = updaterSummary
    }
}

public enum DiagnosticsExporter {
    public static func hardwareModel() -> String {
        var size = 0
        sysctlbyname("hw.model", nil, &size, nil, 0)
        var buf = [CChar](repeating: 0, count: max(size, 1))
        sysctlbyname("hw.model", &buf, &size, nil, 0)
        return String(cString: buf)
    }

    public static func build(_ input: DiagnosticsInput, sanitizer: Sanitizer, now: Date = Date()) -> String {
        var out: [String] = []
        func section(_ title: String) { out.append(""); out.append("== \(title) ==") }
        out.append("Veystrix diagnostics (sanitized)")
        out.append("Generated: \(ISO8601DateFormatter().string(from: now))")
        out.append("Secrets, prompts and your user name are removed. Review before sharing.")

        section("Application")
        out.append("Version: \(input.appVersion) (\(input.appBuild))")
        out.append("macOS: \(ProcessInfo.processInfo.operatingSystemVersionString)")
        out.append("Hardware: \(hardwareModel())")
        out.append("Updater: \(input.updaterSummary)")

        section("Settings")
        let s = input.settings
        out.append("port=\(s.port) decodePool=\(s.decodePool.rawValue)GB supervisedRestarts=\(s.supervisedRestarts) keepMacAwake=\(s.keepMacAwake)")
        out.append("cacheQuotaGB=\(s.cacheQuotaGB) cacheDirectory=\(s.cacheDirectory.isEmpty ? "(engine default)" : "(custom)")")
        out.append("confirmBeforeLoading=\(s.confirmBeforeLoading) requireAPIKey=\(s.requireAPIKey) autoUpdateCheck=\(s.checkForUpdatesAutomatically)")
        out.append("developerMode=\(s.developerRepoPath.isEmpty ? "no" : "yes")")

        section("Engine")
        out.append("State: \(input.engineState)")
        if let stop = input.lastSafetyStop { out.append("Last safety message: \(stop)") }
        if let h = input.health {
            out.append("Health: status=\(h.status ?? "?") ready=\(h.ready.map(String.init) ?? "?") busy=\(h.busy.map(String.init) ?? "?")")
            if let m = h.memoryPolicy {
                out.append("Memory policy: canAdmit=\(m.canAdmit.map(String.init) ?? "?") stop=\(m.stopReason ?? "none") swapGrowthMB=\(m.currentSwapGrowthMb.map { String(format: "%.0f", $0) } ?? "?") ceilingMiB=\(m.swapCeilingMib.map { String(format: "%.0f", $0) } ?? "?")")
            }
        }

        section("Memory and power")
        if let m = input.memory {
            out.append("Physical \(MemorySnapshot.gib(m.physicalBytes)), available ~\(MemorySnapshot.gib(m.availableBytes)), swap used \(MemorySnapshot.gib(m.swapUsedBytes)) of \(MemorySnapshot.gib(m.swapTotalBytes))")
            out.append("AC power: \(m.onACPower.map(String.init) ?? "unknown"), battery: \(m.batteryPercent.map { "\($0)%" } ?? "n/a"), Low Power Mode: \(m.lowPowerMode)")
        } else { out.append("(not sampled)") }

        section("Model folder")
        if let c = input.checkpoint {
            out.append("Verdict: \(c.verdict) type=\(c.modelType ?? "?") shards=\(c.shardCount) size=\(c.humanSize)")
            if !c.missingFiles.isEmpty { out.append("Missing: \(c.missingFiles.prefix(20).joined(separator: ", "))") }
            for n in c.notes { out.append("Note: \(n)") }
        } else { out.append("(none chosen)") }

        if let status = input.supervisorStatusJSON {
            section("Supervisor status (engine)")
            out.append(String(status.prefix(8000)))
        }
        if input.settings.includeLogsInDiagnostics {
            section("Recent log lines")
            out.append(contentsOf: input.logLines.suffix(300))
        } else {
            section("Recent log lines")
            out.append("(excluded by your privacy setting)")
        }
        return sanitizer.clean(out.joined(separator: "\n")) + "\n"
    }
}
