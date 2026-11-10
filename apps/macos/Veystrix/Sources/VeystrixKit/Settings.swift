import Foundation

/// How much of the model's expert cache to keep in memory during decode.
/// The engine accepts 0, 1 or 2 GB; anything larger is refused by the engine itself.
public enum DecodePool: Int, Codable, CaseIterable, Sendable, Identifiable {
    case none = 0, oneGB = 1, twoGB = 2
    public var id: Int { rawValue }
    public var title: String {
        switch self {
        case .none: return "No decode pool (safest)"
        case .oneGB: return "1 GB pool (engine warns)"
        case .twoGB: return "2 GB pool"
        }
    }
    public var explanation: String {
        switch self {
        case .none: return "Reads experts straight from the SSD. Lowest memory use. This is the supervised launcher's default."
        case .oneGB: return "A small pool. The engine allows it with a warning because its hit rate is low."
        case .twoGB: return "Keeps about 2 GB of recently used experts. Needs more free memory; if the engine cannot allocate it, it starts without a pool."
        }
    }
}

/// Which engine the app launches. Selection is always manual; nothing routes between models automatically.
public enum ModelFamily: String, Codable, CaseIterable, Sendable, Identifiable {
    case flashnext
    case nemotron35 = "nemotron35-lightning"
    public var id: String { rawValue }
    public var title: String {
        switch self {
        case .flashnext: return "Qwen 3.8 Flash-Next"
        case .nemotron35: return "NVIDIA Nemotron 3.5 Lightning (experimental)"
        }
    }
    /// Engine architectures (`model_type`) this family accepts.
    public var modelTypes: Set<String> {
        switch self {
        case .flashnext: return ["qwen4_exp", "qwen4_exp_text"]
        case .nemotron35: return ["nemotron_h"]
        }
    }
}

/// User-visible settings. Stored as JSON in Application Support. Unknown or missing keys fall back to defaults,
/// so an older settings file keeps working after an update (see `schemaVersion`).
public struct AppSettings: Codable, Equatable, Sendable {
    public static let currentSchemaVersion = 1

    public var schemaVersion: Int = AppSettings.currentSchemaVersion
    /// Local API port. The engine binds to 127.0.0.1 only.
    public var port: Int = 8080
    /// The Flash-Next folder (kept under the original key so existing settings keep working).
    public var checkpointPath: String = ""
    /// Which engine to launch. Flash-Next stays the default.
    public var selectedModel: ModelFamily = .flashnext
    /// The Nemotron folder. Independent of the Flash-Next folder: switching models never overwrites the other path.
    public var nemotronCheckpointPath: String = ""
    public var decodePool: DecodePool = .none
    /// Let the engine's supervisor restart the server after a crash or memory guard stop (bounded and logged).
    public var supervisedRestarts: Bool = true
    public var keepMacAwake: Bool = true
    public var allowBattery: Bool = true
    public var minimumBatteryPercent: Int = 25
    public var allowLowPowerMode: Bool = false
    /// Persistent cache quota in GB (VEYSTRIX_CACHE_GB). The engine default is 50.
    public var cacheQuotaGB: Int = 50
    /// Empty means the engine's default cache folder (existing caches are reused).
    public var cacheDirectory: String = ""
    /// Ask for confirmation before the model is loaded. The model is never loaded when the app opens.
    public var confirmBeforeLoading: Bool = true
    public var requireAPIKey: Bool = false
    public var checkForUpdatesAutomatically: Bool = false
    public var includeLogsInDiagnostics: Bool = true
    /// Developer mode: run the engine from a source checkout instead of the bundled runtime.
    public var developerRepoPath: String = ""

    public init() {}

    /// The folder of the selected model.
    public var activeCheckpointPath: String {
        get { selectedModel == .nemotron35 ? nemotronCheckpointPath : checkpointPath }
        set { if selectedModel == .nemotron35 { nemotronCheckpointPath = newValue } else { checkpointPath = newValue } }
    }

    enum CodingKeys: String, CodingKey {
        case schemaVersion, port, checkpointPath, selectedModel, nemotronCheckpointPath, decodePool, supervisedRestarts, keepMacAwake, cacheQuotaGB,
             cacheDirectory, confirmBeforeLoading, requireAPIKey, checkForUpdatesAutomatically,
             includeLogsInDiagnostics, developerRepoPath, allowBattery, minimumBatteryPercent, allowLowPowerMode
    }

    public init(from decoder: Decoder) throws {
        let c = try decoder.container(keyedBy: CodingKeys.self)
        var s = AppSettings()
        s.schemaVersion = try c.decodeIfPresent(Int.self, forKey: .schemaVersion) ?? 1
        s.port = try c.decodeIfPresent(Int.self, forKey: .port) ?? s.port
        s.checkpointPath = try c.decodeIfPresent(String.self, forKey: .checkpointPath) ?? s.checkpointPath
        s.selectedModel = (try? c.decodeIfPresent(ModelFamily.self, forKey: .selectedModel)) ?? s.selectedModel
        s.nemotronCheckpointPath = try c.decodeIfPresent(String.self, forKey: .nemotronCheckpointPath) ?? s.nemotronCheckpointPath
        s.decodePool = try c.decodeIfPresent(DecodePool.self, forKey: .decodePool) ?? s.decodePool
        s.supervisedRestarts = try c.decodeIfPresent(Bool.self, forKey: .supervisedRestarts) ?? s.supervisedRestarts
        s.keepMacAwake = try c.decodeIfPresent(Bool.self, forKey: .keepMacAwake) ?? s.keepMacAwake
        s.allowBattery = try c.decodeIfPresent(Bool.self, forKey: .allowBattery) ?? s.allowBattery
        s.minimumBatteryPercent = try c.decodeIfPresent(Int.self, forKey: .minimumBatteryPercent) ?? s.minimumBatteryPercent
        s.allowLowPowerMode = try c.decodeIfPresent(Bool.self, forKey: .allowLowPowerMode) ?? s.allowLowPowerMode
        s.cacheQuotaGB = try c.decodeIfPresent(Int.self, forKey: .cacheQuotaGB) ?? s.cacheQuotaGB
        s.cacheDirectory = try c.decodeIfPresent(String.self, forKey: .cacheDirectory) ?? s.cacheDirectory
        s.confirmBeforeLoading = try c.decodeIfPresent(Bool.self, forKey: .confirmBeforeLoading) ?? s.confirmBeforeLoading
        s.requireAPIKey = try c.decodeIfPresent(Bool.self, forKey: .requireAPIKey) ?? s.requireAPIKey
        s.checkForUpdatesAutomatically = try c.decodeIfPresent(Bool.self, forKey: .checkForUpdatesAutomatically) ?? s.checkForUpdatesAutomatically
        s.includeLogsInDiagnostics = try c.decodeIfPresent(Bool.self, forKey: .includeLogsInDiagnostics) ?? s.includeLogsInDiagnostics
        s.developerRepoPath = try c.decodeIfPresent(String.self, forKey: .developerRepoPath) ?? s.developerRepoPath
        // Migration: a future schema is read as far as it is understood, then stamped current on the next save.
        s.schemaVersion = max(s.schemaVersion, 1)
        self = s
    }

    /// Reasons a configuration cannot be used to start the engine. Empty means valid.
    public func validationProblems() -> [String] {
        var problems: [String] = []
        if !(1024...65535).contains(port) { problems.append("The port must be between 1024 and 65535.") }
        if !(0...100).contains(minimumBatteryPercent) { problems.append("The minimum battery charge must be between 0% and 100%.") }
        if !(1...2000).contains(cacheQuotaGB) { problems.append("The cache quota must be between 1 and 2000 GB.") }
        return problems
    }
}

/// Loads and saves settings atomically with owner-only permissions. Never throws on a damaged file: a damaged
/// file is kept next to the new one as `settings.json.damaged` and defaults are used.
public struct SettingsStore: Sendable {
    public let paths: Paths
    public init(paths: Paths = Paths()) { self.paths = paths }

    public func load() -> AppSettings {
        guard let data = try? Data(contentsOf: paths.settingsFile) else { return AppSettings() }
        do {
            return try JSONDecoder().decode(AppSettings.self, from: data)
        } catch {
            let damaged = paths.settingsFile.appendingPathExtension("damaged")
            try? FileManager.default.removeItem(at: damaged)
            try? FileManager.default.moveItem(at: paths.settingsFile, to: damaged)
            return AppSettings()
        }
    }

    public func save(_ settings: AppSettings) throws {
        try paths.ensureDirectories()
        var toSave = settings
        toSave.schemaVersion = AppSettings.currentSchemaVersion
        let encoder = JSONEncoder()
        encoder.outputFormatting = [.prettyPrinted, .sortedKeys]
        let data = try encoder.encode(toSave)
        let tmp = paths.applicationSupport.appendingPathComponent(".settings.\(UUID().uuidString).tmp")
        try data.write(to: tmp, options: [.atomic])
        try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: tmp.path)
        if FileManager.default.fileExists(atPath: paths.settingsFile.path) {
            _ = try FileManager.default.replaceItemAt(paths.settingsFile, withItemAt: tmp)
        } else {
            try FileManager.default.moveItem(at: tmp, to: paths.settingsFile)
        }
    }
}
