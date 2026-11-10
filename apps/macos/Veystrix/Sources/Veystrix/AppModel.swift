import SwiftUI
import VeystrixKit

/// Ties settings, the engine controller, the checkpoint report and the updater together for the views.
/// Opening the app never starts the engine: starting requires `requestStart()` from a button or menu item.
@MainActor
final class AppModel: ObservableObject {
    let paths = Paths()
    let settingsStore: SettingsStore
    let keyStore: APIKeyStore
    let logs: LogStore
    let engine: EngineController
    let updates = UpdateController()

    @Published var settings: AppSettings { didSet { if settings != oldValue { persist() } } }
    @Published var checkpoint: CheckpointReport?
    @Published var pendingStartConfirmation = false
    @Published var pendingRestartConfirmation = false
    @Published var alertMessage: String?
    @Published var apiKey: String?
    @Published var runtimeDescription = "Checking…"

    init() {
        settingsStore = SettingsStore(paths: paths)
        keyStore = APIKeyStore(paths: paths)
        logs = LogStore(fileURL: paths.engineLogFile)
        engine = EngineController(logs: logs)
        engine.loadedTimeRecorder = LoadedTimeRecorder(store: UsageStore(paths: paths))
        settings = settingsStore.load()
        apiKey = keyStore.load()
        try? paths.ensureDirectories()
        logs.append("Veystrix \(Bundle.main.shortVersion) opened. The engine is stopped.", source: .app)
        refreshRuntime()
        revalidateCheckpoint()
        // Read-only: notice an engine that was started elsewhere (for example ./serve.sh). Starts nothing.
        Task { await engine.detectExternalEngine(port: settings.port, apiKey: effectiveKey) }
    }

    var effectiveKey: String? { settings.requireAPIKey ? (apiKey ?? keyStore.loadOrCreate()) : (apiKey ?? "local") }
    var baseURL: String { "http://127.0.0.1:\(settings.port)/v1" }

    private func persist() {
        do { try settingsStore.save(settings) } catch { alertMessage = "Could not save settings: \(error.localizedDescription)" }
    }

    func refreshRuntime() {
        switch EngineLocator.resolve(settings: settings) {
        case .bundled: runtimeDescription = "Bundled engine runtime"
        case .developer(let repo): runtimeDescription = "Developer checkout: \(paths.abbreviate(repo.path))"
        case nil: runtimeDescription = "No engine runtime found (this is a development build without a bundled engine)"
        }
    }

    // MARK: model folder

    func chooseCheckpoint(_ url: URL) {
        // Choosing a folder is an explicit selection: follow what the folder actually is, so a Nemotron folder is
        // never judged against the Flash-Next rules (or the other way round). Each model keeps its own path.
        if let family = CheckpointValidator.detectFamily(directory: url), family != settings.selectedModel {
            settings.selectedModel = family
            logs.append("The folder is a \(family.title) checkpoint; the model selection was changed to match.", source: .app)
        }
        settings.activeCheckpointPath = url.path
        revalidateCheckpoint()
    }

    func selectModel(_ family: ModelFamily) {
        guard settings.selectedModel != family else { return }
        settings.selectedModel = family
        revalidateCheckpoint()
    }

    func revalidateCheckpoint() {
        let path = settings.activeCheckpointPath
        let family = settings.selectedModel
        guard !path.isEmpty else { checkpoint = nil; return }
        Task.detached(priority: .utility) {
            let report = CheckpointValidator.validate(directory: URL(fileURLWithPath: path), family: family)
            await MainActor.run { [weak self] in self?.checkpoint = report }
        }
    }

    // MARK: engine

    func requestStart() {
        if settings.confirmBeforeLoading { pendingStartConfirmation = true } else { Task { await startNow() } }
    }

    func requestRestart() {
        if settings.confirmBeforeLoading { pendingRestartConfirmation = true } else { Task { await restartNow() } }
    }

    func startNow() async {
        refreshRuntime()
        if settings.requireAPIKey, apiKey == nil { apiKey = keyStore.loadOrCreate() }
        do {
            try await engine.start(settings: settings, checkpoint: checkpoint, apiKey: effectiveKey, userInitiated: true,
                                   source: EngineLocator.resolve(settings: settings))
        } catch {
            alertMessage = "\(error)"
        }
    }

    func restartNow() async {
        do {
            try await engine.restart(settings: settings, checkpoint: checkpoint, apiKey: effectiveKey,
                                     source: EngineLocator.resolve(settings: settings))
        } catch { alertMessage = "\(error)" }
    }

    func rotateKey() { apiKey = keyStore.rotate() }

    // MARK: diagnostics

    func diagnosticsText() -> String {
        let status = try? String(contentsOf: paths.supervisorStatusFile, encoding: .utf8)
        let input = DiagnosticsInput(
            appVersion: Bundle.main.shortVersion, appBuild: Bundle.main.buildNumber, settings: settings,
            engineState: engine.state.label + (engine.isExternallyManaged ? " (started outside the app)" : ""),
            checkpoint: checkpoint, health: engine.health, memory: engine.memory ?? MemoryProbe.snapshot(),
            lastSafetyStop: engine.lastSafetyStop, logLines: logs.tail(400).map { "[\($0.source.rawValue)] \($0.text)" },
            supervisorStatusJSON: status, updaterSummary: updates.summary)
        var secrets: [String] = []
        if let key = apiKey { secrets.append(key) }
        return DiagnosticsExporter.build(input, sanitizer: Sanitizer(extraSecrets: secrets))
    }
}

extension Bundle {
    var shortVersion: String { object(forInfoDictionaryKey: "CFBundleShortVersionString") as? String ?? "dev" }
    var buildNumber: String { object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0" }
}
