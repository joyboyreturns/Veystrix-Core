import SwiftUI
import AppKit
import VeystrixKit

@MainActor private final class HarnessModels: ObservableObject {
    @Published var entries: [InstalledModelsReport.Entry] = []
    @Published var selected = ""
    @Published var loading = false
    @Published var error: String?
    @Published var harnessResults: [String: String] = [:]
    @Published var configuring = false
    private var revision = 0
    func refresh(port: Int, key: String?, source: LaunchPlan.Source?) async {
        revision += 1
        let current = revision
        loading = true
        defer { if current == revision { loading = false } }
        var result = await HealthClient(port: port, apiKey: key).installedModels()
        if result == nil { result = await Self.offlineInventory(source: source) }
        guard current == revision else { return }
        guard let result else {
            error = "Installed models could not be read. Check the bundled engine and refresh."
            return
        }
        entries = result.data
        error = nil
        if !entries.contains(where: { $0.id == selected }) {
            selected = entries.first(where: { $0.loaded })?.id ?? entries.first?.id ?? ""
        }
    }
    func configureHarnesses(port: Int, source: LaunchPlan.Source?) async {
        configuring = true
        defer { configuring = false }
        guard let result = await Self.configure(port: port, source: source) else {
            error = "Harness model registration failed. Check the engine logs."
            return
        }
        harnessResults = result
    }
    nonisolated private static func configure(port: Int, source: LaunchPlan.Source?) async -> [String: String]? {
        await Task.detached(priority: .utility) {
            let executable: URL
            let directory: URL
            switch source {
            case .bundled(let root):
                executable = root.appendingPathComponent("python/bin/python3")
                directory = root.appendingPathComponent("lib")
            case .developer(let root):
                executable = root.appendingPathComponent(".venv/bin/python")
                directory = root
            case nil: return nil
            }
            let process = Process()
            process.executableURL = executable
            process.currentDirectoryURL = directory
            process.arguments = ["-m", "veystrix.harness_configure", "--port", String(port), "--apply"]
            var environment: [String: String] = [:]
            for key in ["HOME", "PATH", "LANG", "LC_ALL", "VEYSTRIX_MODEL_ROOT"] {
                environment[key] = ProcessInfo.processInfo.environment[key]
            }
            environment["PYTHONPATH"] = directory.path
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = environment
            let output = Pipe()
            process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            do {
                try process.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else { return nil }
                return try JSONDecoder().decode([String: String].self, from: data)
            } catch { return nil }
        }.value
    }
    // Run a metadata-only subprocess in a detached task, including when the engine is stopped.
    nonisolated private static func offlineInventory(source: LaunchPlan.Source?) async -> InstalledModelsReport? {
        await Task.detached(priority: .utility) {
            let executable: URL
            let directory: URL
            switch source {
            case .bundled(let root):
                executable = root.appendingPathComponent("python/bin/python3")
                directory = root.appendingPathComponent("lib")
            case .developer(let root):
                executable = root.appendingPathComponent(".venv/bin/python")
                directory = root
            case nil: return nil
            }
            let process = Process()
            process.executableURL = executable
            process.currentDirectoryURL = directory
            process.arguments = ["-m", "veystrix.model_registry"]
            var environment: [String: String] = [:]
            for key in ["HOME", "PATH", "LANG", "LC_ALL", "VEYSTRIX_MODEL_ROOT"] {
                environment[key] = ProcessInfo.processInfo.environment[key]
            }
            environment["PYTHONPATH"] = directory.path
            environment["PYTHONDONTWRITEBYTECODE"] = "1"
            process.environment = environment
            let output = Pipe()
            process.standardOutput = output
            process.standardError = FileHandle.nullDevice
            do {
                try process.run()
                let data = output.fileHandleForReading.readDataToEndOfFile()
                process.waitUntilExit()
                guard process.terminationStatus == 0 else { return nil }
                return try JSONDecoder().decode(InstalledModelsReport.self, from: data)
            } catch { return nil }
        }.value
    }
}

struct HarnessView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var picked = LocalState(Harness.hermes)
    @StateObject private var inventory = HarnessModels()

    private var selected: InstalledModelsReport.Entry? { inventory.entries.first { $0.id == inventory.selected } }
    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Agents").font(.largeTitle.bold())
                GroupBox("Local endpoint") {
                    VStack(alignment: .leading, spacing: 6) {
                        LabeledContent("Base URL") { Text(model.baseURL).textSelection(.enabled) }
                        LabeledContent("Connection") { Text(engine.state.label) }
                        LabeledContent("Active model") { Text(engine.health?.model ?? "None") }
                    }.frame(maxWidth: .infinity, alignment: .leading)
                }
                HStack {
                    Button("Refresh Harness Models") {
                        Task { await inventory.configureHarnesses(port: model.settings.port, source: EngineLocator.resolve(settings: model.settings)) }
                    }.disabled(inventory.configuring || inventory.entries.isEmpty)
                    if inventory.configuring { ProgressView().controlSize(.small) }
                }
                ForEach(inventory.harnessResults.keys.sorted(), id: \.self) { name in
                    Text("\(name): \(inventory.harnessResults[name] ?? "unknown")").font(.caption)
                }
                Picker("Agent", selection: $picked.value) {
                    ForEach(Harness.allCases) { Text($0.rawValue).tag($0) }
                }.pickerStyle(.segmented).accessibilityLabel("Choose an agent")
                HStack {
                    Picker("Installed model", selection: $inventory.selected) {
                        if inventory.entries.isEmpty { Text("No models discovered").tag("") }
                        ForEach(inventory.entries) { entry in Text(entry.displayName + (entry.loaded ? " (loaded)" : "")).tag(entry.id) }
                    }.disabled(inventory.entries.isEmpty)
                    Button("Refresh") { Task { await inventory.refresh(port: model.settings.port, key: model.effectiveKey, source: EngineLocator.resolve(settings: model.settings)) } }.disabled(inventory.loading)
                    if inventory.loading { ProgressView().controlSize(.small) }
                }
                if let entry = selected {
                    HStack {
                        Text(entry.loaded ? "Loaded" : "Not loaded").foregroundStyle(entry.loaded ? .green : .secondary)
                        if !entry.loaded {
                            Button("Load selected model…") { Task { await model.selectInstalled(entry) } }
                                .disabled(engine.isExternallyManaged || engine.state == .starting || engine.state == .stopping)
                        }
                    }
                    Text("Selecting a model here prepares its harness configuration. Loading it restarts the app-managed engine and clears its live cache.")
                        .font(.caption).foregroundStyle(.secondary)
                    let s = HarnessProfiles.snippet(for: picked.value, port: model.settings.port, modelID: entry.id)
                    GroupBox("Where it goes") { Text(s.whereToPutIt).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
                    GroupBox("Configuration") {
                        VStack(alignment: .leading, spacing: 8) {
                            Text(s.content).font(.system(.callout, design: .monospaced)).textSelection(.enabled).frame(maxWidth: .infinity, alignment: .leading)
                            Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(s.content, forType: .string) } label: { Label("Copy", systemImage: "doc.on.doc") }
                        }
                    }
                    ForEach(s.notes, id: \.self) { Label($0, systemImage: "info.circle").font(.callout).foregroundStyle(.secondary) }
                    if !entry.loaded { Text("This model ID is accepted only after this exact installation is loaded. Automatic switching from harness requests is not available yet.").font(.callout).foregroundStyle(.orange) }
                }
                if let error = inventory.error { Text(error).font(.callout).foregroundStyle(.secondary) }
                Text("Refresh Harness Models updates supported local configuration and keeps a backup. Restart the harness to reload its model list.").font(.footnote).foregroundStyle(.secondary)
            }.padding(20)
        }
        .navigationTitle("Agents")
        .task { await inventory.refresh(port: model.settings.port, key: model.effectiveKey, source: EngineLocator.resolve(settings: model.settings)) }
        .onChange(of: engine.state) { _ in
            Task { await inventory.refresh(port: model.settings.port, key: model.effectiveKey, source: EngineLocator.resolve(settings: model.settings)) }
        }
        .onChange(of: engine.health?.installedModelID) { _ in
            Task { await inventory.refresh(port: model.settings.port, key: model.effectiveKey, source: EngineLocator.resolve(settings: model.settings)) }
        }
    }
}
