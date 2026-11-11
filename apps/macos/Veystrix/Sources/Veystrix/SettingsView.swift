import SwiftUI
import AppKit
import VeystrixKit

struct SettingsView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var updates: UpdateController
    @StateObject private var port = LocalState("")

    var body: some View {
        Form {
            Section("Engine") {
                TextField("Local API port", text: $port.value)
                    .onSubmit(applyPort).onChange(of: port.value) { _, _ in applyPort() }
                    .help("The engine always binds to 127.0.0.1. Default 8080.")
                    .accessibilityLabel("Local API port")
                Picker("Decode memory pool", selection: $model.settings.decodePool) {
                    ForEach(DecodePool.allCases) { Text($0.title).tag($0) }
                }
                Text(model.settings.decodePool.explanation).font(.callout).foregroundStyle(.secondary)
                Toggle("Restart automatically after a crash or guard stop (bounded, logged)", isOn: $model.settings.supervisedRestarts)
                Toggle("Keep the Mac awake while the engine runs", isOn: $model.settings.keepMacAwake)
                Text("Flash-Next stops at 3 GiB of total system swap. Power and thermal checks follow the engine settings.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Power") {
                Toggle("Allow operation on battery", isOn: $model.settings.allowBattery)
                Stepper("Minimum battery charge: \(model.settings.minimumBatteryPercent)%", value: $model.settings.minimumBatteryPercent, in: 0...100, step: 5)
                    .disabled(!model.settings.allowBattery)
                Toggle("Allow operation in Low Power Mode", isOn: $model.settings.allowLowPowerMode)
                Text("When battery operation is off, the engine requires AC power. A 0% minimum disables the battery floor. Low Power Mode can reduce speed. Changes apply when the engine starts next; other safety guards remain active.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Loading the model") {
                Toggle("Ask for confirmation before loading", isOn: $model.settings.confirmBeforeLoading)
                Text("The model is never loaded when the app opens. It loads only when you press Start.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Cache") {
                Stepper("Quota: \(model.settings.cacheQuotaGB) GB", value: $model.settings.cacheQuotaGB, in: 5...2000, step: 5)
                LabeledContent("Folder") {
                    Text(model.settings.cacheDirectory.isEmpty ? "Engine default (existing caches are reused)" : model.paths.abbreviate(model.settings.cacheDirectory))
                        .foregroundStyle(.secondary).textSelection(.enabled)
                }
                HStack {
                    Button("Choose Folder…") { chooseCache() }
                    Button("Use Default") { model.settings.cacheDirectory = "" }.disabled(model.settings.cacheDirectory.isEmpty)
                }
                Text("Changing the folder starts a fresh cache. The old one is not deleted or moved.").font(.callout).foregroundStyle(.secondary)
            }
            Section("Local API key") {
                Toggle("Require an API key (recommended on shared Macs)", isOn: $model.settings.requireAPIKey)
                Text("Without a key any local process can use the engine; with it, clients must send the key. Browser pages are refused either way.")
                    .font(.callout).foregroundStyle(.secondary)
                if model.settings.requireAPIKey {
                    LabeledContent("Key") { Text(model.apiKey.map { String($0.prefix(7)) + "…" } ?? "created on first start").foregroundStyle(.secondary) }
                    Button("Rotate Key") { model.rotateKey() }
                }
            }
            Section("Updates") {
                Toggle("Check for updates automatically", isOn: Binding(
                    get: { updates.automaticallyChecks && model.settings.checkForUpdatesAutomatically },
                    set: { model.settings.checkForUpdatesAutomatically = $0; updates.automaticallyChecks = $0 }))
                    .disabled(!updates.isConfigured)
                Button("Check for Updates…") { updates.checkForUpdates() }.disabled(!updates.canCheckForUpdates)
                Text(updates.summary).font(.callout).foregroundStyle(.secondary)
            }
            Section("Privacy") {
                Toggle("Include recent log lines in exported diagnostics", isOn: $model.settings.includeLogsInDiagnostics)
                Text("Veystrix has no telemetry and no account. Diagnostics are written only where you save them, with keys, prompts and your user name removed.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Developer") {
                LabeledContent("Engine runtime", value: model.runtimeDescription)
                LabeledContent("Source checkout") {
                    Text(model.settings.developerRepoPath.isEmpty ? "None" : model.paths.abbreviate(model.settings.developerRepoPath)).foregroundStyle(.secondary)
                }
                HStack {
                    Button("Choose Checkout…") { chooseRepo() }
                    Button("Clear") { model.settings.developerRepoPath = ""; model.refreshRuntime() }.disabled(model.settings.developerRepoPath.isEmpty)
                }
            }
            Section("Where things are stored") {
                LabeledContent("Settings", value: model.paths.abbreviate(model.paths.settingsFile.path))
                LabeledContent("Logs", value: model.paths.abbreviate(model.paths.logsDirectory.path))
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Settings")
        .onAppear { port.value = String(model.settings.port) }
    }

    private func applyPort() {
        if let p = Int(port.value), (1024...65535).contains(p), p != model.settings.port { model.settings.port = p }
    }
    private func chooseCache() {
        let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false; p.canCreateDirectories = true
        if p.runModal() == .OK, let u = p.url { model.settings.cacheDirectory = u.path }
    }
    private func chooseRepo() {
        let p = NSOpenPanel(); p.canChooseDirectories = true; p.canChooseFiles = false
        p.message = "Choose a Veystrix source checkout that contains .venv/bin/python."
        if p.runModal() == .OK, let u = p.url { model.settings.developerRepoPath = u.path; model.refreshRuntime() }
    }
}
