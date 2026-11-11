import SwiftUI
import AppKit
import VeystrixKit

struct CacheSettings: View {
    @EnvironmentObject private var model: AppModel
    @StateObject private var status = LocalState<CacheStatus?>(nil)
    @StateObject private var quota = LocalState(80)
    @StateObject private var amount = LocalState(10)
    @StateObject private var working = LocalState(false)
    @StateObject private var message = LocalState<String?>(nil)
    @StateObject private var confirmation = LocalState<Confirmation?>(nil)

    private struct Confirmation: Identifiable {
        let action: CacheManagement.Action
        let amount: Int?
        let detail: String
        var id: String { action.rawValue }
    }

    var body: some View {
        Section("Universal Cache") {
            if let status = status.value {
                LabeledContent("Stored states", value: "\(gb(status.usedBytes)) of \(gb(status.quotaBytes)) · \(status.snapshots) snapshots")
                LabeledContent("Free SSD space", value: "\(gb(status.freeBytes)) · \(gb(status.reserveBytes)) reserved")
                LabeledContent("Cache health", value: status.healthy ? "Healthy" : "Needs attention")
                modelRow("Qwen3.8 Flash-Next", key: "flashnext", status: status)
                modelRow("Nemotron 3.5 Lightning", key: "nemotron35", status: status)
                if let last = status.lastCleanup {
                    LabeledContent("Last cleanup", value: cleanupText(last))
                }
                LabeledContent("Cache location") {
                    Text(model.paths.abbreviate(status.root)).foregroundStyle(.secondary).textSelection(.enabled)
                }
            }
            HStack {
                Button("Refresh") { refresh() }
                Button("Choose Folder…", action: chooseCache).disabled(model.engine.state.isActive)
                Button("Use Default") { model.settings.cacheDirectory = ""; refresh() }
                    .disabled(model.settings.cacheDirectory.isEmpty || model.engine.state.isActive)
                if working.value { ProgressView().controlSize(.small) }
            }
            HStack {
                TextField("Quota in GB", value: $quota.value, format: .number).frame(maxWidth: 180)
                Button("Apply Quota") { execute(.quota, amount: quota.value) }
                    .disabled(!(1...2000).contains(quota.value))
            }
            HStack {
                TextField("Reclaim GB", value: $amount.value, format: .number).frame(maxWidth: 180)
                Button("Preview Reclaim") { preview() }.disabled(!(1...2000).contains(amount.value))
            }
            HStack {
                ForEach([5, 10, 20, 40], id: \.self) { preset in
                    Button("\(preset) GB") { amount.value = preset }
                }
            }
            Button("Clear All Cached States", role: .destructive) {
                confirmation.value = Confirmation(action: .clear, amount: nil,
                    detail: "Delete every eligible saved inference state from both models? Veystrix will recompute deleted states when needed.")
            }
            if let message = message.value { Text(message).font(.callout).foregroundStyle(.secondary) }
        }
        .disabled(working.value)
        .task {
            refresh()
            while !Task.isCancelled {
                do { try await Task.sleep(for: .seconds(15)) } catch { return }
                if !working.value {
                    do {
                        let fresh: CacheStatus = try await run(.status)
                        status.value = fresh
                    } catch {
                        message.value = error.localizedDescription
                    }
                }
            }
        }
        .confirmationDialog("Remove cached states?", isPresented: Binding(
            get: { confirmation.value != nil }, set: { if !$0 { confirmation.value = nil } }
        ), titleVisibility: .visible, presenting: confirmation.value) { choice in
            Button(choice.action == .clear ? "Clear Cached States" : "Reclaim Storage", role: .destructive) {
                execute(choice.action, amount: choice.amount)
            }
        } message: { choice in
            Text(choice.detail)
        }
    }

    @ViewBuilder
    private func modelRow(_ title: String, key: String, status: CacheStatus) -> some View {
        if let usage = status.models[key] {
            LabeledContent(title, value: "\(gb(usage.bytes)) · \(usage.snapshots) snapshots")
            if let hits = usage.hits, let misses = usage.misses {
                Text("\(hits) hits · \(misses) misses · \(usage.restoredTokens ?? 0) restored tokens")
                    .font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private func gb(_ bytes: Int64) -> String {
        (Double(bytes) / 1_000_000_000).formatted(.number.precision(.fractionLength(0...2))) + " GB"
    }

    private func cleanupText(_ result: CacheReclaim) -> String {
        var resultText = "Reclaimed \(gb(result.reclaimedBytes)) from \(result.snapshots) snapshots."
        if result.shortfallBytes > 0 { resultText += " Protected states left a \(gb(result.shortfallBytes)) shortfall." }
        return resultText
    }

    private func run<Result: Decodable & Sendable>(_ action: CacheManagement.Action, amount: Int? = nil) async throws -> Result {
        try await CacheManagement.run(action, gigabytes: amount, settings: model.settings,
                                      source: EngineLocator.resolve(settings: model.settings))
    }

    private func reload() async throws {
        let fresh: CacheStatus = try await run(.status)
        status.value = fresh
        quota.value = Int((Double(fresh.quotaBytes) / 1_000_000_000).rounded())
    }

    private func refresh() { perform { try await reload() } }

    private func preview() {
        let requested = amount.value
        perform {
            let result: CacheReclaim = try await run(.preview, amount: requested)
            guard result.snapshots > 0 else { message.value = "No eligible snapshots to reclaim."; return }
            let shortfall = result.shortfallBytes > 0 ? " Protected states leave a \(gb(result.shortfallBytes)) shortfall." : ""
            confirmation.value = Confirmation(action: .reclaim, amount: requested,
                detail: "Expected to reclaim \(gb(result.reclaimedBytes)) from \(result.snapshots) snapshots.\(shortfall)")
        }
    }

    private func execute(_ action: CacheManagement.Action, amount: Int?) {
        perform {
            let result: CacheReclaim = try await run(action, amount: amount)
            message.value = cleanupText(result)
            if action == .quota, let amount {
                if result.shortfallBytes == 0 {
                    model.settings.cacheQuotaGB = amount
                    message.value = "Quota updated to \(amount) GB. \(cleanupText(result))"
                } else {
                    message.value = "Quota could not be lowered. \(cleanupText(result))"
                }
            }
            try await reload()
        }
    }

    private func perform(_ action: @escaping @MainActor () async throws -> Void) {
        guard !working.value else { return }
        working.value = true
        Task {
            defer { working.value = false }
            do { try await action() }
            catch { message.value = error.localizedDescription }
        }
    }

    private func chooseCache() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.canCreateDirectories = true
        if panel.runModal() == .OK, let url = panel.url {
            model.settings.cacheDirectory = url.path
            refresh()
        }
    }
}
