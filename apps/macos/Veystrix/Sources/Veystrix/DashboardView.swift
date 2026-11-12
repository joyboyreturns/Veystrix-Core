import SwiftUI
import VeystrixKit

struct DashboardView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var snap = LocalState(MemoryProbe.snapshot())
    private var snapshot: MemorySnapshot { snap.value }

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                if case .error(let message) = engine.state {
                    Banner(symbol: "exclamationmark.triangle.fill", tint: .red, title: "The engine stopped with an error", text: message)
                }
                if let stop = engine.lastSafetyStop {
                    Banner(symbol: "shield.lefthalf.filled", tint: .orange, title: "Safety guard",
                           text: "\(stop)\nA guard stop is a deliberate refusal that protects your Mac. Free memory or plug in, then start again. Do not look for a way around it.")
                }
                ForEach(engine.advisories, id: \.self) { Banner(symbol: "info.circle", tint: .blue, title: "Before you start", text: $0) }
                Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 16) {
                    GridRow {
                        Tile(title: "Engine", value: engine.state.label, detail: engine.isExternallyManaged ? "Started outside the app" : model.runtimeDescription)
                        Tile(title: "Endpoint", value: "127.0.0.1:\(model.settings.port)", detail: "Loopback only")
                    }
                    GridRow {
                        Tile(title: "Startup model", value: checkpointName, detail: model.checkpoint.map { "\($0.humanSize) · \($0.verdictText)" } ?? "No model folder chosen")
                        Tile(title: "Model", value: engine.health?.installedModelID ?? engine.health?.model ?? "—",
                             detail: contextDetail)
                    }
                    GridRow {
                        Tile(title: "Memory", value: MemorySnapshot.gib(snapshot.availableBytes) + " available",
                             detail: "of \(MemorySnapshot.gib(snapshot.physicalBytes)) · swap \(MemorySnapshot.gib(snapshot.swapUsedBytes))")
                        Tile(title: "Power", value: powerText, detail: snapshot.lowPowerMode ? "Low Power Mode is ON" : "Low Power Mode off")
                    }
                    GridRow {
                        Tile(title: "Persistent cache", value: cacheText, detail: "Quota \(model.settings.cacheQuotaGB) GB")
                        Tile(title: "Request", value: requestText, detail: queueText)
                    }
                }
                activityPanel
                Text("Veystrix version \(Bundle.main.shortVersion) (\(Bundle.main.buildNumber)). Engine telemetry comes from the engine's own /health endpoint when it is running.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            .padding(20)
        }
        .task {
            while !Task.isCancelled {
                snap.value = MemoryProbe.snapshot()
                await engine.refreshModels(port: model.settings.port, apiKey: model.effectiveKey)
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
    }

    private var header: some View {
        HStack(alignment: .center, spacing: 12) {
            VStack(alignment: .leading) {
                Text("Dashboard").font(.largeTitle.bold())
                StatusBadge(state: engine.state)
            }
            Spacer()
            Button { model.requestStart() } label: { Label("Start Engine", systemImage: "play.fill") }
                .buttonStyle(.borderedProminent)
                .disabled(engine.state.isActive || engine.state == .stopping || engine.isExternallyManaged || model.checkpoint?.isUsable != true)
                .help(model.checkpoint?.isUsable == true ? "Load the model and start the local server" : "Choose a valid model folder first")
                .accessibilityHint("Loads the model. Nothing starts automatically.")
            Button { Task { await engine.stop() } } label: { Label("Stop", systemImage: "stop.fill") }
                .disabled(engine.state == .stopped || engine.isExternallyManaged)
            Button { model.requestRestart() } label: { Label("Restart", systemImage: "arrow.clockwise") }
                .disabled(engine.state != .running || engine.isExternallyManaged)
        }
    }

    private var checkpointName: String {
        model.settings.activeCheckpointPath.isEmpty ? "—" : (model.settings.activeCheckpointPath as NSString).lastPathComponent
    }
    private var contextDetail: String {
        var parts: [String] = []
        if let l = engine.health?.contextLimit { parts.append("context \(l)") }
        if let c = engine.health?.maxColdPrefill { parts.append("max cold prefill \(c)") }
        return parts.isEmpty ? "Shown when the engine is running" : parts.joined(separator: " · ")
    }
    private var powerText: String {
        switch snapshot.onACPower { case true?: return "AC power"; case false?: return "Battery \(snapshot.batteryPercent.map { "\($0)%" } ?? "")"; default: return "Unknown" }
    }
    private var cacheText: String {
        if let c = engine.health?.cache?.values {
            if let bytes = c["ssd_cache_bytes"] { return ByteCountFormatter.string(fromByteCount: Int64(bytes), countStyle: .file) + " on disk" }
            if let used = c["disk_gb"] ?? c["disk_used_gb"] { return String(format: "%.1f GB on disk", used) }
            if let n = c["entries"] ?? c["disk_entries"] { return "\(Int(n)) entries" }
        }
        return engine.state == .running ? "No data yet" : "Not running"
    }
    private var activityPanel: some View {
        VStack(alignment: .leading, spacing: 12) {
            Text("Live inference").font(.headline)
            let activity = engine.health?.activity
            if engine.health?.busy == true, activity?.active == true {
                HStack {
                    Tile(title: "Stage", value: activity?.stage?.capitalized ?? "Preparing", detail: "Current request")
                    Tile(title: "Prompt size", value: activity?.promptTokens.map { "\($0.formatted()) tokens" } ?? "Calculating…", detail: "Total prompt, including cached tokens")
                }
                if let done = activity?.prefillDone, let total = activity?.prefillTotal {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Prefill: \(done.formatted()) / \(total.formatted()) tokens")
                            .font(.callout)
                        if total > 0 { ProgressView(value: min(Double(done) / Double(total), 1.0)) }
                        else { Text("No prefill required").font(.caption).foregroundStyle(.secondary) }
                    }
                } else {
                    Text("Prefill progress will appear when available.").font(.caption).foregroundStyle(.secondary)
                }
                HStack {
                    Tile(title: "Decode speed", value: activity?.decodeTokensPerSecond.map { String(format: "%.1f tokens/s", $0) } ?? "—", detail: "Average for this request")
                    Tile(title: "Generated", value: activity?.decodeTokens.map { "\($0.formatted()) tokens" } ?? "—", detail: "Includes reasoning tokens")
                }
            } else if engine.health?.busy == true {
                Text("Working · \(engine.health?.worker?.stage?.capitalized ?? "Preparing")")
                Text("Live token progress requires the updated engine.").font(.caption).foregroundStyle(.secondary)
            } else {
                Text(engine.state == .running ? "No active request" : "Start the engine to see activity")
                    .foregroundStyle(.secondary)
            }
            if let cache = engine.health?.cache?.values {
                let used = cache["ssd_cache_bytes"]
                let budget = cache["ssd_cache_budget_bytes"]
                if let used, let budget, budget > 0 {
                    VStack(alignment: .leading, spacing: 4) {
                        Text("Current model cache / shared quota: \(ByteCountFormatter.string(fromByteCount: Int64(used), countStyle: .file)) / \(ByteCountFormatter.string(fromByteCount: Int64(budget), countStyle: .file))")
                            .font(.callout)
                        ProgressView(value: min(used / budget, 1.0))
                        if let entries = cache["ssd_cache_entries"] {
                            Text("\(Int(entries).formatted()) saved entries").font(.caption).foregroundStyle(.secondary)
                        }
                    }
                }
            }
        }
        .padding(16)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
    }
    private var requestText: String {
        guard engine.state == .running else { return "Idle" }
        return (engine.health?.busy ?? false) ? "Working" : "Idle"
    }
    private var queueText: String {
        guard let q = engine.health?.queue?.values, !q.isEmpty else { return "No queue data" }
        return q.sorted { $0.key < $1.key }.prefix(3).map { "\($0.key) \(Int($0.value))" }.joined(separator: " · ")
    }
}

extension CheckpointReport {
    var verdictText: String {
        switch verdict {
        case .compatible: return "compatible"
        case .incomplete: return "incomplete"
        case .incompatible: return "not supported"
        case .notFound: return "not found"
        }
    }
}

struct Tile: View {
    let title: String, value: String, detail: String
    var body: some View {
        VStack(alignment: .leading, spacing: 4) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.title3.weight(.semibold)).lineLimit(2).minimumScaleFactor(0.8)
            Text(detail).font(.caption).foregroundStyle(.secondary).lineLimit(3)
        }
        .frame(maxWidth: .infinity, alignment: .leading)
        .padding(12)
        .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .combine)
    }
}

struct Banner: View {
    let symbol: String, tint: Color, title: String, text: String
    var body: some View {
        HStack(alignment: .top, spacing: 10) {
            Image(systemName: symbol).foregroundStyle(tint).imageScale(.large)
            VStack(alignment: .leading, spacing: 2) {
                Text(title).font(.headline)
                Text(text).font(.callout).textSelection(.enabled)
            }
            Spacer(minLength: 0)
        }
        .padding(12)
        .background(tint.opacity(0.12), in: RoundedRectangle(cornerRadius: 10))
        .accessibilityElement(children: .combine)
    }
}
