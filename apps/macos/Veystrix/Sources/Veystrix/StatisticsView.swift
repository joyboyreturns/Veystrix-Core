import SwiftUI
import VeystrixKit

/// Lifetime usage statistics: which model was used for how long, and the token counts. Everything comes from files in
/// Application Support that survive engine and app restarts; only the "Reset statistics" button removes them.
struct StatisticsView: View {
    @EnvironmentObject var engine: EngineController
    @StateObject private var range = LocalState(UsageRange.allTime)
    @StateObject private var includeWarmups = LocalState(false)
    @StateObject private var summary = LocalState<UsageSummary?>(nil)
    @StateObject private var confirmReset = LocalState(false)
    @StateObject private var message = LocalState<String?>(nil)
    @StateObject private var loadedSignature = LocalState("")
    @StateObject private var cachedRecords = LocalState<[UsageRecord]>([])
    @StateObject private var cachedLoaded = LocalState<[String: [String: Double]]>([:])

    private let store = UsageStore()

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 16) {
                header
                if let s = summary.value {
                    if s.models.isEmpty {
                        emptyState
                    } else {
                        totals(s.total)
                        modelSection(s)
                        footer(s)
                    }
                } else {
                    ProgressView("Reading statistics…")
                }
            }
            .padding(20)
        }
        .task {
            while !Task.isCancelled {
                await reload(force: false)
                try? await Task.sleep(nanoseconds: 5_000_000_000)
            }
        }
        .confirmationDialog("Reset all statistics?", isPresented: $confirmReset.value, titleVisibility: .visible) {
            Button("Reset Statistics", role: .destructive) { reset() }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("This permanently deletes the lifetime usage history stored on this Mac (token counts and times only). Your models, caches and settings are not touched.")
        }
    }

    // MARK: sections

    private var header: some View {
        VStack(alignment: .leading, spacing: 8) {
            HStack {
                VStack(alignment: .leading, spacing: 2) {
                    Text("Statistics").font(.largeTitle.weight(.semibold))
                    Text("Lifetime totals across every session. Counts and times only: no prompts or answers are stored.")
                        .font(.callout).foregroundStyle(.secondary)
                }
                Spacer()
                Picker("Period", selection: $range.value) {
                    ForEach(UsageRange.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.menu).frame(width: 170)
                .onChange(of: range.value) { _, _ in Task { await reload(force: true) } }
            }
            if let m = message.value { Text(m).font(.footnote).foregroundStyle(.secondary) }
        }
    }

    private var emptyState: some View {
        Banner(symbol: "chart.bar", tint: .blue, title: "No usage recorded yet",
               text: "Statistics appear after the engine has answered a request. Everything is stored on this Mac and kept between sessions. Requests made before this feature existed are not included.")
    }

    private func totals(_ t: ModelUsage) -> some View {
        Grid(alignment: .leading, horizontalSpacing: 16, verticalSpacing: 16) {
            GridRow {
                Tile(title: "Requests", value: UsageStats.count(t.requests), detail: "answered")
                Tile(title: "Total tokens", value: UsageStats.compact(t.totalTokens), detail: "\(UsageStats.count(t.totalTokens)) = input + output")
                Tile(title: "Time answering", value: UsageStats.duration(t.activeSeconds),
                     detail: "prefill \(UsageStats.duration(t.prefillSeconds)) · decode \(UsageStats.duration(t.decodeSeconds))")
            }
            GridRow {
                Tile(title: "Input tokens", value: UsageStats.compact(t.inputTokens), detail: "everything the client sent")
                Tile(title: "Cached tokens", value: UsageStats.compact(t.cachedTokens),
                     detail: t.cacheHitRate.map { String(format: "%.0f%% of input served from cache", $0 * 100) } ?? "no input yet")
                Tile(title: "Output tokens", value: UsageStats.compact(t.outputTokens),
                     detail: t.decodeTokensPerSecond.map { String(format: "%.1f tok/s average decode", $0) } ?? "")
            }
            GridRow {
                Tile(title: "Uncached input", value: UsageStats.compact(t.uncachedTokens),
                     detail: t.prefillTokensPerSecond.map { String(format: "%.0f tok/s average prefill", $0) } ?? "computed by the model")
                Tile(title: "Model loaded", value: UsageStats.duration(t.loadedSeconds), detail: "time an engine was ready")
                Tile(title: "Models used", value: "\(summaryModelCount)", detail: "in this period")
            }
        }
    }

    private var summaryModelCount: Int { summary.value?.models.count ?? 0 }

    private func modelSection(_ s: UsageSummary) -> some View {
        VStack(alignment: .leading, spacing: 10) {
            Text("By model").font(.title3.weight(.semibold))
            let maxActive = max(s.models.map(\.activeSeconds).max() ?? 0, 1)
            ForEach(s.models) { u in
                VStack(alignment: .leading, spacing: 8) {
                    HStack {
                        Text(UsageStats.displayName(model: u.model)).font(.headline)
                        Spacer()
                        Text("answering \(UsageStats.duration(u.activeSeconds)) · loaded \(UsageStats.duration(u.loadedSeconds))")
                            .font(.callout).foregroundStyle(.secondary)
                    }
                    ProgressView(value: u.activeSeconds, total: maxActive)
                        .accessibilityLabel("\(UsageStats.displayName(model: u.model)) share of answering time")
                    Grid(alignment: .leading, horizontalSpacing: 24, verticalSpacing: 4) {
                        GridRow { stat("Requests", UsageStats.count(u.requests)); stat("Input", UsageStats.count(u.inputTokens))
                                  stat("Cached", UsageStats.count(u.cachedTokens)); stat("Output", UsageStats.count(u.outputTokens))
                                  stat("Total", UsageStats.count(u.totalTokens)) }
                    }
                    HStack(spacing: 16) {
                        if let r = u.cacheHitRate { Text(String(format: "Cache hit %.0f%%", r * 100)) }
                        if let d = u.decodeTokensPerSecond { Text(String(format: "Decode %.1f tok/s", d)) }
                        if let p = u.prefillTokensPerSecond { Text(String(format: "Prefill %.0f tok/s", p)) }
                    }.font(.caption).foregroundStyle(.secondary)
                }
                .padding(12)
                .background(.quaternary.opacity(0.5), in: RoundedRectangle(cornerRadius: 10))
                .accessibilityElement(children: .contain)
            }
        }
    }

    private func stat(_ title: String, _ value: String) -> some View {
        VStack(alignment: .leading, spacing: 1) {
            Text(title).font(.caption).foregroundStyle(.secondary)
            Text(value).font(.callout.monospacedDigit())
        }
    }

    private func footer(_ s: UsageSummary) -> some View {
        VStack(alignment: .leading, spacing: 8) {
            if let since = s.since {
                Text("Counting since \(since.formatted(date: .abbreviated, time: .omitted)). Input tokens include cached tokens; total = input + output. \"Answering\" is the time spent on requests; \"loaded\" is the time a model was ready, so it can be much longer.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            Toggle("Include automatic warm-up requests", isOn: $includeWarmups.value)
                .onChange(of: includeWarmups.value) { _, _ in Task { await reload(force: true) } }
            if s.warmupRequestsExcluded > 0 {
                Text("\(UsageStats.count(s.warmupRequestsExcluded)) automatic warm-up request(s) are left out because they were not real use.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            HStack {
                Button("Show Files in Finder") { NSWorkspace.shared.activateFileViewerSelecting([store.directory]) }
                Button("Reset Statistics…", role: .destructive) { confirmReset.value = true }
            }
        }
    }

    // MARK: data

    private func reload(force: Bool) async {
        let store = self.store
        let signature = await Task.detached { store.signature() }.value
        if !force, signature == loadedSignature.value, summary.value != nil { return }
        if force || signature != loadedSignature.value {
            let (records, loaded) = await Task.detached { (store.loadRecords(), store.loadLoadedTime()) }.value
            cachedRecords.value = records
            cachedLoaded.value = loaded
            loadedSignature.value = signature
        }
        let records = cachedRecords.value, loaded = cachedLoaded.value
        let range = self.range.value, warm = includeWarmups.value
        summary.value = await Task.detached {
            UsageStats.summarize(records: records, loaded: loaded, range: range, includeWarmups: warm)
        }.value
    }

    private func reset() {
        do {
            try store.reset()
            message.value = "Statistics were reset."
            loadedSignature.value = ""
            Task { await reload(force: true) }
        } catch {
            message.value = "Could not reset statistics: \(error.localizedDescription)"
        }
    }
}
