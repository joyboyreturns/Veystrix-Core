import SwiftUI
import AppKit
import VeystrixKit

struct LogsView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var st = LogsState()
    private var lines: [LogStore.Line] { st.lines }
    private var filter: String { st.filter }
    private var showEngine: Bool { st.showEngine }
    private var showApp: Bool { st.showApp }
    private var exportMessage: String? { st.exportMessage }

    var body: some View {
        VStack(spacing: 0) {
            HStack {
                TextField("Filter", text: $st.filter).textFieldStyle(.roundedBorder).frame(maxWidth: 260)
                Toggle("App", isOn: $st.showApp).toggleStyle(.checkbox)
                Toggle("Engine", isOn: $st.showEngine).toggleStyle(.checkbox)
                Spacer()
                Button("Export Diagnostics…") { export() }
                    .help("Writes a sanitized report: no keys, prompts or user name")
                Button("Clear") { model.logs.clear() }
            }
            .padding(10)
            Divider()
            ScrollViewReader { proxy in
                List(visible) { line in
                    HStack(alignment: .top, spacing: 8) {
                        Text(line.date, format: .dateTime.hour().minute().second()).foregroundStyle(.secondary)
                        Text(line.source == .app ? "app" : "engine").foregroundStyle(line.source == .app ? .blue : .teal).frame(width: 44, alignment: .leading)
                        Text(line.text).textSelection(.enabled)
                    }
                    .font(.system(.caption, design: .monospaced))
                    .id(line.id)
                    .accessibilityElement(children: .combine)
                }
                .onChange(of: lines.last?.id) { _, new in if let new { proxy.scrollTo(new, anchor: .bottom) } }
            }
            if let exportMessage { Text(exportMessage).font(.footnote).padding(8).frame(maxWidth: .infinity, alignment: .leading) }
        }
        .navigationTitle("Logs")
        .task {
            while !Task.isCancelled {
                st.lines = model.logs.tail(1500)
                try? await Task.sleep(nanoseconds: 700_000_000)
            }
        }
    }

    private var visible: [LogStore.Line] {
        lines.filter { ($0.source == .app ? showApp : showEngine) && (filter.isEmpty || $0.text.localizedCaseInsensitiveContains(filter)) }
    }

    private func export() {
        let panel = NSSavePanel()
        panel.nameFieldStringValue = "veystrix-diagnostics.txt"
        panel.message = "Sanitized diagnostics. Review it before you share it."
        guard panel.runModal() == .OK, let url = panel.url else { return }
        do {
            try model.diagnosticsText().write(to: url, atomically: true, encoding: .utf8)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: url.path)
            st.exportMessage = "Saved \(model.paths.abbreviate(url.path))"
        } catch { st.exportMessage = "Could not save: \(error.localizedDescription)" }
    }
}

final class LogsState: ObservableObject {
    @Published var lines: [LogStore.Line] = []
    @Published var filter = ""
    @Published var showEngine = true
    @Published var showApp = true
    @Published var exportMessage: String?
}
