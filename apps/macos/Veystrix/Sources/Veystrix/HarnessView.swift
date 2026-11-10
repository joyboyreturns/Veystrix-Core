import SwiftUI
import AppKit
import VeystrixKit

struct HarnessView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var picked = LocalState(Harness.hermes)

    var body: some View {
        ScrollView {
            VStack(alignment: .leading, spacing: 14) {
                Text("Agents").font(.largeTitle.bold())
                connectionCard
                Picker("Agent", selection: $picked.value) {
                    ForEach(Harness.allCases) { Text($0.rawValue).tag($0) }
                }
                .pickerStyle(.segmented)
                .accessibilityLabel("Choose an agent")
                let s = HarnessProfiles.snippet(for: picked.value, port: model.settings.port,
                                                modelID: engine.models?.data.first?.id ?? HarnessProfiles.defaultModelID)
                GroupBox("Where it goes") { Text(s.whereToPutIt).frame(maxWidth: .infinity, alignment: .leading).textSelection(.enabled) }
                GroupBox("Configuration") {
                    VStack(alignment: .leading, spacing: 8) {
                        Text(s.content).font(.system(.callout, design: .monospaced)).textSelection(.enabled)
                            .frame(maxWidth: .infinity, alignment: .leading)
                        Button { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(s.content, forType: .string) } label: {
                            Label("Copy", systemImage: "doc.on.doc")
                        }
                    }
                }
                ForEach(s.notes, id: \.self) { Label($0, systemImage: "info.circle").font(.callout).foregroundStyle(.secondary) }
                Text("Veystrix does not edit other apps' configuration files. Copy the settings you want.")
                    .font(.footnote).foregroundStyle(.secondary)
            }
            .padding(20)
        }
        .navigationTitle("Agents")
    }

    private var connectionCard: some View {
        GroupBox("Local endpoint") {
            VStack(alignment: .leading, spacing: 6) {
                LabeledContent("Base URL") { Text(model.baseURL).textSelection(.enabled) }
                LabeledContent("Model id") { Text(engine.models?.data.first?.id ?? HarnessProfiles.defaultModelID).textSelection(.enabled) }
                LabeledContent("API key") {
                    Text(model.settings.requireAPIKey ? "required (see Settings)" : "not required on loopback; any value works").foregroundStyle(.secondary)
                }
                LabeledContent("Connection") {
                    Label(connection.label, systemImage: connection == .engineStopped ? "pause.circle" : "network")
                }
            }
        }
    }

    private var connection: ConnectionState {
        switch engine.state {
        case .running: return .reachable(modelID: engine.models?.data.first?.id, contextWindow: engine.models?.data.first?.contextWindow)
        case .starting: return .unreachable
        default: return .engineStopped
        }
    }
}
