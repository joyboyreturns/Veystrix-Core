import SwiftUI
import VeystrixKit

enum SidebarItem: String, CaseIterable, Identifiable {
    case dashboard = "Dashboard", model = "Model", harness = "Agents", logs = "Logs", settings = "Settings"
    var id: String { rawValue }
    var symbol: String {
        switch self {
        case .dashboard: return "gauge.with.dots.needle.67percent"
        case .model: return "externaldrive"
        case .harness: return "point.3.connected.trianglepath.dotted"
        case .logs: return "text.alignleft"
        case .settings: return "gearshape"
        }
    }
}

struct ContentView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var selection = LocalState<SidebarItem?>(.dashboard)

    var body: some View {
        NavigationSplitView {
            List(SidebarItem.allCases, selection: $selection.value) { section in
                Label(section.rawValue, systemImage: section.symbol)
                    .tag(section)
                    .accessibilityLabel(section.rawValue)
            }
            .navigationSplitViewColumnWidth(min: 160, ideal: 190)
            .safeAreaInset(edge: .bottom) { SidebarStatus() }
        } detail: {
            switch selection.value ?? .dashboard {
            case .dashboard: DashboardView()
            case .model: ModelView()
            case .harness: HarnessView()
            case .logs: LogsView()
            case .settings: SettingsView().environmentObject(model.updates)
            }
        }
        .navigationTitle("Veystrix")
        .confirmationDialog("Load the model now?", isPresented: $model.pendingStartConfirmation, titleVisibility: .visible) {
            Button("Start Engine") { Task { await model.startNow() } }
            Button("Cancel", role: .cancel) {}
        } message: {
            Text("Loading uses most of this Mac's memory and reads from the SSD. Plug in and close memory-heavy apps first. Safety guards can refuse to start; that protects your Mac.")
        }
        .confirmationDialog("Restart the engine?", isPresented: $model.pendingRestartConfirmation, titleVisibility: .visible) {
            Button("Restart") { Task { await model.restartNow() } }
            Button("Cancel", role: .cancel) {}
        } message: { Text("The engine stops gracefully, then loads the model again.") }
        .alert("Veystrix", isPresented: Binding(get: { model.alertMessage != nil }, set: { if !$0 { model.alertMessage = nil } })) {
            Button("OK") { model.alertMessage = nil }
        } message: { Text(model.alertMessage ?? "") }
    }
}

struct SidebarStatus: View {
    @EnvironmentObject var engine: EngineController
    var body: some View {
        HStack(spacing: 8) {
            StatusBadge(state: engine.state)
            Spacer()
        }
        .padding(10)
        .background(.bar)
    }
}

/// State is shown with text and a symbol, never by color alone.
struct StatusBadge: View {
    let state: EngineState
    var body: some View {
        Label(state.label, systemImage: symbol)
            .foregroundStyle(tint)
            .font(.callout.weight(.semibold))
            .accessibilityLabel("Engine \(state.label)")
    }
    private var symbol: String {
        switch state {
        case .stopped: return "stop.circle"
        case .starting: return "hourglass"
        case .running: return "checkmark.circle.fill"
        case .stopping: return "hourglass.bottomhalf.filled"
        case .error: return "exclamationmark.triangle.fill"
        }
    }
    private var tint: Color {
        switch state {
        case .running: return .green
        case .error: return .red
        case .starting, .stopping: return .orange
        case .stopped: return .secondary
        }
    }
}
