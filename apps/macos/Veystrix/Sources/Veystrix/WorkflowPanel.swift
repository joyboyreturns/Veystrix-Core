import AppKit
import SwiftUI
import VeystrixKit

struct WorkflowPanel: View {
    @EnvironmentObject private var app: AppModel
    @EnvironmentObject private var engine: EngineController

    let models: [InstalledModelsReport.Entry]
    private struct Form {
        var repository: URL?
        var objective = ""
        var mode = WorkflowMode.swarm
        var mock = true
        var allowEdits = false
        var primaryModel = ""
        var executorModel = ""
        var agents = 3
        var minutes = 60
    }

    private struct Display {
        var history: [WorkflowEntry] = []
        var selected = ""
        var status: WorkflowStatus?
        var operating = false
        var pendingRunner: String?
        var error: String?
        var pollError: String?
        var report: String?
        var showReport = false
    }

    @StateObject private var form = LocalState(Form())
    @StateObject private var display = LocalState(Display())
    private var waitingForRunner: Bool { display.value.pendingRunner == display.value.selected }

    private var client: WorkflowClient {
        WorkflowClient(source: EngineLocator.resolve(settings: app.settings),
                       root: app.paths.applicationSupport.appendingPathComponent("Workflows", isDirectory: true),
                       port: app.settings.port, apiKey: app.apiKey)
    }

    var body: some View {
        VStack(alignment: .leading, spacing: 12) {
            launchForm
            if !display.value.history.isEmpty || display.value.status != nil { progress }
            if let message = display.value.error ?? display.value.pollError {
                Label(message, systemImage: "exclamationmark.circle")
                    .foregroundStyle(.secondary).textSelection(.enabled)
            }
        }
        .task {
            await refreshHistory()
            while !Task.isCancelled {
                await refreshStatus()
                do { try await Task.sleep(for: .seconds(2)) }
                catch { break }
            }
        }
        .onChange(of: display.value.selected) { _, _ in
            display.value.status = nil
            Task { await refreshStatus() }
        }
        .onChange(of: models.map(\.id), initial: true) { _, ids in
            if !ids.contains(form.value.primaryModel) { form.value.primaryModel = models.first(where: \.loaded)?.id ?? ids.first ?? "" }
            if !ids.contains(form.value.executorModel) { form.value.executorModel = "" }
        }
        .sheet(isPresented: $display.value.showReport) { reportSheet }
    }

    private var launchForm: some View {
        GroupBox("Native workflows") {
            VStack(alignment: .leading, spacing: 9) {
                Picker("Mode", selection: $form.value.mode) {
                    ForEach(WorkflowMode.allCases) { Text($0.label).tag($0) }
                }.pickerStyle(.segmented)
                HStack {
                    Button("Choose Repository…", action: chooseRepository)
                    Text(form.value.repository?.path ?? "Choose a Git repository").lineLimit(1).truncationMode(.middle)
                        .foregroundStyle(.secondary)
                }
                TextField("Workflow objective", text: $form.value.objective, axis: .vertical)
                    .lineLimit(2...5)
                Toggle("Use mock backend", isOn: $form.value.mock)
                if form.value.mock {
                    Text("Runs an offline demonstration against the committed repository.").font(.caption).foregroundStyle(.secondary)
                } else {
                    modelPickers
                    Toggle("Allow proposed edits in isolated copies", isOn: $form.value.allowEdits)
                }
                HStack {
                    Stepper("Agents: \(form.value.agents)", value: $form.value.agents, in: 1...4)
                    Stepper("Time budget: \(form.value.minutes) min", value: $form.value.minutes, in: 1...1440, step: 1)
                }
                HStack {
                    Button("Start \(form.value.mode.label)") { start() }
                        .buttonStyle(.borderedProminent)
                        .disabled(display.value.operating || form.value.repository == nil || form.value.objective.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty
                                  || (!form.value.mock && (form.value.primaryModel.isEmpty || engine.state != .running)))
                    if display.value.operating { ProgressView().controlSize(.small) }
                    Spacer()
                    Button("Refresh History") { Task { await refreshHistory() } }
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private var modelPickers: some View {
        VStack(alignment: .leading, spacing: 8) {
            Picker("Architect model", selection: $form.value.primaryModel) {
                if models.isEmpty { Text("No models discovered").tag("") }
                ForEach(models) { Text($0.displayName).tag($0.id) }
            }
            Picker("Executor model", selection: $form.value.executorModel) {
                Text("Use architect model").tag("")
                ForEach(models.filter { $0.id != form.value.primaryModel }) { Text($0.displayName).tag($0.id) }
            }.onChange(of: form.value.primaryModel) { _, id in
                if id == form.value.executorModel { form.value.executorModel = "" }
            }
            if !form.value.executorModel.isEmpty {
                Text("Use a freshly started, idle engine for a two-model run. Veystrix unloads one model before loading the next.")
                    .font(.caption).foregroundStyle(.secondary)
            } else if engine.state != .running {
                Text("Start the engine to run with an installed model.").font(.caption).foregroundStyle(.secondary)
            }
        }
    }

    private var progress: some View {
        GroupBox("Workflow progress") {
            VStack(alignment: .leading, spacing: 9) {
                Picker("Workflow", selection: $display.value.selected) {
                    Text("Choose a workflow").tag("")
                    ForEach(display.value.history) { entry in Text(entry.objective).tag(entry.id) }
                }
                if let status = display.value.status {
                    LabeledContent("Status", value: stateLabel(status))
                    LabeledContent("Architect", value: modelName(status.settings.model))
                    if let executor = status.settings.executorModel {
                        LabeledContent("Executor", value: modelName(executor))
                    }
                    let finished = status.tasks.filter(\.finished).count
                    ProgressView(value: Double(finished), total: Double(max(1, status.tasks.count))) {
                        Text("\(finished) of \(status.tasks.count) tasks finished")
                    }
                    HStack {
                        Button("Pause") { perform(.pause) }.disabled(display.value.operating || !status.canPause)
                        Button("Resume") { perform(.resume) }.disabled(display.value.operating || waitingForRunner || !status.canResume)
                        Button("Cancel", role: .destructive) { perform(.cancel) }.disabled(display.value.operating || !status.canCancel)
                        Spacer()
                        Button("View Report") { loadReport() }.disabled(display.value.operating)
                    }
                    ForEach(status.tasks) { task in
                        Divider()
                        taskRow(task)
                    }
                }
            }.frame(maxWidth: .infinity, alignment: .leading)
        }
    }

    private func taskRow(_ task: WorkflowStatus.WorkItem) -> some View {
        VStack(alignment: .leading, spacing: 4) {
            LabeledContent(task.id) { Text(task.state.capitalized).foregroundStyle(.secondary) }
                .font(.callout.weight(.medium))
            Text(task.objective).font(.callout)
            Text("\(task.role.capitalized) · \(task.attempt) attempt(s)").font(.caption).foregroundStyle(.secondary)
            if !task.dependsOn.isEmpty {
                Text("Depends on \(task.dependsOn.joined(separator: ", "))").font(.caption).foregroundStyle(.secondary)
            }
            if let reason = task.result?.reason { Text(reason).font(.caption).textSelection(.enabled) }
        }
    }

    private var reportSheet: some View {
        VStack(alignment: .leading, spacing: 12) {
            HStack {
                Text("Workflow Report").font(.headline)
                Spacer()
                Button("Copy") {
                    NSPasteboard.general.clearContents()
                    NSPasteboard.general.setString(display.value.report ?? "", forType: .string)
                }
                Button("Show Files") {
                    if let folder = try? client.directory(display.value.selected) { NSWorkspace.shared.open(folder) }
                }
                Button("Done") { display.value.showReport = false }.keyboardShortcut(.cancelAction)
            }
            ScrollView {
                Text(display.value.report ?? "").textSelection(.enabled).font(.system(.callout, design: .monospaced))
                    .frame(maxWidth: .infinity, alignment: .leading)
            }
        }.padding(16).frame(minWidth: 540, minHeight: 320)
    }

    private func modelName(_ id: String) -> String {
        if id == "mock" { return "Mock" }
        return models.first(where: { $0.id == id })?.displayName ?? id
    }

    private func stateLabel(_ status: WorkflowStatus) -> String {
        if status.finished || status.state == "paused" { return status.state.capitalized }
        if status.control == "cancel" { return "Cancelling" }
        if status.control == "pause" { return "Pausing" }
        if waitingForRunner { return "Starting runner" }
        if status.state == "running" && !status.runnerActive { return "Runner interrupted" }
        return status.state.capitalized
    }

    private func chooseRepository() {
        let chooser = NSOpenPanel()
        chooser.canChooseDirectories = true
        chooser.canChooseFiles = false
        chooser.allowsMultipleSelection = false
        if chooser.runModal() == .OK { form.value.repository = chooser.url }
    }

    private func start() {
        guard let repository = form.value.repository else { return }
        let launch = WorkflowLaunch(repository: repository, objective: form.value.objective, mode: form.value.mode,
                                    model: form.value.mock ? "mock" : form.value.primaryModel,
                                    executorModel: form.value.mock || form.value.executorModel.isEmpty ? nil : form.value.executorModel,
                                    seconds: form.value.minutes * 60,
                                    agents: form.value.agents, allowEdits: !form.value.mock && form.value.allowEdits)
        performOperation {
            let id = try await client.start(launch)
            await refreshHistory()
            display.value.selected = id
            trackRunnerStart(id)
            await refreshStatus()
        }
    }

    private func perform(_ action: WorkflowClient.Action) {
        let id = display.value.selected
        performOperation {
            try await client.perform(action, on: id)
            if action == .resume { trackRunnerStart(id) }
            await refreshStatus()
        }
    }

    private func trackRunnerStart(_ id: String) {
        display.value.pendingRunner = id
        Task {
            try? await Task.sleep(for: .seconds(2))
            if display.value.pendingRunner == id { display.value.pendingRunner = nil }
        }
    }

    private func loadReport() {
        let id = display.value.selected
        performOperation {
            display.value.report = try await client.report(id)
            if display.value.selected == id { display.value.showReport = true }
        }
    }

    private func performOperation(_ operation: @escaping @MainActor () async throws -> Void) {
        display.value.operating = true
        display.value.error = nil
        Task {
            defer { display.value.operating = false }
            do { try await operation() }
            catch { display.value.error = error.localizedDescription }
        }
    }

    private func refreshHistory() async {
        do {
            display.value.history = try await client.history()
            if display.value.selected.isEmpty, let newest = display.value.history.first { display.value.selected = newest.id }
        } catch { display.value.error = error.localizedDescription }
    }

    private func refreshStatus() async {
        let id = display.value.selected
        guard !id.isEmpty else { return }
        do {
            let updated = try await client.status(id)
            guard display.value.selected == id else { return }
            display.value.status = updated
            display.value.pollError = nil
            if display.value.pendingRunner == id && (updated.runnerActive || updated.finished) { display.value.pendingRunner = nil }
        } catch {
            if display.value.selected == id { display.value.pollError = error.localizedDescription }
        }
    }
}
