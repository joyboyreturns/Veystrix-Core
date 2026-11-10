import SwiftUI
import AppKit
import VeystrixKit

struct ModelView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController

    var body: some View {
        Form {
            Section("Model") {
                Picker("Model", selection: Binding(get: { model.settings.selectedModel }, set: { model.selectModel($0) })) {
                    ForEach(ModelFamily.allCases) { Text($0.title).tag($0) }
                }
                .disabled(engine.state.isActive)
                Text(model.settings.selectedModel == .nemotron35
                     ? "Experimental. Needs the mlx-community affine 4-bit conversion. Nothing has been measured on real weights yet, so expect it to be slower than Flash-Next. Each model keeps its own folder."
                     : "The default model. Each model keeps its own folder; switching never changes the other one.")
                    .font(.callout).foregroundStyle(.secondary)
                Text("The model is chosen by you. Nothing switches models automatically, and only one model is loaded at a time.")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Model folder") {
                LabeledContent("Folder") {
                    Text(model.settings.activeCheckpointPath.isEmpty ? "None chosen" : model.paths.abbreviate(model.settings.activeCheckpointPath))
                        .textSelection(.enabled).foregroundStyle(model.settings.activeCheckpointPath.isEmpty ? .secondary : .primary)
                }
                HStack {
                    Button("Choose Folder…") { choose() }.disabled(engine.state.isActive)
                    Button("Validate Again") { model.revalidateCheckpoint() }.disabled(model.settings.activeCheckpointPath.isEmpty)
                    Button("Show in Finder") {
                        NSWorkspace.shared.activateFileViewerSelecting([URL(fileURLWithPath: model.settings.activeCheckpointPath)])
                    }.disabled(model.settings.activeCheckpointPath.isEmpty)
                }
            }
            if let c = model.checkpoint {
                Section("Validation (weights are not read)") {
                    LabeledContent("Result") { Label(c.verdictText.capitalized, systemImage: c.isUsable ? "checkmark.seal.fill" : "xmark.octagon.fill").foregroundStyle(c.isUsable ? .green : .red) }
                    LabeledContent("Architecture", value: c.modelType ?? "unknown")
                    LabeledContent("Weight files", value: "\(c.shardCount)")
                    LabeledContent("Disk space used by the model", value: c.humanSize)
                    if !c.missingFiles.isEmpty {
                        LabeledContent("Missing") { Text(c.missingFiles.prefix(8).joined(separator: ", ")).foregroundStyle(.red) }
                    }
                    ForEach(c.notes, id: \.self) { Text($0).font(.callout).foregroundStyle(.secondary) }
                }
            }
            Section("Disk and memory") {
                let snap = MemoryProbe.snapshot()
                LabeledContent("Free disk space", value: snap.cacheVolumeFreeBytes.map { ByteCountFormatter.string(fromByteCount: $0, countStyle: .file) } ?? "unknown")
                Text("Keep at least 20 GB free beyond the model and the persistent cache quota (\(model.settings.cacheQuotaGB) GB by default).")
                    .font(.callout).foregroundStyle(.secondary)
            }
            Section("Compatibility and license") {
                Text(CheckpointValidator.licenseNotice).font(.callout)
                Text("Veystrix never downloads a model. Obtain a checkpoint from its publisher, and read its license first.")
                    .font(.callout).foregroundStyle(.secondary)
            }
        }
        .formStyle(.grouped)
        .navigationTitle("Model")
    }

    private func choose() {
        let panel = NSOpenPanel()
        panel.canChooseDirectories = true
        panel.canChooseFiles = false
        panel.allowsMultipleSelection = false
        panel.prompt = "Choose"
        panel.message = "Choose the folder that contains the model (config.json, tokenizer files and *.safetensors)."
        if panel.runModal() == .OK, let url = panel.url { model.chooseCheckpoint(url) }
    }
}
