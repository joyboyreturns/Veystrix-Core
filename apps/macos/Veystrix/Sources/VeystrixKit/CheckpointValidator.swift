import Foundation

/// Result of checking a model folder. Validation reads only small JSON metadata files and `stat`s the weight
/// shards. It never opens or reads a weight file, so it cannot load a model or use significant memory.
public struct CheckpointReport: Equatable, Sendable {
    public enum Verdict: Equatable, Sendable { case compatible, incomplete, incompatible, notFound }
    public var verdict: Verdict
    public var path: String
    public var modelType: String?
    public var shardCount: Int = 0
    public var totalBytes: Int64 = 0
    public var missingFiles: [String] = []
    public var notes: [String] = []

    public var isUsable: Bool { verdict == .compatible }
    /// Approximate disk size, for display.
    public var humanSize: String { ByteCountFormatter.string(fromByteCount: totalBytes, countStyle: .file) }
}

/// Mirrors the engine's own rules (`macqwen/checkpoints.py: flashnext_compatible`) so the app and the engine agree.
public enum CheckpointValidator {
    static let tokenizerFiles = ["tokenizer.json", "tokenizer_config.json"]
    static let supportedModelTypes: Set<String> = ["qwen4_exp", "qwen4_exp_text"]

    public static func validate(directory: URL, fileManager: FileManager = .default) -> CheckpointReport {
        var report = CheckpointReport(verdict: .notFound, path: directory.path)
        var isDir: ObjCBool = false
        guard fileManager.fileExists(atPath: directory.path, isDirectory: &isDir), isDir.boolValue else {
            report.notes.append("The folder does not exist.")
            return report
        }
        report.verdict = .incompatible

        let config = readJSON(directory.appendingPathComponent("config.json"))
        if config == nil { report.missingFiles.append("config.json") }
        for name in tokenizerFiles where !fileManager.fileExists(atPath: directory.appendingPathComponent(name).path) {
            report.missingFiles.append(name)
        }

        var types = Set<String>()
        if let config {
            if let t = config["model_type"] as? String { types.insert(t) }
            for key in ["text_config", "llm_config"] {
                if let nested = config[key] as? [String: Any], let t = nested["model_type"] as? String { types.insert(t) }
            }
            report.modelType = types.sorted().joined(separator: ", ")
            if types.isDisjoint(with: supportedModelTypes) {
                report.notes.append("Unsupported model type (\(report.modelType ?? "unknown")). This Veystrix build supports the Qwen3.8 Flash-Next family (qwen4_exp).")
                return report
            }
        }

        let indexURL = directory.appendingPathComponent("model.safetensors.index.json")
        guard let index = readJSON(indexURL), let weightMap = index["weight_map"] as? [String: Any], !weightMap.isEmpty else {
            report.missingFiles.append("model.safetensors.index.json")
            report.verdict = .incomplete
            report.notes.append("The weight index is missing or empty.")
            return report
        }
        let shardNames = Set(weightMap.values.compactMap { $0 as? String })
        guard shardNames.count == weightMap.count || !shardNames.isEmpty, shardNames.allSatisfy(isSaneShardName) else {
            report.verdict = .incomplete
            report.missingFiles.append("model.safetensors.index.json (invalid shard names)")
            report.notes.append("The weight index names a file that is not a plain .safetensors file name. It is refused for safety.")
            return report
        }
        report.shardCount = shardNames.count
        for shard in shardNames.sorted() {
            let url = directory.appendingPathComponent(shard)
            if let attrs = try? fileManager.attributesOfItem(atPath: url.path), let size = attrs[.size] as? NSNumber {
                report.totalBytes += size.int64Value
            } else {
                report.missingFiles.append(shard)
            }
        }
        if !report.missingFiles.isEmpty {
            report.verdict = .incomplete
            report.notes.append("\(report.missingFiles.count) required file(s) are missing. Finish or repair the download, then validate again.")
        } else {
            report.verdict = .compatible
            report.notes.append("All \(report.shardCount) weight files are present. Their contents were not read.")
        }
        return report
    }

    /// Same constraints as the engine: a plain `*.safetensors` file name, no separators, not hidden, not absolute.
    static func isSaneShardName(_ name: String) -> Bool {
        name.hasSuffix(".safetensors") && !name.contains("/") && !name.contains("\\")
            && name != "." && name != ".." && !name.hasPrefix(".") && !(name as NSString).isAbsolutePath
    }

    private static func readJSON(_ url: URL) -> [String: Any]? {
        guard let data = try? Data(contentsOf: url, options: .mappedIfSafe), data.count < 64 * 1024 * 1024 else { return nil }
        return (try? JSONSerialization.jsonObject(with: data)) as? [String: Any]
    }

    /// Plain-language licensing and compatibility reminder shown next to a checkpoint.
    public static let licenseNotice = """
    Veystrix does not include a model and does not download one. A checkpoint is licensed by its publisher, \
    not by Veystrix. Read that license before you download, use, modify or share the checkpoint. \
    Supported by this build: the Qwen3.8 Flash-Next family in MLX 4-bit format (architecture qwen4_exp).
    """
}
