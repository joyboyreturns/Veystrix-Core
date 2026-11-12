import Foundation

/// A launched engine process. Abstracted so the controller's state machine is testable without starting anything.
public protocol EngineProcess: AnyObject, Sendable {
    var isRunning: Bool { get }
    func interrupt()      // SIGINT: graceful, the supervisor handles it like Ctrl-C
    func terminate()      // SIGTERM
    func kill()           // SIGKILL
}

public protocol EngineProcessLauncher: Sendable {
    /// Start the plan. `onOutput` receives stdout/stderr text; `onExit` the exit status. Throws if the executable cannot run.
    func launch(_ plan: LaunchPlan,
                onOutput: @escaping @Sendable (String) -> Void,
                onExit: @escaping @Sendable (Int32) -> Void) throws -> EngineProcess
}

public final class SystemEngineProcess: EngineProcess, @unchecked Sendable {
    private let process: Process
    init(_ p: Process) { process = p }
    public var isRunning: Bool { process.isRunning }
    public func interrupt() { if process.isRunning { process.interrupt() } }
    public func terminate() { if process.isRunning { process.terminate() } }
    public func kill() { if process.isRunning { Darwin.kill(process.processIdentifier, SIGKILL) } }
}

public struct SystemEngineLauncher: EngineProcessLauncher {
    public init() {}
    public func launch(_ plan: LaunchPlan,
                       onOutput: @escaping @Sendable (String) -> Void,
                       onExit: @escaping @Sendable (Int32) -> Void) throws -> EngineProcess {
        let p = Process()
        p.executableURL = plan.executable
        p.arguments = plan.arguments
        p.environment = plan.environment
        p.currentDirectoryURL = plan.workingDirectory
        p.standardInput = FileHandle.nullDevice
        let out = Pipe(), err = Pipe()
        p.standardOutput = out
        p.standardError = err
        for pipe in [out, err] {
            pipe.fileHandleForReading.readabilityHandler = { handle in
                let data = handle.availableData
                if data.isEmpty { handle.readabilityHandler = nil; return }
                onOutput(String(decoding: data, as: UTF8.self))
            }
        }
        p.terminationHandler = { proc in
            out.fileHandleForReading.readabilityHandler = nil
            err.fileHandleForReading.readabilityHandler = nil
            onExit(proc.terminationStatus)
        }
        try p.run()
        if plan.keepMacAwake, p.isRunning {
            // Keep the router as our direct child so stop and restart wait on the port owner.
            // The assertion process exits on its own when the router exits.
            let assertion = Process()
            assertion.executableURL = URL(fileURLWithPath: "/usr/bin/caffeinate")
            assertion.arguments = ["-ims", "-w", String(p.processIdentifier)]
            assertion.standardInput = FileHandle.nullDevice
            assertion.standardOutput = FileHandle.nullDevice
            assertion.standardError = FileHandle.nullDevice
            do {
                try assertion.run()
            } catch {
                onOutput("Could not keep the Mac awake: \(error.localizedDescription)\n")
            }
        }
        return SystemEngineProcess(p)
    }
}

/// Where the engine runtime lives.
public enum EngineLocator {
    public static func resolve(settings: AppSettings, resourceURL: URL? = Bundle.main.resourceURL,
                               environment: [String: String] = ProcessInfo.processInfo.environment,
                               fileExists: (String) -> Bool = { FileManager.default.fileExists(atPath: $0) }) -> LaunchPlan.Source? {
        if let resourceURL {
            let root = resourceURL.appendingPathComponent("engine", isDirectory: true)
            if fileExists(root.appendingPathComponent("python/bin/python3").path) { return .bundled(engineRoot: root) }
        }
        let dev = !settings.developerRepoPath.isEmpty ? settings.developerRepoPath : (environment["VEYSTRIX_REPO"] ?? "")
        if !dev.isEmpty, fileExists((dev as NSString).appendingPathComponent(".venv/bin/python")) {
            return .developer(repoRoot: URL(fileURLWithPath: (dev as NSString).expandingTildeInPath))
        }
        return nil
    }
}
