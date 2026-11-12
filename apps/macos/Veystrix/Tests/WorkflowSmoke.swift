import Foundation

@main
struct WorkflowSmoke {
    struct CheckFailed: Error {}

    static func check(_ condition: Bool, _ message: String) throws {
        guard condition else {
            FileHandle.standardError.write(Data((message + "\n").utf8))
            throw CheckFailed()
        }
    }

    static func until(_ client: WorkflowClient, _ id: String,
                      _ condition: (WorkflowStatus) -> Bool) async throws -> WorkflowStatus {
        for _ in 0..<150 {
            let status = try await client.status(id)
            if condition(status) { return status }
            try await Task.sleep(for: .milliseconds(60))
        }
        try check(false, "Timed out waiting for workflow \(id)")
        throw CheckFailed()
    }

    static func main() async throws {
        let arguments = CommandLine.arguments
        try check(arguments.count == 4, "Usage: WorkflowSmoke RUNTIME_OR_CHECKOUT GIT_FIXTURE WORKFLOW_PARENT")
        let runtime = URL(fileURLWithPath: arguments[1], isDirectory: true)
        let source: LaunchPlan.Source = FileManager.default.isExecutableFile(atPath: runtime.appendingPathComponent("python/bin/python3").path)
            ? .bundled(engineRoot: runtime) : .developer(repoRoot: runtime)
        let repo = URL(fileURLWithPath: arguments[2], isDirectory: true)
        let root = URL(fileURLWithPath: arguments[3], isDirectory: true)
        let fixture = root.appendingPathComponent("mock.json")
        try Data(#"""
        {
          "plan": ["[{\"id\":\"inspect\",\"role\":\"explorer\",\"objective\":\"Inspect the snapshot\"}]"],
          "inspect": [{"delay_seconds": 2, "message": {"content": "Mock task finished."}}],
          "summary": ["Mock workflow finished."]
        }
        """#.utf8).write(to: fixture)
        let client = WorkflowClient(source: source,
                                    root: root.appendingPathComponent("Application Support/Veystrix/Workflows", isDirectory: true),
                                    port: 1, apiKey: nil)
        let launch = WorkflowLaunch(repository: repo, objective: "Check the mock workflow controls",
                                    mode: .overnight, mockFixture: fixture)
        let first = try await client.start(launch)
        _ = try await until(client, first) { $0.canPause && $0.tasks.contains { $0.id == "inspect" && $0.state == "running" } }
        try await client.perform(.pause, on: first)
        let paused = try await until(client, first) { $0.state == "paused" && $0.canResume }
        try check(paused.tasks.first { $0.id == "plan" }?.state == "completed", "Planner result was lost")
        try await client.perform(.resume, on: first)
        let finished = try await until(client, first) { $0.state == "completed" }
        try check(finished.tasks.allSatisfy(\.finished), "Progress did not show all tasks complete")
        let report = try await client.report(first)
        try check(report.contains("Mock workflow finished."), "The report did not include the summary")

        let second = try await client.start(launch)
        _ = try await until(client, second) { $0.runnerActive && $0.canCancel }
        try await client.perform(.cancel, on: second)
        _ = try await until(client, second) { $0.state == "cancelled" }
        let entries = try await client.history()
        try check(Set(entries.map(\.id)) == [first, second], "History did not list both workflow journals")
        print("Mock start, progress, pause, resume, report, cancel and history passed.")
    }
}
