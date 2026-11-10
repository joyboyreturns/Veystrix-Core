import SwiftUI
import AppKit
import VeystrixKit

@main
struct VeystrixApp: App {
    @NSApplicationDelegateAdaptor(AppDelegate.self) private var delegate
    @StateObject private var model = AppModel()

    var body: some Scene {
        WindowGroup("Veystrix", id: "main") {
            ContentView()
                .environmentObject(model)
                .environmentObject(model.engine)
                .environmentObject(model.updates)
                .frame(minWidth: 880, minHeight: 560)
                .onAppear { delegate.model = model }
        }
        .commands {
            CommandGroup(after: .appInfo) {
                Button("Check for Updates…") { model.updates.checkForUpdates() }
                    .disabled(!model.updates.canCheckForUpdates)
            }
            CommandMenu("Engine") {
                Button("Start Engine") { model.requestStart() }
                    .keyboardShortcut("r", modifiers: [.command, .shift])
                    .disabled(model.engine.state.isActive)
                Button("Stop Engine") { Task { await model.engine.stop() } }
                    .keyboardShortcut(".", modifiers: [.command])
                    .disabled(model.engine.state == .stopped)
                Button("Restart Engine") { model.requestRestart() }
                    .disabled(model.engine.state != .running)
            }
        }
        Settings {
            SettingsView()
                .environmentObject(model)
                .environmentObject(model.updates)
                .frame(width: 560)
        }
    }
}

/// Makes sure quitting the app never leaves a model process holding memory.
final class AppDelegate: NSObject, NSApplicationDelegate {
    weak var model: AppModel?
    func applicationShouldTerminateAfterLastWindowClosed(_ sender: NSApplication) -> Bool { false }
    @MainActor func applicationWillTerminate(_ notification: Notification) {
        model?.engine.shutdownForQuit()
    }
}
