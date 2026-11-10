import Foundation

/// Every location the app reads or writes, in one place.
///
/// Application state lives in ~/Library/Application Support/Veystrix. The engine's own
/// caches (prefix cache, saved sessions) are NOT moved by the app: they keep their existing
/// locations so that an update or a first launch never orphans hundreds of gigabytes.
public struct Paths: Sendable {
    public let home: URL
    public init(home: URL = FileManager.default.homeDirectoryForCurrentUser) { self.home = home }

    public var applicationSupport: URL { home.appendingPathComponent("Library/Application Support/Veystrix", isDirectory: true) }
    public var settingsFile: URL { applicationSupport.appendingPathComponent("settings.json") }
    public var apiKeyFile: URL { applicationSupport.appendingPathComponent("local-api-key") }
    public var logsDirectory: URL { home.appendingPathComponent("Library/Logs/Veystrix", isDirectory: true) }
    public var appLogFile: URL { logsDirectory.appendingPathComponent("veystrix-app.log") }
    public var engineLogFile: URL { logsDirectory.appendingPathComponent("veystrix-engine.log") }
    /// Where the engine's overnight supervisor writes its status (existing location, read-only for the app).
    public var supervisorStatusFile: URL { home.appendingPathComponent(".flashnext/logs/supervisor-status.json") }
    /// Engine default persistent prefix cache (unchanged so existing caches are reused).
    public var defaultCacheDirectory: URL { home.appendingPathComponent(".flashnext/prefix-cache", isDirectory: true) }
    /// Legacy locations written by earlier MACQWEN builds. Read-only for migration.
    public var legacyPreferences: URL { home.appendingPathComponent(".macqwen/preferences.json") }

    /// Create the app's own directories with owner-only permissions.
    public func ensureDirectories() throws {
        for dir in [applicationSupport, logsDirectory] {
            try FileManager.default.createDirectory(at: dir, withIntermediateDirectories: true,
                                                    attributes: [.posixPermissions: 0o700])
            try FileManager.default.setAttributes([.posixPermissions: 0o700], ofItemAtPath: dir.path)
        }
    }

    /// Abbreviate the home directory so paths can be shown and exported without the user name.
    public func abbreviate(_ path: String) -> String {
        let h = home.path
        if path == h { return "~" }
        if path.hasPrefix(h + "/") { return "~" + path.dropFirst(h.count) }
        return path
    }
}
