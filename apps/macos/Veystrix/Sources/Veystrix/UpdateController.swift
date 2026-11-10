import SwiftUI
import Sparkle

/// Sparkle 2 wrapper. The updater starts only when the build carries an EdDSA public key and an https feed URL.
/// A development build has neither, so it never contacts any server and never installs anything.
@MainActor
final class UpdateController: NSObject, ObservableObject, SPUUpdaterDelegate {
    private var controller: SPUStandardUpdaterController?
    @Published private(set) var isConfigured = false
    @Published private(set) var summary = "Updates are not configured in this build."

    var canCheckForUpdates: Bool { controller?.updater.canCheckForUpdates ?? false }

    var automaticallyChecks: Bool {
        get { controller?.updater.automaticallyChecksForUpdates ?? false }
        set { objectWillChange.send(); controller?.updater.automaticallyChecksForUpdates = newValue }
    }

    override init() {
        super.init()
        let info = Bundle.main
        let key = (info.object(forInfoDictionaryKey: "SUPublicEDKey") as? String)?.trimmingCharacters(in: .whitespaces) ?? ""
        let feed = (info.object(forInfoDictionaryKey: "SUFeedURL") as? String) ?? ""
        guard !key.isEmpty, feed.hasPrefix("https://") else {
            summary = key.isEmpty ? "Updates are off: this build has no update signing key (development build)."
                                  : "Updates are off: the update feed is not an https URL."
            return
        }
        // startingUpdater: true; automatic checks stay off until the user turns them on (SUEnableAutomaticChecks = NO).
        controller = SPUStandardUpdaterController(startingUpdater: true, updaterDelegate: self, userDriverDelegate: nil)
        isConfigured = true
        summary = "Sparkle \(Bundle(for: SPUUpdater.self).infoDictionary?["CFBundleShortVersionString"] as? String ?? "2"); EdDSA-verified updates from \(feed)"
    }

    func checkForUpdates() { controller?.checkForUpdates(nil) }

    // Never install a build older than the one running: Sparkle compares sparkle:version with CFBundleVersion
    // and refuses to go backwards; this additionally rejects any item that is not strictly newer.
    nonisolated func updater(_ updater: SPUUpdater, shouldProceedWithUpdate updateItem: SUAppcastItem, updateCheck: SPUUpdateCheck) throws {
        let current = Bundle.main.object(forInfoDictionaryKey: "CFBundleVersion") as? String ?? "0"
        if updateItem.versionString.compare(current, options: .numeric) != .orderedDescending {
            throw NSError(domain: "io.github.joyboyreturns.veystrix.update", code: 1,
                          userInfo: [NSLocalizedDescriptionKey: "Update is not newer than the installed version; refusing to downgrade."])
        }
    }
}
