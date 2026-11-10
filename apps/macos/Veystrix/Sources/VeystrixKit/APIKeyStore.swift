import Foundation
import Security

/// A local API key for the engine, kept in the macOS Keychain (falling back to an owner-only file).
/// The engine accepts any key on loopback unless "Require API key" is on; the key is never written to logs.
public struct APIKeyStore: Sendable {
    public static let service = "io.github.joyboyreturns.veystrix"
    public static let account = "local-api-key"
    public let paths: Paths
    public init(paths: Paths = Paths()) { self.paths = paths }

    /// Existing key, or nil. Does not create one.
    public func load() -> String? {
        if let k = keychainRead() { return k }
        if let data = try? Data(contentsOf: paths.apiKeyFile), let s = String(data: data, encoding: .utf8) {
            let t = s.trimmingCharacters(in: .whitespacesAndNewlines)
            return t.isEmpty ? nil : t
        }
        return nil
    }

    /// Existing key or a freshly generated 256-bit random one.
    @discardableResult
    public func loadOrCreate() -> String {
        if let k = load() { return k }
        var bytes = [UInt8](repeating: 0, count: 32)
        let status = SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes)
        precondition(status == errSecSuccess, "system random number generator failed")
        let key = "vx-" + bytes.map { String(format: "%02x", $0) }.joined()
        store(key)
        return key
    }

    public func rotate() -> String {
        delete()
        return loadOrCreate()
    }

    private func store(_ key: String) {
        let base: [String: Any] = [kSecClass as String: kSecClassGenericPassword,
                                   kSecAttrService as String: Self.service, kSecAttrAccount as String: Self.account]
        var add = base
        add[kSecValueData as String] = Data(key.utf8)
        add[kSecAttrAccessible as String] = kSecAttrAccessibleWhenUnlockedThisDeviceOnly
        if SecItemAdd(add as CFDictionary, nil) != errSecSuccess {
            // Keychain unavailable (for example an unsigned development build in a restricted session): owner-only file.
            try? paths.ensureDirectories()
            FileManager.default.createFile(atPath: paths.apiKeyFile.path, contents: Data(key.utf8),
                                           attributes: [.posixPermissions: 0o600])
        }
    }

    private func keychainRead() -> String? {
        let query: [String: Any] = [kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: Self.service,
                                    kSecAttrAccount as String: Self.account, kSecReturnData as String: true,
                                    kSecMatchLimit as String: kSecMatchLimitOne]
        var out: CFTypeRef?
        guard SecItemCopyMatching(query as CFDictionary, &out) == errSecSuccess, let data = out as? Data else { return nil }
        return String(data: data, encoding: .utf8)
    }

    private func delete() {
        SecItemDelete([kSecClass as String: kSecClassGenericPassword, kSecAttrService as String: Self.service,
                       kSecAttrAccount as String: Self.account] as CFDictionary)
        try? FileManager.default.removeItem(at: paths.apiKeyFile)
    }
}
