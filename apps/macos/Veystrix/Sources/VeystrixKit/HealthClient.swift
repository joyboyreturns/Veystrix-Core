import Foundation

/// Talks to the engine's local HTTP API. Used only while the engine is starting or running.
/// It never sends a prompt: only `GET /health` and `GET /v1/models`.
public struct HealthClient: Sendable {
    public let port: Int
    public let apiKey: String?
    public init(port: Int, apiKey: String?) { self.port = port; self.apiKey = apiKey }

    public var baseURL: URL { URL(string: "http://127.0.0.1:\(port)")! }

    private func request(_ path: String) -> URLRequest {
        var r = URLRequest(url: baseURL.appendingPathComponent(path), timeoutInterval: 3)
        r.cachePolicy = .reloadIgnoringLocalCacheData
        if let apiKey, !apiKey.isEmpty { r.setValue("Bearer \(apiKey)", forHTTPHeaderField: "Authorization") }
        return r
    }

    private static let session: URLSession = {
        let c = URLSessionConfiguration.ephemeral
        c.timeoutIntervalForRequest = 3
        c.waitsForConnectivity = false
        c.connectionProxyDictionary = [:]            // never route loopback through a proxy
        return URLSession(configuration: c)
    }()

    public enum Outcome: Equatable, Sendable {
        case healthy(HealthReport)
        case unhealthy(HealthReport)       // reachable, but not ready (loading, stopped, stalled)
        case unauthorized
        case unreachable
    }

    public func health() async -> Outcome {
        do {
            let (data, response) = try await Self.session.data(for: request("health"))
            guard let http = response as? HTTPURLResponse else { return .unreachable }
            if http.statusCode == 401 { return .unauthorized }
            let report = (try? JSONDecoder().decode(HealthReport.self, from: data)) ?? HealthReport()
            return report.isHealthy && http.statusCode == 200 ? .healthy(report) : .unhealthy(report)
        } catch {
            return .unreachable
        }
    }

    public func models() async -> ModelsReport? {
        guard let (data, response) = try? await Self.session.data(for: request("v1/models")),
              (response as? HTTPURLResponse)?.statusCode == 200 else { return nil }
        return try? JSONDecoder().decode(ModelsReport.self, from: data)
    }
}
