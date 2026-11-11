import SwiftUI
import AppKit
import VeystrixKit

private struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    var role: String
    var content: String
    var reasoningContent: String?
    var usage: ChatUsage?
    var finishReason: String?
}

private struct ChatConversation: Codable, Identifiable {
    var id = UUID()
    var model: String
    var messages: [ChatMessage] = []
    var title: String { messages.first(where: { $0.role == "user" }).map { String($0.content.prefix(48)) } ?? "New conversation" }
}

@MainActor private final class ChatStore: ObservableObject {
    @Published var conversations: [ChatConversation] = []
    @Published var selection: UUID?
    @Published var error: String?
    @Published var streaming = false
    private var task: Task<Void, Never>?
    private let file = Paths().applicationSupport.appendingPathComponent("chat-history.json")

    private static let streamSession: URLSession = {
        let config = URLSessionConfiguration.ephemeral
        config.timeoutIntervalForRequest = 7200
        config.timeoutIntervalForResource = 0
        config.waitsForConnectivity = false
        config.connectionProxyDictionary = [:]
        return URLSession(configuration: config)
    }()

    init() {
        if let data = try? Data(contentsOf: file), let saved = try? JSONDecoder().decode([ChatConversation].self, from: data) {
            conversations = saved
            selection = saved.first?.id
        }
    }

    func save() {
        do {
            try Paths().ensureDirectories()
            let data = try JSONEncoder().encode(conversations)
            try data.write(to: file, options: .atomic)
            try FileManager.default.setAttributes([.posixPermissions: 0o600], ofItemAtPath: file.path)
        } catch { self.error = "Could not save chat history: \(error.localizedDescription)" }
    }

    func new(model: String) {
        let c = ChatConversation(model: model)
        conversations.insert(c, at: 0); selection = c.id; save()
    }

    func delete() {
        guard !streaming, let id = selection else { return }
        conversations.removeAll { $0.id == id }; selection = conversations.first?.id; save()
    }

    func stop() { task?.cancel() }

    func send(_ text: String,
              model: String,
              port: Int,
              key: String?,
              reasoning: ChatReasoningEffort,
              maxOutputTokens: Int?,
              system: String) {
        guard !streaming, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        if selection == nil { new(model: model) }
        guard let id = selection, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].model == model else {
            error = "This conversation belongs to \(conversations[index].model). Start a new conversation for \(model)."
            return
        }
        conversations[index].messages.append(ChatMessage(role: "user", content: text))
        let priorMessages = conversations[index].messages.map {
            ChatWireMessage(role: $0.role, content: $0.content, reasoningContent: $0.reasoningContent)
        }
        conversations[index].messages.append(ChatMessage(role: "assistant", content: ""))
        save()
        streaming = true
        error = nil
        task = Task {
            defer { streaming = false; save(); task = nil }
            do {
                let wire = ChatRequestBuilder.wireMessages(systemPrompt: system, messages: priorMessages)
                let bodyObj = ChatRequestBuilder.payload(
                    model: model,
                    messages: wire,
                    reasoning: reasoning,
                    maxOutputTokens: maxOutputTokens,
                    stream: true
                )
                let bodyData = try JSONSerialization.data(withJSONObject: bodyObj)
                let request = ChatRequestBuilder.makeURLRequest(port: port, key: key, sessionID: id.uuidString, bodyData: bodyData)
                let (bytes, response) = try await Self.streamSession.bytes(for: request)
                guard let http = response as? HTTPURLResponse else {
                    throw NSError(domain: "Veystrix", code: 0, userInfo: [NSLocalizedDescriptionKey: "Invalid response from engine."])
                }
                guard (200..<300).contains(http.statusCode) else {
                    var errorBytes = Data()
                    for try await byte in bytes {
                        if errorBytes.count < 65536 { errorBytes.append(byte) }
                    }
                    let msg = ChatHTTPErrorParser.message(from: errorBytes, statusCode: http.statusCode)
                    throw NSError(domain: "Veystrix", code: http.statusCode, userInfo: [NSLocalizedDescriptionKey: msg])
                }
                for try await line in bytes.lines {
                    try Task.checkCancellation()
                    let events = ChatSSEParser.parse(line: line)
                    guard !events.isEmpty,
                          let convIdx = conversations.firstIndex(where: { $0.id == id }),
                          !conversations[convIdx].messages.isEmpty else { continue }
                    let msgIdx = conversations[convIdx].messages.count - 1
                    for event in events {
                        switch event {
                        case .reasoning(let chunk):
                            conversations[convIdx].messages[msgIdx].reasoningContent =
                                (conversations[convIdx].messages[msgIdx].reasoningContent ?? "") + chunk
                        case .content(let chunk):
                            conversations[convIdx].messages[msgIdx].content += chunk
                        case .finishReason(let reason):
                            conversations[convIdx].messages[msgIdx].finishReason = reason
                        case .usage(let usage):
                            conversations[convIdx].messages[msgIdx].usage = usage
                        case .error(let msg):
                            self.error = msg
                        case .done:
                            break
                        }
                    }
                }
            } catch is CancellationError {
                error = "Generation stopped by user."
            } catch {
                self.error = error.localizedDescription
            }
        }
    }
}

struct ChatView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var chat = ChatStore()
    @StateObject private var draft = LocalState("")
    @StateObject private var system = LocalState("")
    @StateObject private var effort = LocalState(ChatReasoningEffort.medium)
    @StateObject private var useExplicitLimit = LocalState(false)
    @StateObject private var explicitLimitText = LocalState("")

    private var activeModelID: String? {
        engine.health?.installedModelID ?? engine.health?.model
    }

    private var currentConversation: ChatConversation? {
        chat.conversations.first { $0.id == chat.selection }
    }

    private var lastUsage: ChatUsage? {
        currentConversation?.messages.last(where: { $0.usage != nil })?.usage
    }

    private var lastFinishReason: String? {
        currentConversation?.messages.last(where: { $0.finishReason != nil })?.finishReason
    }

    var body: some View {
        HSplitView {
            VStack(alignment: .leading) {
                HStack {
                    Button("New") { if let active = activeModelID { chat.new(model: active) } }
                        .disabled(chat.streaming || activeModelID == nil)
                    Button("Delete") { chat.delete() }
                        .disabled(chat.streaming || chat.selection == nil)
                }.padding(.horizontal)
                List(selection: $chat.selection) {
                    ForEach(chat.conversations) { c in
                        VStack(alignment: .leading, spacing: 2) {
                            Text(c.title).lineLimit(1)
                            Text(c.model).font(.caption2).foregroundStyle(.secondary)
                        }.tag(Optional(c.id))
                    }
                }.disabled(chat.streaming)
            }.frame(minWidth: 180, idealWidth: 220, maxWidth: 270)

            VStack(alignment: .leading, spacing: 10) {
                HStack(spacing: 12) {
                    VStack(alignment: .leading, spacing: 2) {
                        Text(activeModelID ?? "No model loaded").font(.headline)
                        Text(telemetrySummary)
                            .font(.caption)
                            .foregroundStyle(.secondary)
                    }
                    Spacer()
                    if engine.state == .starting {
                        ProgressView().controlSize(.small)
                        Text("Loading…")
                    }
                    Picker("Reasoning", selection: $effort.value) {
                        ForEach(ChatReasoningEffort.allCases) { level in
                            Text(level.label).tag(level)
                        }
                    }
                    .frame(width: 175)
                }

                HStack(spacing: 12) {
                    Text(useExplicitLimit.value ? "Output limit: explicit debug cap" : "Output limit: Unlimited (stops on EOS, context, safety, or cancel)")
                        .font(.caption)
                        .foregroundStyle(.secondary)
                    Spacer()
                    Toggle("Debug limit", isOn: $useExplicitLimit.value)
                        .toggleStyle(.checkbox)
                        .font(.caption)
                        .disabled(chat.streaming)
                    if useExplicitLimit.value {
                        TextField("Tokens", text: $explicitLimitText.value)
                            .textFieldStyle(.roundedBorder)
                            .frame(width: 90)
                            .disabled(chat.streaming)
                    }
                }

                DisclosureGroup("Optional system prompt") {
                    TextField("No system prompt", text: $system.value).textFieldStyle(.roundedBorder)
                }

                ScrollViewReader { _ in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            if let c = currentConversation {
                                ForEach(c.messages) { m in
                                    VStack(alignment: .leading, spacing: 6) {
                                        HStack {
                                            Text(m.role.capitalized).font(.caption.bold())
                                            if let finish = m.finishReason, m.role == "assistant" {
                                                Text("• \(finish)").font(.caption2).foregroundStyle(.secondary)
                                            }
                                            if let u = m.usage, m.role == "assistant" {
                                                Text("• \(u.completionTokens) out · \(u.promptTokens) in (\(u.cachedTokens) cached)")
                                                    .font(.caption2)
                                                    .foregroundStyle(.secondary)
                                            }
                                            Spacer()
                                            if m.role == "assistant" {
                                                Button("Copy") {
                                                    NSPasteboard.general.clearContents()
                                                    NSPasteboard.general.setString(m.content, forType: .string)
                                                }.buttonStyle(.borderless)
                                            }
                                        }
                                        if let reasoning = m.reasoningContent, !reasoning.isEmpty {
                                            DisclosureGroup("Reasoning") {
                                                Text(reasoning)
                                                    .font(.callout)
                                                    .foregroundStyle(.secondary)
                                                    .textSelection(.enabled)
                                                    .frame(maxWidth: .infinity, alignment: .leading)
                                            }
                                        }
                                        let placeholder = (m.reasoningContent?.isEmpty == false) ? "Thinking…" : "…"
                                        Text(m.content.isEmpty ? placeholder : m.content)
                                            .textSelection(.enabled)
                                    }
                                    .padding(10)
                                    .frame(maxWidth: .infinity, alignment: .leading)
                                    .background(.quaternary, in: RoundedRectangle(cornerRadius: 8))
                                    .id(m.id)
                                }
                            }
                        }.padding(.vertical)
                    }
                }

                if let error = chat.error {
                    Text(error).foregroundStyle(.red).textSelection(.enabled).font(.callout)
                }

                HStack(alignment: .bottom) {
                    TextField("Message", text: $draft.value, axis: .vertical)
                        .lineLimit(2...6)
                        .textFieldStyle(.roundedBorder)
                    if chat.streaming {
                        Button(role: .destructive) { chat.stop() } label: {
                            Label("Stop", systemImage: "stop.circle.fill")
                        }
                        .keyboardShortcut(.escape, modifiers: [])
                    } else {
                        Button("Send") { submitMessage() }
                            .disabled(engine.state != .running || activeModelID == nil || draft.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty)
                            .keyboardShortcut(.return, modifiers: .command)
                    }
                }
            }.padding().frame(minWidth: 420)
        }.navigationTitle("Chat")
    }

    private func submitMessage() {
        guard let active = activeModelID else { return }
        let parsedLimit: Int?
        do {
            parsedLimit = try ChatRequestBuilder.parseExplicitLimit(
                enabled: useExplicitLimit.value,
                rawText: explicitLimitText.value
            )
        } catch {
            chat.error = "\(error)"
            return
        }
        let text = draft.value
        draft.value = ""
        chat.error = nil
        chat.send(
            text,
            model: active,
            port: model.settings.port,
            key: model.effectiveKey,
            reasoning: effort.value,
            maxOutputTokens: parsedLimit,
            system: system.value
        )
    }

    private var telemetrySummary: String {
        var parts: [String] = []
        let contextLimit = engine.health?.contextLimit
        if let act = engine.health?.activity, act.active == true {
            if act.stage == "prefill", let done = act.prefillDone, let total = act.prefillTotal {
                parts.append("Prefilling \(done)/\(total) tok")
            } else if act.stage == "decode", let dec = act.decodeTokens {
                if let rate = act.decodeTokensPerSecond {
                    parts.append(String(format: "Decoding %d tok (%.1f tok/s)", dec, rate))
                } else {
                    parts.append("Decoding \(dec) tok")
                }
            } else if let stage = act.stage {
                parts.append(stage.capitalized)
            }
        }
        if let u = lastUsage {
            if let limit = contextLimit, limit > 0 {
                let pct = min(100, Int((Double(u.totalTokens) / Double(limit)) * 100.0))
                parts.append("Context \(u.totalTokens)/\(limit) tok (\(pct)%)")
            } else {
                parts.append("Tokens \(u.promptTokens) in + \(u.completionTokens) out = \(u.totalTokens)")
            }
            if u.cachedTokens > 0 { parts.append("Cached \(u.cachedTokens)") }
        } else if let limit = contextLimit, limit > 0 {
            parts.append("Context limit \(limit) tok")
        }
        if let finish = lastFinishReason, !chat.streaming {
            parts.append("Last stop: \(finish)")
        }
        if parts.isEmpty {
            return "Chat uses the running model. Loading another model is controlled in Model."
        }
        return parts.joined(separator: "  ·  ")
    }
}
