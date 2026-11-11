import SwiftUI
import AppKit
import VeystrixKit

private struct ChatMessage: Codable, Identifiable {
    var id = UUID()
    var role: String
    var content: String
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
    func send(_ text: String, model: String, port: Int, key: String?, effort: String, system: String) {
        guard !streaming, !text.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty else { return }
        if selection == nil { new(model: model) }
        guard let id = selection, let index = conversations.firstIndex(where: { $0.id == id }) else { return }
        guard conversations[index].model == model else { error = "This conversation belongs to \(conversations[index].model). Start a new conversation for \(model)."; return }
        conversations[index].messages.append(ChatMessage(role: "user", content: text))
        let messages = conversations[index].messages
        conversations[index].messages.append(ChatMessage(role: "assistant", content: ""))
        save(); streaming = true; error = nil
        task = Task {
            defer { streaming = false; save(); task = nil }
            do {
                var request = URLRequest(url: URL(string: "http://127.0.0.1:\(port)/v1/chat/completions")!)
                request.httpMethod = "POST"; request.timeoutInterval = 1800
                request.setValue("application/json", forHTTPHeaderField: "Content-Type")
                if let key { request.setValue("Bearer \(key)", forHTTPHeaderField: "Authorization") }
                request.setValue(id.uuidString, forHTTPHeaderField: "X-Session-Id")
                var wire = [["role": "system", "content": system]]
                wire += messages.map { ["role": $0.role, "content": $0.content] }
                request.httpBody = try JSONSerialization.data(withJSONObject: ["model": model, "messages": wire, "stream": true, "reasoning_effort": effort])
                let (bytes, response) = try await URLSession.shared.bytes(for: request)
                guard let http = response as? HTTPURLResponse, (200..<300).contains(http.statusCode) else {
                    throw NSError(domain: "Veystrix", code: (response as? HTTPURLResponse)?.statusCode ?? 0, userInfo: [NSLocalizedDescriptionKey: "Engine rejected the chat request (HTTP \((response as? HTTPURLResponse)?.statusCode ?? 0)). Check the engine logs."])
                }
                for try await line in bytes.lines {
                    try Task.checkCancellation()
                    guard line.hasPrefix("data:") else { continue }
                    let raw = line.dropFirst(5).trimmingCharacters(in: .whitespaces)
                    if raw == "[DONE]" { break }
                    guard let data = raw.data(using: .utf8),
                          let object = try? JSONSerialization.jsonObject(with: data) as? [String: Any],
                          let choices = object["choices"] as? [[String: Any]],
                          let delta = choices.first?["delta"] as? [String: Any],
                          let chunk = delta["content"] as? String,
                          let i = conversations.firstIndex(where: { $0.id == id }),
                          !conversations[i].messages.isEmpty else { continue }
                    conversations[i].messages[conversations[i].messages.count - 1].content += chunk
                }
            } catch is CancellationError {
                error = "Generation stopped."
            } catch { self.error = error.localizedDescription }
        }
    }
}

struct ChatView: View {
    @EnvironmentObject var model: AppModel
    @EnvironmentObject var engine: EngineController
    @StateObject private var chat = ChatStore()
    @StateObject private var draft = LocalState("")
    @StateObject private var system = LocalState("")
    @StateObject private var effort = LocalState("medium")
    var body: some View {
        HSplitView {
            VStack(alignment: .leading) {
                HStack {
                    Button("New") { if let active = engine.health?.model { chat.new(model: active) } }.disabled(chat.streaming || engine.health?.model == nil)
                    Button("Delete") { chat.delete() }.disabled(chat.streaming || chat.selection == nil)
                }.padding(.horizontal)
                List(selection: $chat.selection) {
                    ForEach(chat.conversations) { c in Text(c.title).tag(Optional(c.id)) }
                }.disabled(chat.streaming)
            }.frame(minWidth: 160, idealWidth: 200, maxWidth: 260)
            VStack(alignment: .leading, spacing: 12) {
                HStack {
                    Text(engine.health?.model ?? "No model loaded").font(.headline)
                    Spacer()
                    if engine.state == .starting { ProgressView().controlSize(.small); Text("Loading…") }
                    Picker("Reasoning", selection: $effort.value) { Text("Low").tag("low"); Text("Medium").tag("medium"); Text("High").tag("high") }.frame(width: 180)
                }
                Text("Chat uses the running model. Loading another model is controlled in Model. A new chat can replace the engine's current warm cache.").font(.caption).foregroundStyle(.secondary)
                DisclosureGroup("Optional system prompt") { TextField("No system prompt", text: $system.value).textFieldStyle(.roundedBorder) }
                ScrollViewReader { proxy in
                    ScrollView {
                        LazyVStack(alignment: .leading, spacing: 14) {
                            if let c = chat.conversations.first(where: { $0.id == chat.selection }) {
                                ForEach(c.messages) { m in
                                    VStack(alignment: .leading, spacing: 4) {
                                        HStack { Text(m.role.capitalized).font(.caption.bold()); Spacer(); if m.role == "assistant" { Button("Copy") { NSPasteboard.general.clearContents(); NSPasteboard.general.setString(m.content, forType: .string) }.buttonStyle(.borderless) } }
                                        Text(m.content.isEmpty ? "…" : m.content).textSelection(.enabled)
                                    }.padding(10).frame(maxWidth: .infinity, alignment: .leading).background(.quaternary, in: RoundedRectangle(cornerRadius: 8)).id(m.id)
                                }
                            }
                        }.padding(.vertical)
                    }
                }
                if let error = chat.error { Text(error).foregroundStyle(.red).textSelection(.enabled) }
                HStack(alignment: .bottom) {
                    TextField("Message", text: $draft.value, axis: .vertical).lineLimit(2...6).textFieldStyle(.roundedBorder)
                    if chat.streaming { Button("Stop") { chat.stop() } }
                    else { Button("Send") { if let active = engine.health?.model { let text = draft.value; draft.value = ""; chat.send(text, model: active, port: model.settings.port, key: model.effectiveKey, effort: effort.value, system: system.value) } }.disabled(engine.state != .running || engine.health?.model == nil || draft.value.trimmingCharacters(in: .whitespacesAndNewlines).isEmpty) }
                }
            }.padding().frame(minWidth: 380)
        }.navigationTitle("Chat")
    }
}
