import SwiftUI

/// View-local state as an ObservableObject. Used instead of the `@State` property wrapper because, with the SDK 27
/// toolchain, `@State` is a macro whose plugin ships only with Xcode; this keeps the package buildable with just the
/// Command Line Tools as well as with Xcode.
final class LocalState<Value>: ObservableObject {
    @Published var value: Value
    init(_ value: Value) { self.value = value }
}
