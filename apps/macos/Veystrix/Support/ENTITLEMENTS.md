# Entitlements

Comments are not allowed inside the .entitlements plists (the code-signing parser rejects `--` in XML
comments), so the reasoning lives here.

**Veystrix.entitlements (the app).** Empty on purpose. The app runs with Hardened Runtime
(`codesign --options runtime`) and is **not sandboxed**: it has to launch the engine, read a model folder the
user chooses and write to the engine's existing cache folders (`~/.flashnext`). No other entitlement is requested.

**Engine.entitlements (the bundled Python runtime executables).** `allow-jit` and
`allow-unsigned-executable-memory`: CPython's ctypes/cffi closures and MLX need writable and executable memory
under Hardened Runtime. Every nested library is signed with the same identity, so library validation passes
and `disable-library-validation` is **not** requested. These two are the conservative guess; they must be
validated on a real Developer ID signed and notarized build (run the engine's import smoke test and a short
supervised decode on the owner's Mac), and any entitlement that turns out to be unnecessary must be removed.
