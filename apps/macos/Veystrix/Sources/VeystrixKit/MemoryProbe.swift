import Foundation
import Darwin
import IOKit.ps

public struct MemorySnapshot: Equatable, Sendable {
    public var physicalBytes: UInt64
    /// Free + inactive + purgeable, a practical "available" estimate (the engine uses its own, stricter measurement).
    public var availableBytes: UInt64
    public var swapUsedBytes: UInt64
    public var swapTotalBytes: UInt64
    public var onACPower: Bool?
    public var batteryPercent: Int?
    public var lowPowerMode: Bool
    public var cacheVolumeFreeBytes: Int64?

    public static func gib(_ bytes: UInt64) -> String { String(format: "%.1f GiB", Double(bytes) / 1_073_741_824) }
}

/// Reads system memory, swap and power state. Read-only, cheap, and never touches a model or the GPU.
public enum MemoryProbe {
    public static func snapshot(volumeForFreeSpace: URL? = nil) -> MemorySnapshot {
        var phys: UInt64 = 0
        var size = MemoryLayout<UInt64>.size
        sysctlbyname("hw.memsize", &phys, &size, nil, 0)

        var swap = xsw_usage()
        var swapSize = MemoryLayout<xsw_usage>.size
        sysctlbyname("vm.swapusage", &swap, &swapSize, nil, 0)

        var available: UInt64 = 0
        var stats = vm_statistics64()
        var count = mach_msg_type_number_t(MemoryLayout<vm_statistics64>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &stats) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics64(mach_host_self(), HOST_VM_INFO64, $0, &count)
            }
        }
        if result == KERN_SUCCESS {
            let page = UInt64(vm_kernel_page_size)
            available = (UInt64(stats.free_count) + UInt64(stats.inactive_count) + UInt64(stats.purgeable_count)) * page
        }

        let power = powerState()
        var free: Int64?
        if let url = volumeForFreeSpace ?? FileManager.default.urls(for: .userDirectory, in: .localDomainMask).first {
            free = (try? url.resourceValues(forKeys: [.volumeAvailableCapacityForImportantUsageKey]))?
                .volumeAvailableCapacityForImportantUsage
        }
        return MemorySnapshot(physicalBytes: phys, availableBytes: available,
                              swapUsedBytes: swap.xsu_used, swapTotalBytes: swap.xsu_total,
                              onACPower: power.ac, batteryPercent: power.percent,
                              lowPowerMode: ProcessInfo.processInfo.isLowPowerModeEnabled,
                              cacheVolumeFreeBytes: free)
    }

    static func powerState() -> (ac: Bool?, percent: Int?) {
        guard let info = IOPSCopyPowerSourcesInfo()?.takeRetainedValue(),
              let list = IOPSCopyPowerSourcesList(info)?.takeRetainedValue() as? [CFTypeRef] else { return (nil, nil) }
        var ac: Bool?
        var percent: Int?
        for source in list {
            guard let d = IOPSGetPowerSourceDescription(info, source)?.takeUnretainedValue() as? [String: Any] else { continue }
            if let state = d[kIOPSPowerSourceStateKey] as? String { ac = (state == kIOPSACPowerValue) }
            if let cap = d[kIOPSCurrentCapacityKey] as? Int { percent = cap }
        }
        return (ac, percent)
    }

    /// Battery use is allowed; a charge below `minPercent` (default 25) refuses the start. AC power always passes.
    /// Mirrors the engine's own gate (`models/flashnext/power_gate.py`), which stays the authority. 0 disables the floor.
    public static let defaultMinBatteryPercent = 25
    public static func powerRefusal(_ s: MemorySnapshot, minPercent: Int = defaultMinBatteryPercent) -> String? {
        if s.onACPower == true || minPercent <= 0 { return nil }
        if s.onACPower == nil { return "The power source could not be read, so the engine will not start. Plug in and try again." }
        guard let percent = s.batteryPercent else { return "On battery and the charge level could not be read, so the engine will not start. Plug in and try again." }
        return percent < minPercent ? "Battery is \(percent)%, below the \(minPercent)% minimum. Plug in or charge before starting the engine." : nil
    }

    /// Cheap advice shown before Start. The engine's own admission guard remains the authority and is never bypassed.
    public static func advisories(_ s: MemorySnapshot, checkpointBytes: Int64?) -> [String] {
        var out: [String] = []
        if s.onACPower == false { out.append("The Mac is on battery (\(s.batteryPercent.map { "\($0)%" } ?? "charge unknown")). The engine runs on battery above \(defaultMinBatteryPercent)% and stops below it; plug in for long sessions. Results measured on battery are preliminary.") }
        if s.lowPowerMode { out.append("Low Power Mode is on. Turn it off before a long session.") }
        if s.availableBytes < 6 * 1_073_741_824 {
            out.append("Only \(MemorySnapshot.gib(s.availableBytes)) of memory looks available. Close memory-heavy apps; the engine will refuse to start if it is not enough.")
        }
        if s.swapUsedBytes > 2 * 1_073_741_824 {
            out.append("Swap is already \(MemorySnapshot.gib(s.swapUsedBytes)). The engine's swap guard may refuse requests.")
        }
        if let free = s.cacheVolumeFreeBytes, free < 20 * 1_073_741_824 {
            out.append("Less than 20 GB of disk space is free.")
        }
        return out
    }
}
