import Darwin
import FluidUse
import Foundation

/// CPU load of this process and the whole machine, plus Neural Engine figures: real ANE
/// and CPU power when a `powermetrics` log is being written (that tool needs root, so the
/// user starts it), otherwise the ANE duty cycle derived from the model calls themselves.
@MainActor
final class SystemMonitor: ObservableObject {
    @Published var processCPUPercent: Double = 0
    @Published var systemCPUPercent: Double = 0
    @Published var aneDutyPercent: Double = 0
    @Published var anePowerMilliwatts: Int?
    @Published var cpuPowerMilliwatts: Int?
    @Published var powerLogActive = false

    /// `sudo powermetrics -i 500 --samplers cpu_power,ane_power -o <this path>`
    let powerLogURL: URL
    /// asitop's convention: ANE utilization = ANE power / an assumed 8 W peak.
    let anePeakMilliwatts: Double

    /// ANE power as a share of the assumed peak, when a powermetrics log is live.
    var anePowerPercent: Double? {
        anePowerMilliwatts.map { min(100, Double($0) / anePeakMilliwatts * 100) }
    }
    private var task: Task<Void, Never>?
    private var lastRusage = rusage()
    private var lastWall = ContinuousClock.now
    private var lastTicks: (busy: UInt64, total: UInt64)?
    private var modelBusy: Duration = .zero
    /// Recent (busy, wall) pairs; the duty cycle is averaged over about two seconds.
    private var busyWindow: [(busy: Double, wall: Double)] = []

    init() {
        let path = ProcessInfo.processInfo.environment["CUA_DEMO_POWERMETRICS"] ?? "/tmp/cua-powermetrics.log"
        powerLogURL = URL(fileURLWithPath: path)
        let peak = ProcessInfo.processInfo.environment["CUA_DEMO_ANE_MAX_MW"].flatMap(Double.init) ?? 8000
        anePeakMilliwatts = peak
    }

    func start() {
        guard task == nil else { return }
        getrusage(RUSAGE_SELF, &lastRusage)
        lastWall = ContinuousClock.now
        task = Task { [weak self] in
            while !Task.isCancelled {
                try? await Task.sleep(for: .milliseconds(500))
                self?.sample()
            }
        }
    }

    func resetActivity() {
        modelBusy = .zero
        busyWindow = []
        aneDutyPercent = 0
    }

    /// Called with each model call's wall time; feeds the duty-cycle figure.
    func recordModelCall(_ latency: Duration) {
        modelBusy += latency
    }

    private func sample() {
        let now = ContinuousClock.now
        let wall = now - lastWall
        let wallSeconds = Double(wall.components.seconds) + Double(wall.components.attoseconds) / 1e18
        guard wallSeconds > 0 else { return }
        var usage = rusage()
        getrusage(RUSAGE_SELF, &usage)
        let cpuSeconds =
            Double(usage.ru_utime.tv_sec - lastRusage.ru_utime.tv_sec)
            + Double(usage.ru_utime.tv_usec - lastRusage.ru_utime.tv_usec) / 1e6
            + Double(usage.ru_stime.tv_sec - lastRusage.ru_stime.tv_sec)
            + Double(usage.ru_stime.tv_usec - lastRusage.ru_stime.tv_usec) / 1e6
        processCPUPercent = max(0, cpuSeconds / wallSeconds * 100)
        lastRusage = usage
        let busySeconds = Double(modelBusy.components.seconds) + Double(modelBusy.components.attoseconds) / 1e18
        busyWindow.append((busySeconds, wallSeconds))
        if busyWindow.count > 4 { busyWindow.removeFirst(busyWindow.count - 4) }
        let windowBusy = busyWindow.reduce(0) { $0 + $1.busy }
        let windowWall = busyWindow.reduce(0) { $0 + $1.wall }
        aneDutyPercent = windowWall > 0 ? min(100, windowBusy / windowWall * 100) : 0
        modelBusy = .zero
        lastWall = now
        sampleSystemCPU()
        samplePowerLog()
    }

    private func sampleSystemCPU() {
        var info = host_cpu_load_info()
        var count = mach_msg_type_number_t(MemoryLayout<host_cpu_load_info>.size / MemoryLayout<integer_t>.size)
        let result = withUnsafeMutablePointer(to: &info) {
            $0.withMemoryRebound(to: integer_t.self, capacity: Int(count)) {
                host_statistics(mach_host_self(), HOST_CPU_LOAD_INFO, $0, &count)
            }
        }
        guard result == KERN_SUCCESS else { return }
        let user = UInt64(info.cpu_ticks.0)
        let system = UInt64(info.cpu_ticks.1)
        let idle = UInt64(info.cpu_ticks.2)
        let nice = UInt64(info.cpu_ticks.3)
        let busy = user + system + nice
        let total = busy + idle
        if let last = lastTicks, total > last.total {
            systemCPUPercent = Double(busy - last.busy) / Double(total - last.total) * 100
        }
        lastTicks = (busy, total)
    }

    /// Reads the tail of the powermetrics log for the latest "ANE Power" and "CPU Power" lines.
    private func samplePowerLog() {
        guard let handle = try? FileHandle(forReadingFrom: powerLogURL) else {
            powerLogActive = false
            anePowerMilliwatts = nil
            cpuPowerMilliwatts = nil
            return
        }
        defer { try? handle.close() }
        let size = (try? handle.seekToEnd()) ?? 0
        let start = size > 8192 ? size - 8192 : 0
        try? handle.seek(toOffset: start)
        guard let data = try? handle.readToEnd(), let text = String(data: data, encoding: .utf8) else { return }
        func last(_ label: String) -> Int? {
            var value: Int?
            for line in text.components(separatedBy: "\n") where line.hasPrefix(label) {
                let digits = line.drop { !$0.isNumber }.prefix { $0.isNumber }
                if let number = Int(digits) { value = number }
            }
            return value
        }
        let ane = last("ANE Power:")
        let cpu = last("CPU Power:")
        powerLogActive = ane != nil || cpu != nil
        anePowerMilliwatts = ane
        cpuPowerMilliwatts = cpu
    }
}
