import Darwin
import Foundation
import SystemConfiguration

/// The app's one API server, started at launch and kept until quit. It
/// always talks to the primary model (AppModels), which outlives its window,
/// so the API keeps working with every window closed while the app runs.
///
/// 全体設定 shows its token and addresses and can let other devices on the
/// network in (allowsExternalAccess): the server then listens on every
/// interface instead of 127.0.0.1 only, still answering nothing without the
/// token. While that is on the token is kept across launches - a phone or
/// another Mac cannot read the token file - until 作り直す replaces it;
/// otherwise it is new at every launch, as before.
@MainActor
final class APIHost: ObservableObject {
    static let shared = APIHost()

    @Published private(set) var token: APIToken?
    @Published private(set) var startError: String?
    @Published private(set) var isListening = false
    @Published private(set) var allowsExternalAccess =
        UserDefaults.standard.bool(forKey: APIHost.externalAccessKey)

    private static let externalAccessKey = "H3cApp.apiAllowsExternalAccess"
    private var server: HTTPServer?
    /// What the server's router checks requests against; replacing the
    /// token swaps it here without a restart.
    private let currentToken = APITokenBox()

    func start() {
        guard server == nil else { return }
        startError = nil
        // A token another device may hold is reused while external access
        // is on; anything else gets a fresh one.
        let reused = allowsExternalAccess ? (token ?? APIToken.load(from: h3APITokenPath)) : token
        guard let token = reused ?? APIToken.generate() else {
            startError = "no random bytes for the API token"
            return
        }
        currentToken.set(token)
        let box = currentToken
        let server = HTTPServer(port: h3APIPort, acceptsExternal: allowsExternalAccess,
                                ownHostNames: allowsExternalAccess ? Self.ownHostNames() : []) { request in
            guard box.authorizes(request) else { return APIToken.unauthorizedResponse() }
            return await AppModels.shared.primary.handleAPIRequest(request)
        }
        do {
            try server.start()
        } catch {
            startError = String(describing: error)
            return
        }
        // Only once the port is ours, so a second copy of the app that
        // cannot bind it never replaces the running one's token.
        do {
            try token.save(to: h3APITokenPath)
        } catch {
            server.stop()
            startError = String(describing: error)
            return
        }
        self.server = server
        self.token = token
        isListening = true
    }

    /// At quit. The token file goes unless other devices may still use the
    /// token - and goes first: once the port is free another copy of the
    /// app may take it and write its own token, which must stay.
    func stop() {
        if !allowsExternalAccess { token?.remove(from: h3APITokenPath) }
        server?.stop()
        server = nil
        isListening = false
    }

    /// Lets other devices in, or shuts them out again, by listening anew.
    /// Turning it off also replaces the token, so one handed to another
    /// device stops working.
    func setAllowsExternalAccess(_ allowed: Bool) {
        guard allowed != allowsExternalAccess else { return }
        UserDefaults.standard.set(allowed, forKey: Self.externalAccessKey)
        server?.stop()
        server = nil
        isListening = false
        allowsExternalAccess = allowed
        if !allowed { token = nil }
        start()
        AppModels.shared.refreshAPIStatus()
    }

    /// 作り直す: a new token from now on; the old one stops working at once.
    func regenerateToken() {
        guard let fresh = APIToken.generate() else { return }
        do {
            try fresh.save(to: h3APITokenPath)
        } catch {
            startError = String(describing: error)
            return
        }
        currentToken.set(fresh)
        token = fresh
    }

    /// The addresses scripts can use: this Mac's own first, then - with
    /// external access on - its network name and each IPv4 address other
    /// devices may reach it at.
    var addresses: [String] {
        var hosts = ["127.0.0.1"]
        if allowsExternalAccess { hosts += Self.ownHostNames().sorted() + Self.networkAddresses() }
        return hosts.map { "http://\($0):\(h3APIPort)" }
    }

    /// What 「APIサーバー」 says in `model`'s window.
    func status(for model: GenerationViewModel) -> String {
        if let startError { return String(localized: "起動できませんでした（\(startError)）") }
        guard isListening else { return String(localized: "起動しています…") }
        var address = "http://127.0.0.1:\(h3APIPort)"
        if allowsExternalAccess { address = String(localized: "\(address)（ほかの機器からも受け付け中）") }
        return model === AppModels.shared.primary
            ? address : String(localized: "\(address)（別のウィンドウで受け付け中）")
    }

    /// The names other devices find this Mac by: its Bonjour name
    /// ("Name.local", as in System Settings > General > Sharing).
    static func ownHostNames() -> Set<String> {
        guard let name = SCDynamicStoreCopyLocalHostName(nil) as String?, !name.isEmpty else { return [] }
        return [name.lowercased() + ".local"]
    }

    /// The IPv4 addresses of this Mac's active network interfaces, loopback
    /// aside.
    static func networkAddresses() -> [String] {
        var list: UnsafeMutablePointer<ifaddrs>?
        guard getifaddrs(&list) == 0, let first = list else { return [] }
        defer { freeifaddrs(list) }
        var result: [String] = []
        for entry in sequence(first: first, next: { $0.pointee.ifa_next }) {
            let flags = Int32(entry.pointee.ifa_flags)
            guard let address = entry.pointee.ifa_addr, address.pointee.sa_family == sa_family_t(AF_INET),
                  flags & IFF_UP != 0, flags & IFF_LOOPBACK == 0 else { continue }
            var text = [CChar](repeating: 0, count: Int(INET_ADDRSTRLEN))
            address.withMemoryRebound(to: sockaddr_in.self, capacity: 1) { pointer in
                var inAddress = pointer.pointee.sin_addr
                _ = inet_ntop(AF_INET, &inAddress, &text, socklen_t(text.count))
            }
            let string = String(cString: text)
            if !string.isEmpty, !result.contains(string) { result.append(string) }
        }
        return result
    }
}

/// The token the running server accepts, read on its connection threads
/// and replaced from the main actor.
final class APITokenBox: @unchecked Sendable {
    private let lock = NSLock()
    private var token: APIToken?

    func set(_ token: APIToken) {
        lock.lock()
        self.token = token
        lock.unlock()
    }

    func authorizes(_ request: HTTPRequest) -> Bool {
        lock.lock()
        let token = self.token
        lock.unlock()
        return token?.authorizes(request) ?? false
    }
}
