import Foundation
import Security

/// The local API's credential. 127.0.0.1 is reachable by every account and
/// process on this Mac, not only this user's, so every request must carry a
/// secret only this user can read: 32 random bytes as hex, new each time
/// the server starts (unless other devices may use it, see APIHost), kept
/// in a file of mode 0600 (`api-token` in the app's Application Support
/// folder) and sent as `Authorization: Bearer <token>`. See API.md.
struct APIToken: Sendable {
    let value: String

    /// A fresh token; nil only if the system's random source fails.
    static func generate() -> APIToken? {
        var bytes = [UInt8](repeating: 0, count: 32)
        guard SecRandomCopyBytes(kSecRandomDefault, bytes.count, &bytes) == errSecSuccess else { return nil }
        return APIToken(value: bytes.map { String(format: "%02x", $0) }.joined())
    }

    /// The token a previous launch left in `path`, if the file is still
    /// as this app writes it: a regular file (not a link) of this user's
    /// that nobody else can read or write, holding 64 hex digits.
    static func load(from path: String) -> APIToken? {
        var info = stat()
        guard lstat(path, &info) == 0, info.st_mode & S_IFMT == S_IFREG,
              info.st_uid == getuid(), info.st_mode & 0o077 == 0,
              let data = FileManager.default.contents(atPath: path), data.count == 64 else { return nil }
        let value = String(decoding: data, as: UTF8.self)
        guard value.allSatisfy({ $0.isASCII && $0.isHexDigit && !$0.isUppercase }) else { return nil }
        return APIToken(value: value)
    }

    /// Whether `request` carries this token as `Authorization: Bearer ...`,
    /// compared in constant time.
    func authorizes(_ request: HTTPRequest) -> Bool {
        guard let header = request.headers["authorization"] else { return false }
        let parts = header.split(separator: " ", maxSplits: 1)
        guard parts.count == 2, parts[0].lowercased() == "bearer" else { return false }
        let sent = Array(parts[1].trimmingCharacters(in: .whitespaces).utf8)
        let expected = Array(value.utf8)
        guard sent.count == expected.count else { return false }
        return timingsafe_bcmp(sent, expected, expected.count) == 0
    }

    /// The answer to a request without the right token (never the token).
    static func unauthorizedResponse() -> HTTPResponse {
        var response = HTTPResponse.error(401, "missing or wrong API token: send \"Authorization: Bearer <token>\" "
            + "with the token from H3cApp's General Settings or ~/Library/Application Support/H3cApp/api-token")
        response.headers["WWW-Authenticate"] = "Bearer"
        return response
    }

    /// Writes the token to `path`, readable by this user only. The file is
    /// created under a new name with mode 0600 before the token goes in, so
    /// it is never readable by anyone else even for a moment, then renamed
    /// over `path` - which replaces whatever is there without following it
    /// when it is a symbolic link (and fails when it is a folder). The
    /// folder, if it has to be created, is owner-only too.
    func save(to path: String) throws {
        let fm = FileManager.default
        let directory = (path as NSString).deletingLastPathComponent
        if !fm.fileExists(atPath: directory) {
            try fm.createDirectory(atPath: directory, withIntermediateDirectories: true,
                                   attributes: [.posixPermissions: 0o700])
        }
        let temporary = directory + "/.api-token-" + UUID().uuidString
        let fd = open(temporary, O_WRONLY | O_CREAT | O_EXCL | O_NOFOLLOW | O_CLOEXEC, 0o600)
        guard fd >= 0 else { throw APITokenError.failed("cannot create \(temporary): \(String(cString: strerror(errno)))") }
        let bytes = Array(value.utf8)
        var failure: String?
        if fchmod(fd, 0o600) != 0 || bytes.withUnsafeBytes({ Darwin.write(fd, $0.baseAddress, $0.count) }) != bytes.count {
            failure = String(cString: strerror(errno))
        }
        if close(fd) != 0, failure == nil { failure = String(cString: strerror(errno)) }
        if failure == nil, rename(temporary, path) != 0 { failure = String(cString: strerror(errno)) }
        if let failure {
            unlink(temporary)
            throw APITokenError.failed("cannot write \(path): \(failure)")
        }
    }

    /// Deletes the file at `path` if it is a regular file still holding this
    /// token (never one another copy of the app wrote since).
    func remove(from path: String) {
        let fm = FileManager.default
        guard (try? fm.attributesOfItem(atPath: path)[.type] as? FileAttributeType) == .typeRegular,
              let data = fm.contents(atPath: path), String(decoding: data, as: UTF8.self) == value else { return }
        unlink(path)
    }
}

enum APITokenError: Error, CustomStringConvertible {
    case failed(String)
    var description: String {
        switch self {
        case .failed(let message): return message
        }
    }
}
