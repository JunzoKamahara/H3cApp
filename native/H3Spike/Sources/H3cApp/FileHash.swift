import CryptoKit
import Foundation

/// SHA-256 of a file, read in chunks so a 20 GB weight file never sits in
/// memory. Used to check downloaded SwiftVR weights against the pinned
/// revision's object ids (SwiftVRModel) and to recognise LoRA files whose
/// origin is recorded (ModelLibrary.knownLoRAProvenance).
enum FileHash {
    /// Lower-case hex digest. `onBytes` is told how many bytes were read
    /// after each chunk (for progress); the read stops on task cancellation.
    static func sha256(ofFile path: String, onBytes: ((Int64) -> Void)? = nil) throws -> String {
        let handle = try FileHandle(forReadingFrom: URL(fileURLWithPath: path))
        defer { try? handle.close() }
        var hasher = SHA256()
        while true {
            try Task.checkCancellation()
            let done: Bool = try autoreleasepool {
                guard let chunk = try handle.read(upToCount: 8 << 20), !chunk.isEmpty else { return true }
                hasher.update(data: chunk)
                onBytes?(Int64(chunk.count))
                return false
            }
            if done { break }
        }
        return hasher.finalize().map { String(format: "%02x", $0) }.joined()
    }

    static func sha256(of data: Data) -> String {
        SHA256.hash(data: data).map { String(format: "%02x", $0) }.joined()
    }
}
