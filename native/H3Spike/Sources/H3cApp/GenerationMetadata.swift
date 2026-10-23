import Foundation

// What made a video, inside the video: an XMP packet in the MP4 (Adobe's
// top-level 'uuid' box, which exiftool and Adobe's tools read) with
//   - IPTC's AI fields (Photo Metadata 2025.1 / Video Metadata Hub 1.7):
//     Iptc4xmpExt:AIPromptInformation, AISystemUsed, AISystemVersionUsed;
//   - Iptc4xmpExt:DigitalSourceType trainedAlgorithmicMedia (made by a
//     trained model) and dc:description "AI-generated with MiniMax H3";
//   - xmp:CreatorTool, xmp:CreateDate;
//   - h3capp:GenerationSettings: the video's record (the .json next to it -
//     settings, seed, references, lip sync, clips) as JSON, to make it
//     again. Absolute paths in it keep only the file name.
// Written when a video is made, when clips are joined, and on export
// (where the prompt and settings can be left out).
enum GenerationMetadata {
    static let namespace = "https://junzokamahara.github.io/H3cApp/ns/1.0/"
    static let exportSettingKey = "h3c-app.exportGenerationSettings"

    /// 動画を書き出す… records the prompt and settings unless turned off.
    static var exportsSettings: Bool {
        get { UserDefaults.standard.object(forKey: exportSettingKey) as? Bool ?? true }
        set { UserDefaults.standard.set(newValue, forKey: exportSettingKey) }
    }

    /// Adobe's UUID for an XMP box in ISO base media files.
    private static let xmpUUID: [UInt8] = [0xBE, 0x7A, 0xCF, 0xCB, 0x97, 0xA9, 0x42, 0xE8,
                                           0x9C, 0x71, 0x99, 0x94, 0x91, 0xE3, 0xAF, 0xAC]

    /// The XMP packet for a video made as `record` (a joined video also
    /// passes its clips' records). Without `includeSettings`, only that it is
    /// AI-generated and by what - no prompt, no settings.
    static func xmp(for record: ProjectVideoRecord, clips: [ProjectVideoRecord] = [],
                    includeSettings: Bool) -> String {
        let variants = Set(([record] + clips).map(variant)).sorted()
        var attributes: [(String, String)] = [
            ("xmp:CreatorTool", "H3cApp \(record.appVersion)"),
            ("xmp:CreateDate", ISO8601DateFormatter().string(from: record.completedAt)),
            ("Iptc4xmpExt:DigitalSourceType", "http://cv.iptc.org/newscodes/digitalsourcetype/trainedAlgorithmicMedia"),
            ("Iptc4xmpExt:AISystemUsed", "MiniMax H3"),
            ("Iptc4xmpExt:AISystemVersionUsed", variants.joined(separator: ", ")),
        ]
        var elements = """
                <dc:description><rdf:Alt><rdf:li xml:lang="x-default">\(escape(AIDisclosure.label))</rdf:li></rdf:Alt></dc:description>

        """
        if includeSettings {
            elements += "        <Iptc4xmpExt:AIPromptInformation>\(escape(promptInformation(record)))</Iptc4xmpExt:AIPromptInformation>\n"
            elements += "        <h3capp:GenerationSettings>\(escape(settingsJSON(record, clips: clips)))</h3capp:GenerationSettings>\n"
        } else {
            attributes.removeAll { $0.0 == "xmp:CreateDate" }
        }
        let attributeText = attributes.map { "\n        \($0.0)=\"\(escape($0.1))\"" }.joined()
        return """
        <?xpacket begin="\u{FEFF}" id="W5M0MpCehiHzreSzNTczkc9d"?>
        <x:xmpmeta xmlns:x="adobe:ns:meta/">
          <rdf:RDF xmlns:rdf="http://www.w3.org/1999/02/22-rdf-syntax-ns#">
            <rdf:Description rdf:about=""
                xmlns:dc="http://purl.org/dc/elements/1.1/"
                xmlns:xmp="http://ns.adobe.com/xap/1.0/"
                xmlns:Iptc4xmpExt="http://iptc.org/std/Iptc4xmpExt/2008-02-29/"
                xmlns:h3capp="\(namespace)"\(attributeText)>
        \(elements)    </rdf:Description>
          </rdf:RDF>
        </x:xmpmeta>
        <?xpacket end="w"?>
        """
    }

    /// Puts `xmp` in the MP4 at `url`: a 'uuid' box at the end of the file,
    /// replacing one this wrote before. The rest of the file is untouched.
    static func write(_ xmp: String, into url: URL) throws {
        let handle = try FileHandle(forUpdating: url)
        defer { try? handle.close() }
        let end = try handle.seekToEnd()
        // Walk the top-level boxes; drop the last one if it is an XMP box.
        var offset: UInt64 = 0
        var lastXMP: UInt64?
        while offset + 8 <= end {
            try handle.seek(toOffset: offset)
            guard let header = try handle.read(upToCount: 8), header.count == 8 else { break }
            var size = header[0 ..< 4].reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
            let type = String(decoding: header[4 ..< 8], as: UTF8.self)
            var headerLength: UInt64 = 8
            if size == 1 {
                guard let large = try handle.read(upToCount: 8), large.count == 8 else { break }
                size = large.reduce(UInt64(0)) { $0 << 8 | UInt64($1) }
                headerLength = 16
            } else if size == 0 {
                size = end - offset
            }
            guard size >= headerLength, offset + size <= end else { break }
            lastXMP = nil
            if type == "uuid", offset + headerLength + 16 <= end {
                try handle.seek(toOffset: offset + headerLength)
                if let uuid = try handle.read(upToCount: 16), Array(uuid) == xmpUUID, offset + size == end {
                    lastXMP = offset
                }
            }
            offset += size
        }
        if let lastXMP { try handle.truncate(atOffset: lastXMP) }
        let payload = Data(xmp.utf8)
        var box = Data()
        let size = UInt32(8 + 16 + payload.count)
        box.append(contentsOf: withUnsafeBytes(of: size.bigEndian, Array.init))
        box.append(contentsOf: Array("uuid".utf8))
        box.append(contentsOf: xmpUUID)
        box.append(payload)
        try handle.seekToEnd()
        try handle.write(contentsOf: box)
    }

    /// Writes the XMP for `record` into `url`, logging rather than failing:
    /// the video is fine without it.
    static func tag(_ url: URL, record: ProjectVideoRecord, clips: [ProjectVideoRecord] = [],
                    includeSettings: Bool = true) {
        do {
            try write(xmp(for: record, clips: clips, includeSettings: includeSettings), into: url)
        } catch {
            NSLog("H3cApp: cannot write XMP into %@: %@", url.lastPathComponent, error.localizedDescription)
        }
    }

    /// FL2VA, or Ref2VA when references shaped it.
    private static func variant(_ record: ProjectVideoRecord) -> String {
        let draft = record.draft
        return draft.creationMethod == CreationMethod.image.rawValue
            && draft.imageInputMode == ImageInputMode.referenceImage.rawValue && !draft.references.isEmpty
            ? "Ref2VA" : "FL2VA"
    }

    /// The prompt as the model got it; for a joined video, the common prompt
    /// and what each clip added.
    static func promptInformation(_ record: ProjectVideoRecord) -> String {
        let draft = record.draft
        var lines = [draft.prompt]
        let clipTexts = draft.customClips?.map(\.prompt) ?? draft.clipPrompts ?? []
        for (index, text) in clipTexts.enumerated() {
            let trimmed = text.trimmingCharacters(in: .whitespacesAndNewlines)
            if !trimmed.isEmpty { lines.append("Clip \(index + 1): \(trimmed)") }
        }
        return lines.joined(separator: "\n\n")
    }

    /// The record (and a joined video's clips') as compact JSON, absolute
    /// paths cut to their file names.
    static func settingsJSON(_ record: ProjectVideoRecord, clips: [ProjectVideoRecord]) -> String {
        func object(_ value: ProjectVideoRecord) -> Any? {
            (try? ProjectFiles.encoder.encode(value)).flatMap { try? JSONSerialization.jsonObject(with: $0) }
        }
        func scrub(_ value: Any) -> Any {
            switch value {
            case let text as String where text.hasPrefix("/"):
                return URL(fileURLWithPath: text).lastPathComponent
            case let list as [Any]:
                return list.map(scrub)
            case let dictionary as [String: Any]:
                return dictionary.mapValues(scrub)
            default:
                return value
            }
        }
        var root: [String: Any] = ["schema": "h3capp.generation/1", "model": "MiniMax H3"]
        if let main = object(record) { root["record"] = scrub(main) }
        if !clips.isEmpty { root["clips"] = clips.compactMap(object).map(scrub) }
        guard let data = try? JSONSerialization.data(withJSONObject: root, options: [.sortedKeys, .withoutEscapingSlashes]) else {
            return "{}"
        }
        return String(decoding: data, as: UTF8.self)
    }

    private static func escape(_ text: String) -> String {
        text.replacingOccurrences(of: "&", with: "&amp;")
            .replacingOccurrences(of: "<", with: "&lt;")
            .replacingOccurrences(of: ">", with: "&gt;")
            .replacingOccurrences(of: "\"", with: "&quot;")
    }
}
