import XCTest
@testable import H3cApp

/// The text encoder MiniMax H3 Image shares with a video model
/// (SharedTextEncoder): what the app's API cannot reach - the answer to the
/// question about a different encoder, and a switch cut short in states no
/// switch leaves by itself. Nothing is generated; the encoders are a few
/// kilobytes.
///
/// The tests make the app's models, which read and write its Application
/// Support folder: they are skipped unless CFFIXED_USER_HOME names a folder
/// for them (tools/image_shared_text_encoder/README.md).
@MainActor
final class SharedTextEncoderTests: XCTestCase {
    private var root = ""
    private var acquired: [GenerationViewModel] = []

    override func setUp() async throws {
        try XCTSkipIf(ProcessInfo.processInfo.environment["CFFIXED_USER_HOME"] == nil,
                      "set CFFIXED_USER_HOME to an empty folder: the tests use the app's Application Support folder")
        root = NSTemporaryDirectory() + "h3-shared-text-encoder-" + UUID().uuidString
        try FileManager.default.createDirectory(atPath: root, withIntermediateDirectories: true)
    }

    override func tearDown() async throws {
        // Nothing to undo after a skipped test - and nothing of the real
        // app's may be touched: only what setUp and the tests made, under
        // the folder CFFIXED_USER_HOME names.
        guard !root.isEmpty, let home = ProcessInfo.processInfo.environment["CFFIXED_USER_HOME"] else { return }
        for model in acquired {
            model.imageGenerationEnded()
            AppModels.shared.release(model)
        }
        acquired = []
        try? FileManager.default.removeItem(atPath: root)
        let image = (H3ImageModel.h3.directory as NSString).resolvingSymlinksInPath
        if image.hasPrefix((home as NSString).resolvingSymlinksInPath + "/") {
            try? FileManager.default.removeItem(atPath: image)
        }
        // The test runner's own preferences, not the app's.
        for key in UserDefaults.standard.dictionaryRepresentation().keys
        where key.hasPrefix("H3ModelLibrary.") || key.hasPrefix("h3c-app.") || key.hasPrefix("H3cApp.") {
            UserDefaults.standard.removeObject(forKey: key)
        }
    }

    /// A stand-in video model holding these files as its text encoder.
    private func videoModel(_ name: String, _ files: [String: Data]) throws -> String {
        let path = root + "/" + name
        let folder = SharedTextEncoder.folder(inVideoModel: path)
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        for (file, data) in files { try data.write(to: URL(fileURLWithPath: folder + "/" + file)) }
        return path
    }

    private func write(_ files: [String: Data], to folder: String) throws {
        try FileManager.default.createDirectory(atPath: folder, withIntermediateDirectories: true)
        for (file, data) in files { try data.write(to: URL(fileURLWithPath: folder + "/" + file)) }
    }

    private func names(in folder: String) -> [String] {
        ((try? FileManager.default.contentsOfDirectory(atPath: folder)) ?? []).sorted()
    }

    /// A model as another window's, beside the primary one.
    private func anotherWindow() -> GenerationViewModel {
        let first = AppModels.shared.acquire().model
        let second = AppModels.shared.acquire().model
        acquired += [first, second]
        return second === AppModels.shared.primary ? first : second
    }

    private func data(_ byte: UInt8, _ count: Int = 4096) -> Data { Data(repeating: byte, count: count) }

    // MARK: Generating in another window

    func testNoSwitchWhileAnotherWindowGenerates() {
        let setup = H3ImageModelSetup.of(.h3)
        let window = anotherWindow()
        XCTAssertFalse(AppModels.shared.isGeneratingImageAnywhere)
        window.imageGenerationBegan(index: 1, total: 1)
        XCTAssertTrue(AppModels.shared.isGeneratingImageAnywhere)
        XCTAssertEqual(setup.shareTextEncoder(videoModelPaths: []),
                       SharedTextEncoder.SwitchError.generating.localizedDescription)
        XCTAssertNotNil(setup.chooseTextEncoder(videoModel: root + "/None"))
        XCTAssertEqual(setup.phase, .idle)
        // Its window closes: the generation counts until it has stopped.
        AppModels.shared.release(window)
        XCTAssertTrue(AppModels.shared.isGeneratingImageAnywhere)
        XCTAssertEqual(setup.shareTextEncoder(videoModelPaths: []),
                       SharedTextEncoder.SwitchError.generating.localizedDescription)
        window.imageGenerationEnded()
        XCTAssertFalse(AppModels.shared.isGeneratingImageAnywhere)
    }

    /// The question about a different encoder is up, a generation begins in
    /// another window, and the answer is yes: refused, the choice kept.
    func testConfirmingAChoiceWhileAnotherWindowGenerates() async throws {
        let image = H3ImageModel.h3.directory
        try FileManager.default.createDirectory(atPath: image, withIntermediateDirectories: true)
        let chosen = try videoModel("Chosen", ["model.gguf": data(1)])
        let other = try videoModel("Other", ["model.gguf": data(2)])
        try SharedTextEncoder.choose(
            videoModel: chosen, files: SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: chosen)),
            forImageModel: image)

        let setup = H3ImageModelSetup.of(.h3)
        XCTAssertNil(setup.chooseTextEncoder(videoModel: other))
        for _ in 0 ..< 100 where setup.pendingChoice == nil { try await Task.sleep(nanoseconds: 50_000_000) }
        XCTAssertEqual(setup.pendingChoice?.videoModel, other, "a different encoder waits for the user's word")
        XCTAssertEqual(SharedTextEncoder.record(forImageModel: image)?.videoModel, chosen)

        let window = anotherWindow()
        window.imageGenerationBegan(index: 1, total: 1)
        XCTAssertNotNil(setup.confirmChoice(true), "refused while an image is generated")
        XCTAssertEqual(setup.pendingChoice?.videoModel, other, "the choice is kept waiting")
        XCTAssertEqual(SharedTextEncoder.record(forImageModel: image)?.videoModel, chosen, "and nothing is recorded")

        window.imageGenerationEnded()
        XCTAssertNil(setup.confirmChoice(true))
        XCTAssertNil(setup.pendingChoice)
        XCTAssertEqual(SharedTextEncoder.record(forImageModel: image)?.videoModel, other)
    }

    // MARK: While the text encoder is compared or chosen

    /// A link to the install is the install, 高解像度化 by redrawing waits as
    /// generating does, and a cancelled reading keeps the setup busy until
    /// its task has stopped.
    func testNothingBeginsUntilAReadingHasStopped() async throws {
        let image = H3ImageModel.h3.directory
        try FileManager.default.createDirectory(atPath: image, withIntermediateDirectories: true)
        let link = root + "/Link"
        try FileManager.default.createSymbolicLink(atPath: link, withDestinationPath: image)
        XCTAssertEqual(H3ImageModel.installed(at: link), .h3, "the folder under another path")
        let chosen = try videoModel("Chosen", ["model.gguf": data(1)])
        func problem(_ path: String) -> String? {
            GenerationViewModel.imageTextEncoderProblem(model: path, name: "Linked", videoModelPaths: [chosen])
        }
        ModelLibrary.registerImageModel(path: link, name: "Linked")
        let window = anotherWindow()
        XCTAssertEqual(window.library.activeImageModel?.path, link)
        XCTAssertNil(problem(link))

        let setup = H3ImageModelSetup.of(.h3)
        XCTAssertNil(setup.chooseTextEncoder(videoModel: chosen))
        XCTAssertNotNil(problem(image))
        XCTAssertNotNil(problem(link), "also for the install reached through a link")
        XCTAssertEqual(window.imageUpscaleBlockedReason(.redraw), problem(link), "redrawing waits as generating does")

        setup.cancel()
        XCTAssertTrue(setup.isBusy, "busy until the task has stopped")
        XCTAssertNotNil(problem(link))
        XCTAssertNotNil(setup.chooseTextEncoder(videoModel: chosen), "nothing else begins meanwhile")
        XCTAssertNotNil(setup.shareTextEncoder(videoModelPaths: [chosen]))
        for _ in 0 ..< 100 where setup.isBusy { try await Task.sleep(nanoseconds: 50_000_000) }
        XCTAssertEqual(setup.phase, .idle)
        XCTAssertNil(SharedTextEncoder.record(forImageModel: image), "a cancelled choice records nothing")
        XCTAssertNil(problem(link))
    }

    // MARK: Giving up a copy

    /// Only the files that were compared may go with the folder: a copy's
    /// folder holding anything else is neither compared nor deleted.
    func testACopyHoldingAnythingElseIsNotGivenUp() throws {
        let files = ["model.gguf": data(1)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        let own = SharedTextEncoder.ownFolder(ofImageModel: image)
        try write(files, to: own)
        func refusal(_ body: () throws -> Void) -> String? {
            do { try body() } catch SharedTextEncoder.SwitchError.extra(let name) { return "extra " + name
            } catch SharedTextEncoder.SwitchError.differs(let name) { return "differs " + name
            } catch SharedTextEncoder.SwitchError.changed(let path) { return "changed " + (path as NSString).lastPathComponent
            } catch SharedTextEncoder.SwitchError.engine { return "engine"
            } catch { return "other \(error)" }
            return nil
        }
        var compared: SharedTextEncoder.Compared?
        XCTAssertNil(refusal { compared = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in } })
        let marks = try XCTUnwrap(compared)
        XCTAssertNotNil(marks.files["model.gguf"]?.sha256)

        // A folder of the user's, a hidden file, a file not compared.
        for (extra, make) in [
            ("backup", { try self.write(["notes.txt": self.data(7, 10)], to: own + "/backup") }),
            (".notes", { try self.data(7, 10).write(to: URL(fileURLWithPath: own + "/.notes")) }),
            ("other.bin", { try self.data(7, 10).write(to: URL(fileURLWithPath: own + "/other.bin")) }),
        ] as [(String, () throws -> Void)] {
            try make()
            XCTAssertEqual(refusal { _ = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in } },
                           extra == "other.bin" ? "other nothingToGiveUp" : "extra " + extra)
            XCTAssertEqual(refusal { try SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: marks) },
                           "extra " + extra)
            XCTAssertTrue(FileManager.default.fileExists(atPath: own + "/" + extra), "\(extra) is still there")
            XCTAssertTrue(FileManager.default.fileExists(atPath: own + "/model.gguf"))
            XCTAssertFalse(FileManager.default.fileExists(atPath: own + ".removing"))
            XCTAssertNil(SharedTextEncoder.record(forImageModel: image))
            try FileManager.default.removeItem(atPath: own + "/" + extra)
        }
        if FileManager.default.fileExists(atPath: own + "/backup/notes.txt") { XCTFail("the loop removes it") }

        // The Finder's .DS_Store is not the user's: past these checks (and
        // stopped at the next one, the folder not being an image model).
        try data(0, 10).write(to: URL(fileURLWithPath: own + "/.DS_Store"))
        XCTAssertEqual(refusal { try SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: marks) }, "engine")
        XCTAssertEqual(names(in: own), [".DS_Store", "model.gguf"])
    }

    /// The copy is deleted for what was compared, not for what is there
    /// later: a file written to or replaced since stops the switch, also
    /// when its size is the same - and so does a change of the video
    /// model's file, which is what would be read from then on.
    func testACopyChangedSinceItWasComparedIsNotGivenUp() throws {
        let files = ["model.gguf": data(1)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        let own = SharedTextEncoder.ownFolder(ofImageModel: image)
        let copy = URL(fileURLWithPath: own + "/model.gguf")
        let encoder = URL(fileURLWithPath: SharedTextEncoder.folder(inVideoModel: source) + "/model.gguf")
        try write(files, to: own)
        func date(_ url: URL) throws -> Date {
            try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.modificationDate] as? Date)
        }
        func number(_ url: URL) throws -> UInt64 {
            try XCTUnwrap(FileManager.default.attributesOfItem(atPath: url.path)[.systemFileNumber] as? NSNumber).uint64Value
        }
        func switching(_ compared: SharedTextEncoder.Compared) -> String {
            do { try SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: compared)
            } catch SharedTextEncoder.SwitchError.changed(let path) { return "changed " + path
            } catch SharedTextEncoder.SwitchError.engine { return "engine"
            } catch { return "other \(error)" }
            return "switched"
        }
        func unchanged() {
            XCTAssertEqual(names(in: own), ["model.gguf"])
            XCTAssertFalse(FileManager.default.fileExists(atPath: own + ".removing"))
            XCTAssertNil(SharedTextEncoder.record(forImageModel: image))
        }
        // A date of whole seconds, which can be given to another file exactly.
        let whole = Date(timeIntervalSinceReferenceDate: (Date().timeIntervalSinceReferenceDate - 100).rounded(.down))
        try FileManager.default.setAttributes([.modificationDate: whole], ofItemAtPath: copy.path)
        var compared = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in }
        // Nothing changed: past the checks (stopped at the next one, the
        // folder not being an image model).
        XCTAssertEqual(switching(compared), "engine")

        // Written to in place: the same size and file, a later date.
        let read = try date(copy), file = try number(copy)
        XCTAssertEqual(read, whole)
        let handle = try FileHandle(forWritingTo: copy)
        try handle.write(contentsOf: data(9))
        try handle.close()
        try FileManager.default.setAttributes([.modificationDate: read.addingTimeInterval(1)], ofItemAtPath: copy.path)
        XCTAssertEqual(try number(copy), file)
        XCTAssertEqual(switching(compared), "changed " + copy.path)
        unchanged()

        // Replaced by another file of the same size, given the date the
        // one compared had.
        try data(9).write(to: copy, options: .atomic)
        try FileManager.default.setAttributes([.modificationDate: read], ofItemAtPath: copy.path)
        XCTAssertEqual(try date(copy), read)
        XCTAssertNotEqual(try number(copy), file)
        XCTAssertEqual(switching(compared), "changed " + copy.path)
        unchanged()
        XCTAssertEqual(try Data(contentsOf: copy), data(9))

        // The video model's file changed after the two were found the same.
        try data(1).write(to: copy)
        compared = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in }
        XCTAssertEqual(switching(compared), "engine")
        let was = try date(encoder)
        try data(9).write(to: encoder)
        try FileManager.default.setAttributes([.modificationDate: was.addingTimeInterval(1)], ofItemAtPath: encoder.path)
        XCTAssertEqual(switching(compared), "changed " + SharedTextEncoder.folder(inVideoModel: source))
        unchanged()
        XCTAssertEqual(try Data(contentsOf: copy), data(1))
    }

    // MARK: Deleting a video model

    /// A video model listed or recorded under a link is that video model:
    /// deleting its files counts the image models that read it through the
    /// link, and the link is not what they would read next.
    func testDeletingAVideoModelCountsWhatReadsItThroughALink() throws {
        let files = ["model.gguf": data(1)]
        let video = try videoModel("Real", files)
        let alias = root + "/Alias"
        try FileManager.default.createSymbolicLink(atPath: alias, withDestinationPath: video)
        let image = root + "/Image"
        try FileManager.default.createDirectory(atPath: image, withIntermediateDirectories: true)
        try SharedTextEncoder.choose(
            videoModel: alias, files: SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: alias)),
            forImageModel: image)
        let images = [(name: "Image", path: image)]

        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: image, videoModelPaths: [alias, video]), .shared(videoModel: alias))
        XCTAssertEqual(SharedTextEncoder.candidates(in: [alias, video]), [alias], "one folder, listed twice")
        for deleted in [video, alias] {
            let found = SharedTextEncoder.dependents(onVideoModel: deleted, imageModels: images, videoModelPaths: [alias, video])
            XCTAssertEqual(found.map(\.directory), [image], "deleting \(deleted)")
            XCTAssertNil(found.first?.next, "the other path of the same folder is no replacement")
            XCTAssertEqual(found.first?.needsChoice, false)
        }

        // Another folder with the same encoder is one.
        let twin = try videoModel("Twin", files)
        _ = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: twin))
        let found = SharedTextEncoder.dependents(onVideoModel: video, imageModels: images, videoModelPaths: [alias, video, twin])
        XCTAssertEqual(found.first?.next, twin)
        XCTAssertEqual(found.first?.sameEncoder, true)
    }

    // MARK: A switch cut short

    /// The record of a switch whose copy held `files`, the video model's
    /// encoder being those same files; `source` is that video model.
    private func switchRecord(image: String, source: String, files: [String: Data]) throws {
        try FileManager.default.createDirectory(atPath: image, withIntermediateDirectories: true)
        let marks = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: source))
        try SharedTextEncoder.write(
            SharedTextEncoder.Record(videoModel: source, files: marks, removing: files.mapValues { Int64($0.count) }),
            forImageModel: image)
    }

    /// text_encoder/ holds a file - perhaps part of an encoder - and the
    /// video model has gone: what is left of the copy is not deleted for it.
    func testALeftoverIsKeptWhenTheFoldersOwnFilesMayBePartial() throws {
        let files = ["a.gguf": data(1), "b.gguf": data(2, 8192)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        try switchRecord(image: image, source: source, files: files)
        try FileManager.default.removeItem(atPath: source)
        let own = SharedTextEncoder.ownFolder(ofImageModel: image)
        let leaving = own + ".removing"

        // Part of the copy deleted, and one file back in text_encoder/.
        try write(["a.gguf": files["a.gguf"]!], to: own)
        try write(["b.gguf": files["b.gguf"]!], to: leaving)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: []), .left(8192))
        XCTAssertEqual(names(in: leaving), ["b.gguf"])
        XCTAssertEqual(names(in: own), ["a.gguf"])

        // The whole copy left, and a file in text_encoder/: neither deleted
        // nor put back over it.
        try write(["a.gguf": files["a.gguf"]!], to: leaving)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: []), .left(4096 + 8192))
        XCTAssertEqual(names(in: leaving), ["a.gguf", "b.gguf"])
        XCTAssertEqual(names(in: own), ["a.gguf"])
        XCTAssertNotNil(SharedTextEncoder.record(forImageModel: image)?.removing)

        // With text_encoder/ empty, the whole copy is put back.
        try FileManager.default.removeItem(atPath: own + "/a.gguf")
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: []), .restored)
        XCTAssertEqual(names(in: own), ["a.gguf", "b.gguf"])
        XCTAssertFalse(FileManager.default.fileExists(atPath: leaving))
        XCTAssertNil(SharedTextEncoder.record(forImageModel: image)?.removing)
    }

    /// The copy left behind goes only for a video model whose files are
    /// known to have the recorded SHA-256, and only when it is what the
    /// record names.
    func testALeftoverGoesOnlyForAVerifiedEncoder() throws {
        let files = ["model.gguf": data(1)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        try switchRecord(image: image, source: source, files: files)
        let leaving = SharedTextEncoder.ownFolder(ofImageModel: image) + ".removing"

        // Not what the record names: another size.
        try write(["model.gguf": data(1, 100)], to: leaving)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source]), .unrecognized(100))
        // A file the record does not list.
        try write(["model.gguf": files["model.gguf"]!, "other.bin": data(3, 10)], to: leaving)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source]), .unrecognized(4096 + 10))
        try FileManager.default.removeItem(atPath: leaving + "/other.bin")

        // The video model's file has changed since (the same size and first
        // bytes would do no better: its SHA-256 is no longer known).
        let encoder = SharedTextEncoder.folder(inVideoModel: source) + "/model.gguf"
        try data(9).write(to: URL(fileURLWithPath: encoder))
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source]), .restored,
                       "nothing verified to read: the whole copy is put back, not deleted")
        try FileManager.default.moveItem(atPath: SharedTextEncoder.ownFolder(ofImageModel: image), toPath: leaving)
        try switchRecord(image: image, source: try videoModel("Same", files), files: files)

        // What the record names, and a video model known to hold the same.
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [root + "/Same"]), .removed)
        XCTAssertFalse(FileManager.default.fileExists(atPath: leaving))
        XCTAssertNil(SharedTextEncoder.record(forImageModel: image)?.removing)
    }

    /// Without a record of a switch nothing is touched.
    func testALeftoverWithNoRecordIsKept() throws {
        let image = root + "/Image"
        let source = try videoModel("Source", ["model.gguf": data(1)])
        let leaving = SharedTextEncoder.ownFolder(ofImageModel: image) + ".removing"
        try write(["model.gguf": data(1)], to: leaving)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source]), .unrecognized(4096))
        XCTAssertEqual(names(in: leaving), ["model.gguf"])
    }
}
