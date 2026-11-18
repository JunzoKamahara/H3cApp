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

    /// The video model named is the one offered, not only when it is the
    /// first that would do.
    func testASwitchToTheVideoModelNamed() throws {
        let contents = ["model.gguf": Data((0 ..< (2 << 20)).map { UInt8(truncatingIfNeeded: $0 &* 31) })]
        let first = try videoModel("First", contents)
        let second = try videoModel("Second", contents)
        let different = try videoModel("Different", ["model.gguf": data(3, 2 << 20)])
        let image = root + "/Image"
        try write(contents, to: SharedTextEncoder.ownFolder(ofImageModel: image))
        let listed = [first, second, different]
        func offer(_ named: String?) -> String? {
            SharedTextEncoder.possibleSwitch(forImageModel: image, videoModelPaths: listed, named: named)?.videoModel
        }
        XCTAssertEqual(offer(nil), first)
        XCTAssertEqual(offer(first), first)
        XCTAssertEqual(offer(second), second, "a later one that holds the same")
        XCTAssertNil(offer(different), "one that does not")
        XCTAssertNil(offer(root + "/NotListed"))
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

    /// What is left of a copy goes only for the encoder the copy was
    /// compared with - not for another one chosen since.
    func testALeftoverIsNotDeletedForAnEncoderChosenSince() throws {
        let files = ["a.gguf": data(1), "b.gguf": data(2, 8192)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        try switchRecord(image: image, source: source, files: files)
        let leaving = SharedTextEncoder.ownFolder(ofImageModel: image) + ".removing"
        // Part of the copy was deleted, and the video model has gone since.
        try write(["b.gguf": files["b.gguf"]!], to: leaving)
        try FileManager.default.removeItem(atPath: source)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: []), .left(8192))

        // Another video model, with a different encoder, is chosen.
        let other = try videoModel("Other", ["a.gguf": data(8), "b.gguf": data(9, 8192)])
        let marks = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: other))
        try SharedTextEncoder.choose(videoModel: other, files: marks, forImageModel: image)
        let record = try XCTUnwrap(SharedTextEncoder.record(forImageModel: image))
        XCTAssertEqual(record.files, marks)
        XCTAssertNotNil(record.removing)
        XCTAssertNotEqual(record.removingEncoder, marks, "what the copy was compared with stays on record")

        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [other]), .left(8192),
                       "the encoder chosen now is not the one the copy was the same as")
        XCTAssertEqual(names(in: leaving), ["b.gguf"])
        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: image, videoModelPaths: [other]), .shared(videoModel: other))

        // The first video model back, with the encoder the copy was compared with.
        let back = try videoModel("Source", files)
        _ = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: back))
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [other, back]), .removed)
        let after = try XCTUnwrap(SharedTextEncoder.record(forImageModel: image))
        XCTAssertEqual(after.videoModel, other, "the choice stays")
        XCTAssertNil(after.removing)
        XCTAssertNil(after.removingEncoder)
    }

    /// What a switch left is settled on one thread while another encoder
    /// is chosen on another: whichever comes first, the choice is what the
    /// record says afterwards (the settling does not write the record it
    /// read before the choice over it), and the leftover has gone.
    func testAChoiceMadeWhileALeftoverIsSettledIsKept() async throws {
        // Many small files: the settling takes long enough to be met halfway.
        var files: [String: Data] = [:], others: [String: Data] = [:]
        for index in 0 ..< 48 {
            files["part-\(index).gguf"] = data(UInt8(index), 1024)
            others["part-\(index).gguf"] = data(UInt8(200 - index), 1024)
        }
        let source = try videoModel("Source", files)
        let other = try videoModel("Other", others)
        let sourceMarks = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: source))
        let otherMarks = try SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: other))
        let manifest = files.mapValues { Int64($0.count) }

        for round in 0 ..< 60 {
            let image = root + "/Image-\(round)"
            let leaving = SharedTextEncoder.ownFolder(ofImageModel: image) + ".removing"
            try write(files, to: leaving)
            try SharedTextEncoder.write(
                SharedTextEncoder.Record(videoModel: source, files: sourceMarks, removing: manifest, removingEncoder: sourceMarks),
                forImageModel: image)
            // The choice begins a little later each round, up to about the
            // time the settling takes.
            let delay = UInt32(round % 20) * 150
            let settling = Task.detached {
                SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source, other])
            }
            let choosing = Task.detached {
                usleep(delay)
                try SharedTextEncoder.choose(videoModel: other, files: otherMarks, forImageModel: image)
            }
            let settled = await settling.value
            try await choosing.value
            let record = try XCTUnwrap(SharedTextEncoder.record(forImageModel: image))
            XCTAssertEqual(settled, .removed, "round \(round)")
            XCTAssertEqual(record.videoModel, other, "round \(round): the choice is not written over")
            XCTAssertEqual(record.files, otherMarks, "round \(round)")
            XCTAssertNil(record.removing, "round \(round)")
            XCTAssertNil(record.removingEncoder, "round \(round)")
            XCTAssertFalse(FileManager.default.fileExists(atPath: leaving), "round \(round)")
            if record.videoModel != other { break }
        }
    }

    // MARK: A switch and the deletion of its video model

    /// Once the deletion of a video model's files is decided, no copy is
    /// given up for it and no leftover deleted for it: the copy would be
    /// the last one.
    func testNoCopyGoesForAVideoModelBeingDeleted() throws {
        let files = ["model.gguf": data(1)]
        let source = try videoModel("Source", files)
        let image = root + "/Image"
        let own = SharedTextEncoder.ownFolder(ofImageModel: image)
        try write(files, to: own)
        let compared = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in }

        // The comparison is done; the deletion is decided before the switch.
        XCTAssertEqual(SharedTextEncoder.beginRemoval(ofVideoModel: source, imageModels: [(name: "Image", path: image)],
                                                      videoModelPaths: [source]).count, 0, "the image model has its own")
        XCTAssertThrowsError(try SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: compared) { _, _ in })
        XCTAssertEqual(names(in: own), ["model.gguf"], "the copy stays")
        XCTAssertNil(SharedTextEncoder.record(forImageModel: image))
        XCTAssertNil(SharedTextEncoder.possibleSwitch(forImageModel: image, videoModelPaths: [source]))
        XCTAssertThrowsError(try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in })

        // What an earlier switch left is not deleted for it either: put back.
        let leaving = own + ".removing"
        try FileManager.default.moveItem(atPath: own, toPath: leaving)
        try SharedTextEncoder.write(
            SharedTextEncoder.Record(videoModel: source, files: compared.files, removing: ["model.gguf": 4096],
                                     removingEncoder: compared.files), forImageModel: image)
        XCTAssertEqual(SharedTextEncoder.finishInterruptedSwitch(imageModel: image, videoModelPaths: [source]), .restored)
        XCTAssertEqual(names(in: own), ["model.gguf"])

        // The deletion not made after all: the switch goes through, and a
        // deletion asked for after it is told of the image model.
        SharedTextEncoder.endRemoval(ofVideoModel: source)
        try SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: compared) { _, _ in }
        XCTAssertFalse(FileManager.default.fileExists(atPath: own))
        XCTAssertEqual(SharedTextEncoder.beginRemoval(ofVideoModel: source, imageModels: [(name: "Image", path: image)],
                                                      videoModelPaths: [source]).map(\.directory), [image])
        SharedTextEncoder.endRemoval(ofVideoModel: source)
    }

    /// A copy is given up for a video model on one thread while that video
    /// model's files are deleted on another, as Model Management does it:
    /// whichever comes first, the image model is left with an encoder to
    /// read - its copy, or the video model's, which is then not deleted
    /// without a word about the image model.
    func testASwitchAndTheDeletionOfItsVideoModelAtOnce() async throws {
        var files: [String: Data] = [:]
        for index in 0 ..< 32 { files["part-\(index).gguf"] = data(UInt8(index), 1024) }
        // One video model for every round (put back after a round that
        // deleted it: the same files, whose SHA-256 stays known).
        let source = try videoModel("Source", files)
        let trash = root + "/Trash"
        var switched = 0, deleted = 0
        for round in 0 ..< 80 {
            let image = root + "/Image-\(round)"
            let own = SharedTextEncoder.ownFolder(ofImageModel: image)
            try write(files, to: own)
            let compared = try SharedTextEncoder.compareContents(imageModel: image, videoModel: source) { _, _ in }
            // Now the switch ahead, now the deletion, by up to about the
            // time the switch takes to check the copy before deleting it.
            let lag = UInt32(round / 2 % 40) * 100
            let switchLag = round % 2 == 0 ? 0 : lag, deleteLag = round % 2 == 0 ? lag : 0
            let switching = Task.detached { () -> Bool in
                usleep(switchLag)
                return (try? SharedTextEncoder.switchToShared(imageModel: image, videoModel: source, compared: compared) { _, _ in }) != nil
            }
            let deleting = Task.detached { () -> Bool in
                usleep(deleteLag)
                // The sheet warned of no image model: this one had its own.
                let readers = SharedTextEncoder.beginRemoval(ofVideoModel: source, imageModels: [(name: "Image", path: image)],
                                                             videoModelPaths: [source])
                defer { SharedTextEncoder.endRemoval(ofVideoModel: source) }
                guard readers.isEmpty else { return false }
                return (try? FileManager.default.moveItem(atPath: source, toPath: trash)) != nil
            }
            let didSwitch = await switching.value, didDelete = await deleting.value
            let copy = names(in: own).count == files.count
            let encoder = names(in: SharedTextEncoder.folder(inVideoModel: source)).count == files.count
            XCTAssertTrue(copy || encoder, "round \(round): nothing left to read (switched \(didSwitch), deleted \(didDelete))")
            XCTAssertFalse(didSwitch && didDelete, "round \(round): the copy and the video model both went")
            XCTAssertEqual(copy, !didSwitch, "round \(round)")
            XCTAssertEqual(encoder, !didDelete, "round \(round)")
            switched += didSwitch ? 1 : 0
            deleted += didDelete ? 1 : 0
            if !(copy || encoder) { break }
            if didDelete { try FileManager.default.moveItem(atPath: trash, toPath: source) }
        }
        // Both orders were met.
        XCTAssertGreaterThan(switched, 0)
        XCTAssertGreaterThan(deleted, 0)
    }

    // MARK: What is known of a file

    /// The SHA-256 found by reading a file through is that file's: another
    /// put in its place is not known, also with the same size and date.
    func testAFilePutInPlaceOfOneReadThroughIsNotKnown() throws {
        let video = try videoModel("Video", ["model.gguf": data(1)])
        let folder = SharedTextEncoder.folder(inVideoModel: video)
        let file = folder + "/model.gguf"
        let whole = Date(timeIntervalSinceReferenceDate: (Date().timeIntervalSinceReferenceDate - 100).rounded(.down))
        try FileManager.default.setAttributes([.modificationDate: whole], ofItemAtPath: file)
        let read = try SharedTextEncoder.fullFingerprint(of: folder)
        XCTAssertNotNil(read["model.gguf"]?.sha256)
        XCTAssertEqual(SharedTextEncoder.fingerprint(of: folder), read, "known while it is the file that was read")

        // Another file of the same size, given the same date.
        try data(9).write(to: URL(fileURLWithPath: file), options: .atomic)
        try FileManager.default.setAttributes([.modificationDate: whole], ofItemAtPath: file)
        XCTAssertEqual(try FileManager.default.attributesOfItem(atPath: file)[.modificationDate] as? Date, whole)
        let now = try XCTUnwrap(SharedTextEncoder.fingerprint(of: folder)?["model.gguf"])
        XCTAssertNil(now.sha256, "not the file whose SHA-256 was found")
        XCTAssertNotEqual(now.head, read["model.gguf"]?.head, "and its first bytes are read anew")
        XCTAssertFalse(SharedTextEncoder.isSame(SharedTextEncoder.fingerprint(of: folder), read))
        XCTAssertEqual(try SharedTextEncoder.fullFingerprint(of: folder)["model.gguf"]?.sha256, FileHash.sha256(of: data(9)))
    }

    // MARK: Deleting a video model

    /// A video model's files are not deleted from under an image being
    /// generated in any window while an image model reads its text
    /// encoder; and once the deletion is decided, nothing new reads it.
    func testAVideoModelsFilesStayWhileAnImageThatReadsThemIsGenerated() throws {
        let files = ["model.gguf": data(1)]
        let read = try videoModel("Read", files)
        let unread = try videoModel("Unread", ["model.gguf": data(2)])
        let image = root + "/Image"
        try FileManager.default.createDirectory(atPath: image, withIntermediateDirectories: true)
        try SharedTextEncoder.choose(
            videoModel: read, files: SharedTextEncoder.fullFingerprint(of: SharedTextEncoder.folder(inVideoModel: read)),
            forImageModel: image)
        let images = [(name: "Image", path: image)]
        func problem(_ video: String) -> String? {
            SharedTextEncoder.removalProblem(videoModel: video, imageModels: images, videoModelPaths: [read, unread])
        }
        XCTAssertNil(problem(read), "nothing is generated")

        let window = anotherWindow()
        window.imageGenerationBegan(index: 1, total: 1)
        XCTAssertNotNil(problem(read))
        XCTAssertNil(problem(unread), "no image model reads that one")
        AppModels.shared.release(window)
        XCTAssertNotNil(problem(read), "its window closed, the generation not yet stopped")
        window.imageGenerationEnded()
        XCTAssertNil(problem(read))

        // A folder with no record reads the video model in use in its
        // window - here the second of the list, which a job was started with.
        let loose = root + "/Loose"
        try FileManager.default.createDirectory(atPath: loose, withIntermediateDirectories: true)
        func looseProblem(_ video: String) -> String? {
            SharedTextEncoder.removalProblem(videoModel: video, imageModels: [(name: "Loose", path: loose)],
                                             videoModelPaths: [read, unread])
        }
        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: loose, videoModelPaths: [read, unread]), .shared(videoModel: read))
        let alias = root + "/UnreadAlias"
        try FileManager.default.createSymbolicLink(atPath: alias, withDestinationPath: unread)
        window.imageGenerationBegan(index: 1, total: 1)
        GenerationViewModel.imageGenerators[ObjectIdentifier(window)] =
            (model: "Loose", textEncoder: SharedTextEncoder.folder(inVideoModel: alias))
        XCTAssertNotNil(looseProblem(unread), "the folder the job reads, whatever the list's order says")
        window.imageGenerationBegan(index: 2, total: 2)
        XCTAssertNotNil(looseProblem(unread), "also for the later images of a batch")
        window.imageGenerationEnded()
        XCTAssertNil(looseProblem(unread))
        XCTAssertNil(looseProblem(read))

        // The deletion decided: not read any more, before the files have gone.
        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: image, videoModelPaths: [read, unread]), .shared(videoModel: read))
        SharedTextEncoder.beginRemoval(ofVideoModel: read)
        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: image, videoModelPaths: [read, unread]), .needsChoice)
        XCTAssertNotNil(GenerationViewModel.imageTextEncoderProblem(model: image, name: "Image", videoModelPaths: [read, unread]))
        SharedTextEncoder.endRemoval(ofVideoModel: read)
        XCTAssertEqual(SharedTextEncoder.reading(forImageModel: image, videoModelPaths: [read, unread]), .shared(videoModel: read))
    }

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
            SharedTextEncoder.Record(videoModel: source, files: marks, removing: files.mapValues { Int64($0.count) },
                                     removingEncoder: marks),
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
