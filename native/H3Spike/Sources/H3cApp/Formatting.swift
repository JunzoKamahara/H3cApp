import Foundation

func formatElapsed(_ seconds: Double) -> String {
    let total = max(Int(seconds.rounded()), 0)
    let minutes = total / 60
    let secs = total % 60
    // A long つなげて生成 runs past an hour: h:mm:ss rather than 95:30.
    if minutes >= 60 { return String(format: "%d:%02d:%02d", minutes / 60, minutes % 60, secs) }
    return minutes > 0 ? String(format: "%d:%02d", minutes, secs) : String(localized: "\(secs)秒")
}

/// Joins short items in one-line summaries (" ・ " in Japanese).
let summarySeparator = String(localized: " ・ ")
/// Joins the facts about one file ("Q8_0・21.44 GB"; " · " in English).
let itemSeparator = String(localized: "・")

private let clockFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.timeStyle = .short
    formatter.dateStyle = .none
    return formatter
}()

private let dayAndClockFormatter: DateFormatter = {
    let formatter = DateFormatter()
    formatter.timeStyle = .short
    formatter.dateStyle = .short
    formatter.doesRelativeDateFormatting = true
    return formatter
}()

/// A finish time: the clock alone today, with the day ("明日 1:20") after.
func formatClockTime(_ date: Date) -> String {
    Calendar.current.isDateInToday(date)
        ? clockFormatter.string(from: date) : dayAndClockFormatter.string(from: date)
}

/// A lip-sync start offset in seconds, without trailing zeros ("12.5", "3").
func formatAudioOffset(_ seconds: Double) -> String {
    let rounded = (seconds * 100).rounded() / 100
    return rounded == rounded.rounded() ? String(Int(rounded)) : String(format: "%g", rounded)
}
