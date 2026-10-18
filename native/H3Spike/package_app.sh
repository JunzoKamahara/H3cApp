#!/bin/sh
# Assembles H3cApp.app from a release build of the H3cApp SPM target.
# The app is named H3cApp everywhere the user sees it - the bundle folder,
# the executable inside it, Info.plist, the Application Support folder
# (renamed from h3c-app at first launch, see h3AppSupportDirectory). The
# bundle ID (dev.kamahara.h3c-app) and the UserDefaults keys keep the old
# spelling so existing settings, presets and permissions carry over.
set -eu

script_dir=$(cd "$(dirname "$0")" && pwd)
repo_root=$(cd "$script_dir/../.." && pwd)
build_dir="$script_dir/.build"
app_bundle="$build_dir/H3cApp.app"
# This repo lives under ~/Documents, which Google Drive's desktop app
# watches/syncs by default; it periodically opens swift build's llbuild
# SQLite database (build.db) for that, and the resulting lock contention is
# what caused the "disk I/O error" below in practice, repeatably - not
# generic llbuild flakiness. Building to a scratch path outside that synced
# tree sidesteps it entirely (verified: reliable across many consecutive
# builds, vs. frequent failures at the in-tree default path).
scratch_dir="${TMPDIR:-/tmp}H3cApp-build-scratch"

# swift build's own llbuild database can still occasionally hit a transient
# "disk I/O error" writing its bookkeeping *after* compiling and linking
# have already succeeded, which still fails the process's exit code - so
# don't trust that exit code, and instead check for a freshly written
# binary below.
rm -f "$scratch_dir/release/H3cApp"
swift build --package-path "$script_dir" --scratch-path "$scratch_dir" -c release || true
if [ ! -x "$scratch_dir/release/H3cApp" ]; then
    echo "swift build did not produce $scratch_dir/release/H3cApp" >&2
    exit 1
fi

rm -rf "$app_bundle"
mkdir -p "$app_bundle/Contents/MacOS" "$app_bundle/Contents/Resources"
cp "$scratch_dir/release/H3cApp" "$app_bundle/Contents/MacOS/H3cApp"
cp "$repo_root/h3_shaders.metal" "$app_bundle/Contents/Resources/h3_shaders.metal"
cp "$script_dir/Packaging/AppIcon.icns" "$app_bundle/Contents/Resources/AppIcon.icns"
cp "$script_dir/Packaging/Info.plist" "$app_bundle/Contents/Info.plist"
# UI strings: Japanese keys, English and Japanese tables (see
# check_localizations.sh).
cp -R "$script_dir/Packaging/en.lproj" "$script_dir/Packaging/ja.lproj" "$app_bundle/Contents/Resources/"
# License texts travel with the binary (BSD/MIT/Apache require it for
# binary redistribution). ccv's COPYING also carries the licenses of the
# third-party code ccv bundles, so it's included whenever ccv is linked in.
cp "$repo_root/LICENSE" "$app_bundle/Contents/Resources/LICENSE"
cp "$repo_root/THIRD_PARTY_NOTICES.md" "$app_bundle/Contents/Resources/THIRD_PARTY_NOTICES.md"
if [ -n "${CCV_DIR:-}" ]; then
    cp "$CCV_DIR/COPYING" "$app_bundle/Contents/Resources/ccv-COPYING.txt"
fi

echo "Packaged $app_bundle"

# Signing/notarization are opt-in via env vars so the plain dev build above
# stays untouched when they're unset.
#   H3C_SIGN_IDENTITY   "Developer ID Application: NAME (TEAMID)" - required to sign
#   H3C_NOTARY_PROFILE  keychain profile from `notarytool store-credentials` - required to notarize
if [ -n "${H3C_SIGN_IDENTITY:-}" ]; then
    echo "Signing with identity: $H3C_SIGN_IDENTITY"
    codesign --force --deep --options runtime \
        --sign "$H3C_SIGN_IDENTITY" \
        "$app_bundle"
    codesign --verify --deep --strict --verbose=2 "$app_bundle"
    echo "Signed $app_bundle"

    if [ -n "${H3C_NOTARY_PROFILE:-}" ]; then
        zip_path="$build_dir/H3cApp.zip"
        rm -f "$zip_path"
        ditto -c -k --keepParent "$app_bundle" "$zip_path"
        echo "Submitting for notarization (profile: $H3C_NOTARY_PROFILE)..."
        xcrun notarytool submit "$zip_path" --keychain-profile "$H3C_NOTARY_PROFILE" --wait
        xcrun stapler staple "$app_bundle"
        rm -f "$zip_path"
        echo "Notarized and stapled $app_bundle"
    fi
fi
