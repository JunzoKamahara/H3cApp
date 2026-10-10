# H3cApp site: publishing and updating

The GitHub Pages site lives in `docs/`: four static HTML pages, one CSS
file, one small script, and media. No build step, no analytics, no
external requests at page load (the YouTube and other external pages are
plain links).

```
docs/
  index.html              top page (#examples #features #requirements #flow #performance #download)
  getting-started.html    #install #models #compute #first-video
  guide.html              #images #projects #speed #size #prompts #advanced
  faq.html                #model-not-found #slow #download #fast-mode #saved
                          #project-required #old-mac #gatekeeper #update #report #license
  en/                     English versions of the four pages, same file names and
                          the same anchors; assets are shared via ../assets/
  assets/css/style.css
  assets/js/main.js       copy buttons only
  assets/images/          app-window.jpg, app-composer.jpg, og-image.jpg, icons
  assets/video/           ex1-3.mp4 and posters
  .nojekyll  robots.txt  sitemap.xml
website-facts.md          sources for every number and UI name (not published)
note-handoff.md           brief for the note article (not published)
```

All links inside the site are relative, so it works under the project
path `/H3cApp/` and from a local folder.

Japanese is the main language (`/H3cApp/`, x-default); English lives in
`/H3cApp/en/`. Every page links its counterpart in the header and footer
(the 日本語 / English switch) and declares both with `hreflang`. Keep the
two in step: same sections, same anchors, same numbers. English UI names
are the app's own English strings (`en.lproj/Localizable.strings` of the
release), e.g. Export Video…, Fast (int8 cache), Advanced Settings.
`main.js` picks the copy-button messages from `<html lang>`.

## Publishing (first time)

0. The site URL is `https://junzokamahara.github.io/H3cApp/`, which needs
   the repository to be named **H3cApp** (Settings → General → Repository
   name; renamed from h3c-app). GitHub redirects the old repository URLs
   (web, clone, release downloads); update a local clone with
   `git remote set-url origin https://github.com/JunzoKamahara/H3cApp.git`.
   Never create another repository named h3c-app, or the redirects stop.
1. Merge the site branch into `main` (the site must be on the branch Pages
   serves).
2. GitHub → repository **Settings → Pages** → **Build and deployment** →
   Source: **Deploy from a branch** → Branch: **main**, folder **/docs** →
   Save. (No Actions workflow is needed for a static site; see
   https://docs.github.com/en/pages/getting-started-with-github-pages/creating-a-github-pages-site)
3. After the first deployment finishes (Settings → Pages shows the URL),
   open `https://junzokamahara.github.io/H3cApp/` and check the four pages,
   the videos and the copy button.
4. If the URL differs (custom domain, renamed repository), update in every
   page: `<link rel="canonical">`, `og:url`, `og:image`; and
   `robots.txt`, `sitemap.xml`, `website-facts.md`, `note-handoff.md`.
5. Optionally set the repository's **About → Website** to the site URL.

## Local preview

The site must also work under `/H3cApp/`, so preview it there:

```bash
mkdir -p /tmp/h3c-site && ln -sfn "$PWD/docs" /tmp/h3c-site/H3cApp
python3 -m http.server 8765 --directory /tmp/h3c-site
```

Then open http://localhost:8765/H3cApp/. Check widths around 1440, 768
and 390 px (no horizontal scroll), the anchors in the table of contents,
video playback with sound, and the copy button (it needs a secure context
- localhost counts - and falls back to selecting the text).

## When a new release comes out

1. Collect the facts again and update `website-facts.md`: release tag,
   commit, date, DMG name and size (`gh release view --json assets`),
   `LSMinimumSystemVersion`, UI strings that changed (compare the
   release's `ja.lproj/Localizable.strings` and the views listed in
   `website-facts.md`), model sizes if the Hugging Face repository
   changed.
2. Describe only what the release contains, not `main`.
3. Update the version text: index.html (#download paragraph and the note
   with the DMG name), getting-started.html (#install step 2), and the
   "v0.4.4" mentions (search the pages for `0.4.4`).
4. If the UI changed, retake `app-window.jpg` from the released app and
   replace any wording that no longer matches.
5. Re-run the local preview checks, update `<lastmod>` in `sitemap.xml`,
   commit, and merge to `main`.

## Media guidelines

- Generated examples must come from H3cApp itself, with the prompt, size,
  length, settings and version recorded in `website-facts.md`.
- Videos are re-encoded for the web (H.264 + AAC, `+faststart`), shown with
  controls, never autoplaying, with a poster and `preload="none"`.
- Screenshots are real windows (`screencapture -l <window id>`); never mock
  up UI.

### Screenshots still missing

The agent that built the site could not operate the app's UI (only the
HTTP API), so these real screens were not captured. Add them to the
relevant sections when available:

- First-launch download wizard ("MiniMax-H3 モデルのダウンロード"), idle and
  during download → getting-started.html#models
- Model manager → getting-started.html#models / faq.html#model-not-found
- Advanced settings, 計算方式 and サイズ sections → getting-started.html#compute,
  guide.html#speed
- "画像から" with 最初・最後の画像 and with 参照画像・動画・音声 → guide.html#images
- English-UI versions of app-window.jpg and app-composer.jpg for the en/
  pages (they currently show the Japanese UI, captioned as such): run the
  release app with macOS (or the app's language) set to English
- "プロジェクトを選んで生成" sheet and the project video list → guide.html#projects
