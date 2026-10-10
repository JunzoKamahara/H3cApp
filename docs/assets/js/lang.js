// Sends a first-time visitor to the page in their browser's language:
// Japanese when the browser's first language is Japanese, English otherwise.
// A language chosen with the 日本語 / English switch (?lang=… on those
// links) is remembered and always wins. Crawlers are left alone; the pages
// declare each other with hreflang. Loaded in <head> so it runs before the
// page is drawn.
(function () {
  "use strict";
  var html = document.documentElement;
  var pageLang = (html.getAttribute("lang") || "").slice(0, 2);
  var alt = html.getAttribute("data-alt-href");
  if (!alt || (pageLang !== "ja" && pageLang !== "en")) return;

  var key = "h3c-site-lang";
  function load() {
    try { return localStorage.getItem(key); } catch (e) { return null; }
  }
  function save(value) {
    try { localStorage.setItem(key, value); } catch (e) { /* private mode */ }
  }
  function go() {
    var target = new URL(alt, location.href);
    target.hash = location.hash;
    location.replace(target.href);
  }

  var chosen = null;
  try { chosen = new URLSearchParams(location.search).get("lang"); } catch (e) {}
  if (chosen === "ja" || chosen === "en") {
    save(chosen);
    return;
  }

  var stored = load();
  if (stored === "ja" || stored === "en") {
    if (stored !== pageLang) go();
    return;
  }

  if (/bot|crawl|spider|slurp|facebookexternalhit|embedly|preview/i.test(navigator.userAgent || "")) return;
  var languages = navigator.languages && navigator.languages.length
    ? navigator.languages : [navigator.language || ""];
  var wanted = /^ja(-|$)/i.test(languages[0] || "") ? "ja" : "en";
  if (wanted !== pageLang) go();
})();
