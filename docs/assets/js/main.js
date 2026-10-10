// Copy buttons for prompt examples. Without JavaScript the text stays in a
// read-only textarea that can be selected and copied by hand; the buttons
// are only shown once this script runs.
(function () {
  "use strict";

  function setStatus(el, text, kind) {
    el.textContent = text;
    el.classList.remove("ok", "error");
    if (kind) el.classList.add(kind);
  }

  function fallbackCopy(textarea) {
    textarea.focus();
    textarea.select();
    try {
      return document.execCommand("copy");
    } catch (e) {
      return false;
    }
  }

  var english = (document.documentElement.lang || "").indexOf("en") === 0;
  var messages = english ? {
    ok: "Copied. Paste it into the app's prompt field.",
    error: "Couldn't copy automatically. Press ⌘C to copy the selected text."
  } : {
    ok: "コピーしました。アプリのプロンプト欄に貼り付けてください。",
    error: "自動でコピーできませんでした。選択された文字を ⌘C でコピーしてください。"
  };

  document.querySelectorAll("[data-copy-target]").forEach(function (button) {
    var target = document.getElementById(button.getAttribute("data-copy-target"));
    var status = document.getElementById(button.getAttribute("data-copy-status"));
    if (!target || !status) return;
    button.hidden = false;

    button.addEventListener("click", function () {
      var text = target.value;
      var done = function () {
        setStatus(status, messages.ok, "ok");
      };
      var failed = function () {
        target.focus();
        target.select();
        setStatus(status, messages.error, "error");
      };
      if (navigator.clipboard && window.isSecureContext) {
        navigator.clipboard.writeText(text).then(done, function () {
          if (fallbackCopy(target)) done(); else failed();
        });
      } else if (fallbackCopy(target)) {
        done();
      } else {
        failed();
      }
    });
  });
})();
