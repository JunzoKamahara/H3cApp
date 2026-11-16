// The image comparison page: in each panel, two pictures of one subject side
// by side - two image models, or two ways of enlarging. Each panel names its
// folder (data-base); a picture is <base><subject>/<choice>.jpg. Without
// JavaScript the panels stay hidden and the tables below them still read.
(function () {
  "use strict";

  var english = (document.documentElement.lang || "").indexOf("en") === 0;

  document.querySelectorAll("[data-image-compare]").forEach(function (root) {
    var base = root.getAttribute("data-base");
    var subject = root.querySelector("[data-role='subject']");
    var note = root.querySelector("[data-role='note']");
    var where = root.querySelector("[data-role='where']");
    var panes = ["left", "right"].map(function (side) {
      var figure = root.querySelector("[data-pane='" + side + "']");
      return {
        select: root.querySelector("[data-role='" + side + "']"),
        img: figure.querySelector("img"),
        label: figure.querySelector("figcaption")
      };
    });

    function show() {
      var name = subject.value;
      var option = subject.options[subject.selectedIndex];
      panes.forEach(function (pane) {
        var choice = pane.select.options[pane.select.selectedIndex];
        var caption = choice.text;
        var detail = choice.getAttribute("data-" + name);
        pane.label.textContent = detail ? caption + (english ? " (" + detail + ")" : "（" + detail + "）") : caption;
        pane.img.src = base + name + "/" + pane.select.value + ".jpg";
        pane.img.alt = option.text + ": " + caption;
      });
      if (note) note.textContent = option.getAttribute("data-note") || "";
      if (where) {
        where.src = base + name + "/where.jpg";
        where.alt = option.getAttribute("data-where") || "";
      }
    }

    [subject].concat(panes.map(function (pane) { return pane.select; })).forEach(function (select) {
      select.addEventListener("change", show);
    });
    root.hidden = false;
    show();
  });
})();
