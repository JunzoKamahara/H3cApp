// The enlarging page's comparison: two methods side by side on one clip, as a
// still or as video. The two videos start together and the second follows the
// first, so differences in motion can be seen in the same moment. Without
// JavaScript the panel stays hidden and the page reads as before.
(function () {
  "use strict";

  var root = document.getElementById("compare-panel");
  if (!root) return;

  var base = root.getAttribute("data-base");
  var clip = document.getElementById("compare-clip");
  var left = document.getElementById("compare-left");
  var right = document.getElementById("compare-right");
  var modes = root.querySelectorAll("[data-mode]");
  var speeds = root.querySelectorAll("[data-speed]");
  var play = document.getElementById("compare-play");
  var videoControls = document.getElementById("compare-video-controls");
  var note = document.getElementById("compare-note");
  var panes = [
    { select: left, img: root.querySelector("[data-pane='left'] img"), video: root.querySelector("[data-pane='left'] video"), label: root.querySelector("[data-pane='left'] figcaption") },
    { select: right, img: root.querySelector("[data-pane='right'] img"), video: root.querySelector("[data-pane='right'] video"), label: root.querySelector("[data-pane='right'] figcaption") }
  ];
  var mode = "still";
  var speed = 1;
  var playing = true;
  var english = (document.documentElement.lang || "").indexOf("en") === 0;

  function setPressed(buttons, attribute, value) {
    buttons.forEach(function (button) {
      button.setAttribute("aria-pressed", String(button.getAttribute(attribute) === value));
    });
  }

  function start() {
    var first = panes[0].video, second = panes[1].video;
    first.playbackRate = second.playbackRate = speed;
    if (!playing) {
      first.pause();
      second.pause();
      return;
    }
    if (!second.ended && Math.abs(second.currentTime - first.currentTime) > 0.05) second.currentTime = first.currentTime;
    [first, second].forEach(function (video) {
      var started = video.play();
      if (started && started.catch) started.catch(function () {});
    });
  }

  function follow() {
    var first = panes[0].video, second = panes[1].video;
    if (mode === "video" && playing && !first.paused && !second.seeking && Math.abs(second.currentTime - first.currentTime) > 0.05) {
      second.currentTime = first.currentTime;
    }
  }

  function show() {
    var name = clip.value;
    panes.forEach(function (pane) {
      var method = pane.select.value;
      var path = base + name + "/" + method;
      pane.label.textContent = pane.select.options[pane.select.selectedIndex].text;
      pane.img.hidden = mode !== "still";
      pane.video.hidden = mode !== "video";
      if (mode === "still") {
        pane.video.pause();
        pane.img.src = path + ".jpg";
        pane.img.alt = pane.label.textContent;
      } else if (pane.video.getAttribute("src") !== path + ".mp4") {
        pane.video.setAttribute("src", path + ".mp4");
        pane.video.setAttribute("aria-label", pane.label.textContent);
      }
    });
    videoControls.hidden = mode !== "video";
    note.textContent = clip.options[clip.selectedIndex].getAttribute("data-note");
    setPressed(modes, "data-mode", mode);
    if (mode === "video") start();
  }

  clip.addEventListener("change", show);
  left.addEventListener("change", show);
  right.addEventListener("change", show);
  modes.forEach(function (button) {
    button.addEventListener("click", function () {
      mode = button.getAttribute("data-mode");
      show();
    });
  });
  speeds.forEach(function (button) {
    button.addEventListener("click", function () {
      speed = Number(button.getAttribute("data-speed"));
      setPressed(speeds, "data-speed", button.getAttribute("data-speed"));
      start();
    });
  });
  play.addEventListener("click", function () {
    playing = !playing;
    play.textContent = playing ? (english ? "Pause" : "一時停止") : (english ? "Play" : "再生");
    start();
  });
  // Both restart together, so neither loops ahead of the other.
  panes[0].video.addEventListener("ended", function () {
    panes[0].video.currentTime = 0;
    panes[1].video.currentTime = 0;
    start();
  });

  root.hidden = false;
  show();
  window.setInterval(follow, 250);
})();
