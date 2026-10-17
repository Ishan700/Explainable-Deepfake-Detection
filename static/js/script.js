/* Upload page: drag & drop, filename echo, submit spinner.
   Result page: segmented control that swaps the evidence image.
   Both pages: light/dark theme toggle (remembered in localStorage). */
(function () {
  "use strict";

  // ---------- theme ----------
  var root = document.documentElement;
  var toggle = document.getElementById("theme-toggle");

  function currentTheme() {
    var set = root.getAttribute("data-theme");
    if (set === "light" || set === "dark") return set;
    return window.matchMedia && window.matchMedia("(prefers-color-scheme: dark)").matches ? "dark" : "light";
  }
  if (toggle) {
    toggle.addEventListener("click", function () {
      var next = currentTheme() === "dark" ? "light" : "dark";
      root.setAttribute("data-theme", next);
      try { localStorage.setItem("theme", next); } catch (e) { /* storage may be unavailable */ }
    });
  }

  // ---------- upload page ----------
  var dz = document.getElementById("dropzone");
  var input = document.getElementById("media-input");
  var fileLabel = document.getElementById("dz-file");
  var form = document.getElementById("upload-form");
  var submitBtn = document.getElementById("submit-btn");

  function showFile(file) {
    if (!file) return;
    var mb = (file.size / (1024 * 1024)).toFixed(1);
    fileLabel.textContent = file.name + " (" + mb + " MB)";
    fileLabel.hidden = false;
    dz.classList.add("has-file");
  }

  if (dz && input) {
    ["dragenter", "dragover"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.add("drag"); });
    });
    ["dragleave", "drop"].forEach(function (ev) {
      dz.addEventListener(ev, function (e) { e.preventDefault(); dz.classList.remove("drag"); });
    });
    dz.addEventListener("drop", function (e) {
      if (e.dataTransfer && e.dataTransfer.files && e.dataTransfer.files.length) {
        input.files = e.dataTransfer.files;
        showFile(input.files[0]);
      }
    });
    input.addEventListener("change", function () { showFile(input.files[0]); });
  }

  if (form && submitBtn) {
    form.addEventListener("submit", function () {
      submitBtn.classList.add("loading");
      submitBtn.querySelector(".btn-label").textContent = "Analyzing";
    });
  }

  // ---------- result page ----------
  document.querySelectorAll(".tabs").forEach(function (group) {
    var img = document.getElementById(group.dataset.target);
    if (!img) return;
    group.querySelectorAll(".tab").forEach(function (btn) {
      btn.addEventListener("click", function () {
        group.querySelectorAll(".tab").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        img.src = btn.dataset.src;
        img.classList.toggle("contain", btn.dataset.fit === "contain");
      });
      var pre = new Image(); pre.src = btn.dataset.src; // preload so switching is instant
    });
  });
})();
