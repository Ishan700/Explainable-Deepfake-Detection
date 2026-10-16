/* Upload page: drag & drop, filename echo, submit spinner.
   Result page: segmented control that swaps the Grad-CAM / regions / crop image. */
(function () {
  "use strict";

  // ---------- upload page ----------
  const dz = document.getElementById("dropzone");
  const input = document.getElementById("media-input");
  const fileLabel = document.getElementById("dz-file");
  const form = document.getElementById("upload-form");
  const submitBtn = document.getElementById("submit-btn");

  function showFile(file) {
    if (!file) return;
    const mb = (file.size / (1024 * 1024)).toFixed(1);
    fileLabel.textContent = file.name + "  (" + mb + " MB)";
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
      submitBtn.querySelector(".btn-label").textContent = "Analyzing…";
    });
  }

  // ---------- result page ----------
  document.querySelectorAll(".segmented").forEach(function (group) {
    const img = document.getElementById(group.dataset.target);
    if (!img) return;
    group.querySelectorAll(".seg").forEach(function (btn) {
      btn.addEventListener("click", function () {
        group.querySelectorAll(".seg").forEach(function (b) { b.classList.remove("active"); });
        btn.classList.add("active");
        img.src = btn.dataset.src;
      });
    });
    // Preload the alternatives so switching is instant.
    group.querySelectorAll(".seg").forEach(function (btn) { const i = new Image(); i.src = btn.dataset.src; });
  });
})();
