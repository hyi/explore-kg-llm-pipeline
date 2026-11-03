(function () {
  function labelPathHitSlider() {
    const slider = document.getElementById("path-hit-slider");
    const input = slider && slider.querySelector('input[type="number"]');
    if (!input) {
      return null;
    }

    input.id = "path-hit-slider-input";
    input.name = "semantic-path-hits";
    input.setAttribute("aria-labelledby", "path-hit-slider-label");
    const label = document.getElementById("path-hit-slider-label");
    if (label) {
      label.htmlFor = input.id;
    }

    slider.querySelectorAll('[role="slider"]').forEach(function (handle) {
      handle.setAttribute("aria-labelledby", "path-hit-slider-label");
    });
    return slider;
  }

  const layoutObserver = new MutationObserver(function () {
    const slider = labelPathHitSlider();
    if (!slider) {
      return;
    }
    layoutObserver.disconnect();
    const sliderObserver = new MutationObserver(labelPathHitSlider);
    sliderObserver.observe(slider, { childList: true, subtree: true });
  });

  layoutObserver.observe(document.documentElement, { childList: true, subtree: true });
})();
