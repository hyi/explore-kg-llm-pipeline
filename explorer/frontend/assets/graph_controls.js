(function () {
  window.dash_clientside = window.dash_clientside || {};
  window.dash_clientside.kgExplorer = window.dash_clientside.kgExplorer || {};

  function currentTriggerId() {
    const context = window.dash_clientside.callback_context;
    if (!context || !context.triggered || context.triggered.length === 0) {
      return null;
    }
    const propId = context.triggered[0].prop_id || "";
    return propId.split(".")[0];
  }

  window.dash_clientside.kgExplorer.zoomGraph = function (zoomInClicks, zoomOutClicks) {
    const triggerId = currentTriggerId();
    if (triggerId !== "zoom-in-button" && triggerId !== "zoom-out-button") {
      return window.dash_clientside.no_update;
    }

    const cy = window.cy;
    if (!cy || typeof cy.zoom !== "function") {
      return { ok: false, reason: "cytoscape-unavailable", at: Date.now() };
    }

    const factor = triggerId === "zoom-in-button" ? 1.25 : 0.8;
    const minZoom = typeof cy.minZoom === "function" ? cy.minZoom() : 0.15;
    const maxZoom = typeof cy.maxZoom === "function" ? cy.maxZoom() : 3;
    const nextZoom = Math.max(minZoom, Math.min(maxZoom, cy.zoom() * factor));
    const center = {
      x: cy.width() / 2,
      y: cy.height() / 2,
    };

    cy.zoom({
      level: nextZoom,
      renderedPosition: center,
    });

    return { ok: true, zoom: nextZoom, at: Date.now() };
  };
})();
