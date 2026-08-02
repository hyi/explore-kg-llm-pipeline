from __future__ import annotations

import os

from dash import Dash

from explorer.frontend.callbacks import register_callbacks
from explorer.frontend.layout import build_layout


def create_app() -> Dash:
    app = Dash(__name__, suppress_callback_exceptions=True, title="Human-Guided Knowledge Graph Explorer")
    app.layout = build_layout()
    register_callbacks(app)
    return app


app = create_app()
server = app.server


if __name__ == "__main__":
    app.run(
        host=os.getenv("HOST", "0.0.0.0"),
        port=int(os.getenv("PORT", "8050")),
        debug=os.getenv("DASH_DEBUG", "").lower() in {"1", "true", "yes"},
    )
