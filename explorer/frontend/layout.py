from __future__ import annotations

from dash import dcc, html

from explorer.frontend.constants import DEFAULT_PATH_HITS
from explorer.frontend.state import empty_session_store, new_client_state


def build_layout() -> html.Div:
    return html.Div(
        [
            dcc.Store(id="client-state-store", storage_type="memory", data=new_client_state()),
            dcc.Store(id="session-store", storage_type="memory", data=empty_session_store()),
            dcc.Store(id="candidate-paths-store", storage_type="memory", data=[]),
            dcc.Store(id="selected-path-id-store", storage_type="memory"),
            dcc.Store(id="context-store", storage_type="memory", data={}),
            dcc.Store(id="graph-viewport-signal", storage_type="memory"),
            html.Header(
                [
                    html.Div(
                        [
                            html.H1("Human-Guided Knowledge Graph Explorer", className="app-title"),
                            html.P(
                                "Search paths, inspect graph context, expand what looks useful, and save discoveries in the current session.",
                                className="subtitle",
                            ),
                        ],
                        className="hero-copy",
                    ),
                    html.Div(
                        [
                            html.Div("Search -> Candidate Paths -> Human Exploration -> Save Session"),
                            html.Div(id="session-metrics", className="metrics"),
                        ],
                        className="hero-card",
                    ),
                ],
                className="hero",
            ),
            html.Main(
                [
                    html.Section(
                        [
                            html.Label("Natural language query", htmlFor="query-input"),
                            dcc.Input(
                                id="query-input",
                                type="text",
                                placeholder="Example: genes involved in chemoresistance in cancer",
                                debounce=True,
                                className="query-input",
                            ),
                            html.Div(
                                [
                                    html.Div(
                                        [
                                            html.Label("Semantic path hits"),
                                            dcc.Slider(
                                                id="path-hit-slider",
                                                min=1,
                                                max=25,
                                                step=1,
                                                value=DEFAULT_PATH_HITS,
                                                marks={1: "1", 5: "5", 10: "10", 15: "15", 20: "20", 25: "25"},
                                                tooltip={"placement": "bottom", "always_visible": False},
                                            ),
                                        ],
                                        className="slider-wrap",
                                    ),
                                    html.Button("Search paths", id="search-button", n_clicks=0, className="primary-button"),
                                ],
                                className="search-actions",
                            ),
                            dcc.Loading(
                                html.Div(id="status-message", className="status"),
                                type="circle",
                                delay_show=250,
                            ),
                        ],
                        className="panel search-panel",
                    ),
                    html.Section(
                        [
                            html.Div(
                                [
                                    html.H2("Candidate Paths"),
                                    html.Div(id="candidate-paths", className="path-list"),
                                ],
                                className="panel candidates-panel",
                            ),
                            html.Div(id="path-details", className="panel details-panel"),
                        ],
                        className="workspace",
                    ),
                    html.Section(
                        [
                            html.H2("Saved Session"),
                            html.Div(id="saved-session"),
                        ],
                        className="panel",
                    ),
                ],
                className="page",
            ),
        ],
    )
