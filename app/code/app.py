"""
CarValuate: Dash multi-page app covering Assignments 1-3.

Four pages share one instrument-panel-styled nav bar:
  /         home page comparing all three models
  /old      A1 model (scikit-learn pipeline, regression)        -- unchanged from A2
  /new      A2 model (from-scratch linear regression)           -- unchanged from A2
  /predict  A3 model (from-scratch multinomial logistic regression, 4 price classes)
"""

import dash
import dash_bootstrap_components as dbc
from dash import Dash, html, dcc, Input, Output

app = Dash(
    __name__,
    use_pages=True,
    external_stylesheets=[
        dbc.themes.BOOTSTRAP,  # required by the A1 page (dbc.Card, dbc.Row, ...)
        "https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap",
        "https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css",
    ],
    suppress_callback_exceptions=True,
)
app.title = "CarValuate | AI Price Predictor"
server = app.server

NAV_ITEMS = [
    ("nav-home", "Home", "/"),
    ("nav-old", "A1 model", "/old"),
    ("nav-new", "A2 model", "/new"),
    ("nav-predict", "A3 classifier", "/predict"),
]

navbar = html.Div(
    [
        html.A(
            [html.Span(className="cv-brand-mark"), "CarValuate"],
            href="/",
            className="cv-brand",
        ),
        html.Div(
            [dcc.Link(label, href=href, id=id_, className="cv-nav-link") for id_, label, href in NAV_ITEMS],
            className="cv-nav-links",
        ),
    ],
    className="cv-navbar",
)

app.layout = html.Div(
    [
        dcc.Location(id="cv-url"),
        navbar,
        dash.page_container,
    ]
)


@dash.callback(
    [Output(id_, "className") for id_, _, _ in NAV_ITEMS],
    Input("cv-url", "pathname"),
)
def highlight_active_tab(pathname):
    base = "cv-nav-link"
    return [f"{base} active" if pathname == href else base for _, _, href in NAV_ITEMS]


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=8050)
