"""
Predict page — the classifier form. Loads the model from the MLflow
Model Registry when reachable, falling back to the local bundle
(app/code/model/car_price_classifier.pkl) saved by the training
notebook.
"""

import dash
import pandas as pd
from dash import Input, Output, State, dcc, html

import model_inference as mi

dash.register_page(__name__, path="/predict", name="Predict")

PREDICT_FN, CLASS_LABELS, CLASS_RANGES, MODEL_SOURCE = mi.load_predictor()
OWNER_OPTIONS = list(mi.load_bundle()["owner_scale"].keys())


def field(id_, label, unit=None, placeholder=None):
    return html.Div(
        [
            html.Label([label, html.Span(f" ({unit})", className="unit") if unit else None]),
            dcc.Input(id=id_, type="number", placeholder=placeholder, className="form-control", debounce=True),
        ],
        className="cv-field",
    )


def dropdown_field(id_, label, options):
    return html.Div(
        [
            html.Label(label),
            dcc.Dropdown(id=id_, options=[{"label": o, "value": o} for o in options],
                         placeholder="Select", className="dash-dropdown",
                         searchable=False, clearable=False),
        ],
        className="cv-field",
    )


def text_field(id_, label, placeholder=None):
    return html.Div(
        [
            html.Label(label),
            dcc.Input(id=id_, type="text", placeholder=placeholder, className="form-control"),
        ],
        className="cv-field",
    )


form = html.Div(
    [
        html.Div(
            [
                html.Div("Vehicle", className="cv-group-title"),
                html.Div(
                    [
                        text_field("brand", "Brand", placeholder="Maruti, Hyundai..."),
                        field("year", "Year", placeholder="2015"),
                        field("km_driven", "Distance", unit="km", placeholder="50000"),
                    ],
                    className="cv-field-row",
                ),
            ],
            className="cv-group",
        ),
        html.Div(
            [
                html.Div("Engine & efficiency", className="cv-group-title"),
                html.Div(
                    [
                        field("mileage", "Mileage", unit="kmpl", placeholder="18.5"),
                        field("engine", "Engine", unit="CC", placeholder="1197"),
                        field("max_power", "Max power", unit="bhp", placeholder="82.0"),
                    ],
                    className="cv-field-row",
                ),
            ],
            className="cv-group",
        ),
        html.Div(
            [
                html.Div("Condition & sale", className="cv-group-title"),
                html.Div(
                    [
                        field("seats", "Seats", placeholder="5"),
                        dropdown_field("fuel", "Fuel type", ["Diesel", "Petrol"]),
                        dropdown_field("transmission", "Transmission", ["Manual", "Automatic"]),
                    ],
                    className="cv-field-row",
                ),
                html.Div(
                    [
                        dropdown_field("owner", "Owner status", OWNER_OPTIONS),
                        dropdown_field("seller_type", "Seller type", ["Individual", "Dealer", "Trustmark Dealer"]),
                    ],
                    className="cv-field-row",
                    style={"marginTop": "14px"},
                ),
            ],
            className="cv-group",
        ),
        html.Button("Predict price class", id="predict-btn", n_clicks=0, className="cv-btn"),
    ],
    className="cv-panel",
)

readout = html.Div(
    [
        html.Div(
            [html.I(className="fa-solid fa-flask", style={"fontSize": "0.75rem"}), " Classifier"],
            className="cv-badge",
        ),
        dcc.Loading(
            html.Div(
                html.Div(
                    "Fill in the spec sheet and predict the price class and the reading "
                    "will appear here.",
                    className="cv-readout-empty",
                ),
                id="prediction-output",
            ),
            type="dot",
        ),
        html.Div(
            f"Model source: {MODEL_SOURCE}",
            style={"color": "var(--paper-dim)", "fontSize": "0.78rem", "marginTop": "16px"},
        ),
    ],
    className="cv-readout",
)

layout = html.Div(
    [
        html.Div(
            [
                html.H1("Price class predictor"),
                html.P("Estimate which of the 4 price bands a used car falls into, from its specifications."),
            ],
            className="cv-hero",
        ),
        html.Div([form, readout], className="cv-grid"),
    ],
    className="cv-page",
)


def probability_bar(class_id, label, price_range, prob, is_top):
    return html.Div(
        [
            html.Div(
                [
                    html.Span(f"Class {class_id} · {label}", style={"fontWeight": 600 if is_top else 400}),
                    html.Span(f"{prob * 100:.1f}%", style={"fontVariantNumeric": "tabular-nums"}),
                ],
                style={"display": "flex", "justifyContent": "space-between", "marginBottom": "4px", "fontSize": "0.85rem"},
            ),
            html.Div(
                html.Div(className="cv-bar-fill" + (" cv-bar-top" if is_top else ""),
                         style={"width": f"{max(prob * 100, 1.5):.1f}%"}),
                className="cv-bar-track",
            ),
            html.Div(price_range, style={"color": "var(--paper-dim)", "fontSize": "0.75rem", "marginTop": "2px", "marginBottom": "14px"}),
        ]
    )


@dash.callback(
    Output("prediction-output", "children"),
    Input("predict-btn", "n_clicks"),
    State("year", "value"),
    State("km_driven", "value"),
    State("mileage", "value"),
    State("engine", "value"),
    State("max_power", "value"),
    State("seats", "value"),
    State("owner", "value"),
    State("fuel", "value"),
    State("transmission", "value"),
    State("seller_type", "value"),
    State("brand", "value"),
    prevent_initial_call=True,
)
def on_predict(n_clicks, year, km_driven, mileage, engine, max_power, seats,
               owner_label, fuel, transmission, seller_type, brand):
    owner_scale = mi.load_bundle()["owner_scale"]
    row = pd.DataFrame([{
        "year": year, "km_driven": km_driven, "mileage": mileage, "engine": engine,
        "max_power": max_power, "seats": seats,
        "owner": owner_scale.get(owner_label),
        "fuel": fuel, "transmission": transmission, "seller_type": seller_type, "brand": brand,
    }])

    try:
        preds, probs = PREDICT_FN(row)
        pred_class = int(preds[0])
        proba_row = probs[0]
    except Exception as exc:
        return html.Div(
            [html.I(className="fa-solid fa-triangle-exclamation", style={"marginRight": "8px"}),
             f"Could not predict a class: {exc}"],
            className="cv-alert-error",
        )

    bars = [
        probability_bar(c, CLASS_LABELS[c], CLASS_RANGES[c], proba_row[c], c == pred_class)
        for c in sorted(CLASS_LABELS.keys())
    ]

    return html.Div(
        [
            html.Div("Predicted class", className="cv-readout-label"),
            html.Div(f"{pred_class} · {CLASS_LABELS[pred_class]}", className="cv-readout-value", style={"fontSize": "1.9rem"}),
            html.Div(f"Estimated range: {CLASS_RANGES[pred_class]}",
                     style={"color": "var(--paper-dim)", "fontSize": "0.85rem", "marginBottom": "20px"}),
            html.Div(bars),
        ]
    )
