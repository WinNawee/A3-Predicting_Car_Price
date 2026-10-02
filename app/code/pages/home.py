"""
Home page — the three models side by side.

A1 and A2 predict a price (regression); A3 predicts which of four price
bands a car falls into (classification), so their scores aren't directly
comparable — each card shows the metric that fits its task.
"""

import dash
from dash import html, dcc
import model_inference as mi

dash.register_page(__name__, path="/", name="Home")

bundle = mi.load_bundle()
CLASS_LABELS = bundle["class_labels"]
CLASS_RANGES = bundle["class_price_ranges"]

hero = html.Div(
    [
        html.H1("CarValuate"),
        html.P(
            "Three ways to value a used car from the same dataset: a scikit-learn "
            "pipeline from Assignment 1, a linear regression written from scratch for "
            "Assignment 2, and new in Assignment 3 (a multinomial logistic regression), "
            "also written from scratch, that predicts which of four price bands a car "
            "falls into instead of an exact price."
        ),
    ],
    className="cv-hero",
)


def model_card(title, subtitle, value, value_label, href, link_text, accent=False):
    return html.Div(
        [
            html.Div(title, className="cv-group-title"),
            html.Div(subtitle, style={"color": "var(--paper-dim)", "fontSize": "0.85rem", "marginBottom": "16px"}),
            html.Div(value, className="cv-readout-value",
                     style={"fontSize": "1.9rem", "color": "var(--brass)" if accent else "var(--paper)"}),
            html.Div(value_label, className="cv-readout-label"),
            dcc.Link(link_text, href=href, className="cv-nav-link",
                     style={"display": "inline-block", "marginTop": "16px", "padding": "8px 0",
                            **({"color": "var(--brass)"} if accent else {})}),
        ],
        className="cv-panel",
    )


model_cards = html.Div(
    [
        model_card("A1 model", "scikit-learn pipeline · regression", "~0.92", "Test R²",
                   "/old", "Open A1 model →"),
        model_card("A2 model", "from-scratch linear regression", "0.875", "Test R²",
                   "/new", "Open A2 model →"),
        model_card("A3 classifier", "from-scratch logistic regression · 4 classes", "0.730",
                   "Test accuracy (macro F1 0.726)", "/predict", "Open A3 classifier →", accent=True),
    ],
    className="cv-grid-3",
    style={"marginBottom": "32px"},
)

class_cards = html.Div(
    [
        html.Div("A3 price classes", className="cv-group-title"),
        html.Div(
            [
                html.Div(
                    [
                        html.Div(f"Class {c}", style={"color": "var(--paper-dim)", "fontSize": "0.8rem"}),
                        html.Div(CLASS_LABELS[c], style={"fontSize": "1.15rem", "fontWeight": 700,
                                                          "color": "var(--paper)", "margin": "4px 0"}),
                        html.Div(CLASS_RANGES[c], style={"color": "var(--paper-dim)", "fontSize": "0.85rem"}),
                    ],
                    style={"background": "var(--panel-raised)", "border": "1px solid var(--line)",
                           "borderRadius": "8px", "padding": "16px"},
                )
                for c in sorted(CLASS_LABELS.keys())
            ],
            className="cv-grid-4",
        ),
    ],
    className="cv-panel",
    style={"marginBottom": "32px"},
)

explanation = html.Div(
    [
        html.Div("Which one should you use?", className="cv-group-title"),
        html.P(
            "If you need a single price estimate, the A1 model is the most accurate "
            "(R² ≈ 0.92 vs. 0.875 for A2): it's a scikit-learn ensemble, which captures "
            "non-linear patterns a plain linear model can't. A2's value is transparency ("
            "every coefficient is inspectable, and it was tuned through a fully tracked "
            "cross-validated search in MLflow.)",
            style={"color": "var(--paper-dim)", "lineHeight": "1.7", "marginBottom": "14px"},
        ),
        html.P(
            "The A3 classifier answers a price band and shows "
            "how confident it is in each band. It was selected on a held-out validation "
            "set, with duplicate listings removed before splitting, and the test set was "
            "scored exactly once, so its 0.730 accuracy is an honest estimate. Every "
            "precision / recall / F1 metric behind it is implemented from scratch and "
            "cross-checked against scikit-learn.",
            style={"color": "var(--paper-dim)", "lineHeight": "1.7"},
        ),
    ],
    className="cv-panel",
)

layout = html.Div([hero, model_cards, class_cards, explanation], className="cv-page")
