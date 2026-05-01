import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, ctx
from dash import Input, Output, State

from qc_logic import get_quantum_state, ket_0, get_transformation_matrix, get_probs, H, PX, PY, PZ, apply_gate
from visualization import draw_state_arc, plot_sphere, plot_state

base_fig = plot_sphere()
fig = go.Figure(base_fig)

app = Dash(__name__)


def latex_complex(z):
    real = round(float(np.real(z)), 3)
    imag = round(float(np.imag(z)), 3)
    if imag == 0:
        return f"{real:.3f}"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.3f} {sign} {abs(imag):.3f}\\mathrm{{i}}"


def build_math_text(U, gate_symbol, state, probs):
    u00 = latex_complex(U[0, 0])
    u01 = latex_complex(U[0, 1])
    u10 = latex_complex(U[1, 0])
    u11 = latex_complex(U[1, 1])
    s0 = latex_complex(state[0])
    s1 = latex_complex(state[1])
    p0 = float(probs[0])
    p1 = float(probs[1])
    return (
        "$$\\LARGE \\begin{gathered}"
        + "U(\\theta,\\phi) = \\begin{bmatrix}"
        + f"{u00} & {u01} \\\\ {u10} & {u11}"
        + "\\end{bmatrix} \\\\[1em]"
        + f"{gate_symbol}\\,U(\\theta,\\phi)\\lvert 0 \\rangle = \\begin{{bmatrix}}"
        + f"{s0} \\\\ {s1}"
        + "\\end{bmatrix} \\\\[2em]"
        + f"P(0) = {p0:.3f},\\quad P(1) = {p1:.3f}"
        + "\\end{gathered}$$"
    )

app.layout = html.Div([
    dcc.Store(id='selected-gate', data='I'),
    html.H2(""),
    html.Div(style={'display': 'flex', 'width': '96%', 'margin': '0 auto', 'alignItems': 'stretch', 'gap': '2%'}, children=[
        html.Div(style={'width': '34%', 'height': '80vh', 'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center', 'justifyContent': 'center', 'paddingLeft': '28px'}, children=[
            html.Div(
                dcc.Markdown(id='math-display', mathjax=True, style={'textAlign': 'center'}),
                style={
                    'width': '100%',
                    'padding': '8px 0 10px 0',
                    'display': 'flex',
                    'justifyContent': 'center',
                    'alignItems': 'center'
                }
            ),
            html.Div(style={'width': '68%', 'display': 'flex', 'flexDirection': 'column', 'gap': '18px', 'marginTop': '8px'}, children=[
                html.Div([
                    dcc.Markdown("$\\theta$", mathjax=True, style={'fontSize': '1em', 'width': '34px', 'textAlign': 'center', 'marginRight': '10px', 'lineHeight': '1'}),
                    html.Div(dcc.Slider(0, 360, value=45, id='theta', vertical=False, updatemode='mouseup'), style={'flex': '1'})
                ], style={'display': 'flex', 'alignItems': 'center', 'transform': 'scale(1.15)', 'transformOrigin': 'center'}),
                html.Div([
                    dcc.Markdown("$\\phi$", mathjax=True, style={'fontSize': '1em', 'width': '34px', 'textAlign': 'center', 'marginRight': '10px', 'lineHeight': '1'}),
                    html.Div(dcc.Slider(0, 360, value=45, id='phi', vertical=False, updatemode='mouseup'), style={'flex': '1'})
                ], style={'display': 'flex', 'alignItems': 'center', 'transform': 'scale(1.15)', 'transformOrigin': 'center'})
            ])
        ]),
        html.Div([dcc.Graph(id='graph', figure=fig, style={"height": "80vh"})], style={'width': '64%'})
    ]),
    dcc.Markdown(
        r"""
        $$\LARGE \lvert\psi\rangle = a\lvert 0\rangle + b\lvert 1\rangle$$

        $$\LARGE \lvert a\rvert^2 + \lvert b\rvert^2 = 1$$

        $$\LARGE \lvert\psi\rangle = \cos\theta\,\lvert 0\rangle + e^{i\phi}\sin\theta\,\lvert 1\rangle$$
        """,
        mathjax=True,
        style={'textAlign': 'center', 'padding': '16px 0 6px 0', 'lineHeight': '1.35'}
    ),
    html.Div(
        style={'display': 'flex', 'justifyContent': 'center', 'gap': '26px', 'padding': '22px 0 10px 0'},
        children=[
            html.Div([
                dcc.Markdown(r"$$\LARGE H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}$$", mathjax=True,
                             style={'textAlign': 'center'}),
                html.Button('Apply H', id='btn-h', n_clicks=0, style={'marginTop': '8px', 'padding': '8px 14px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(r"$$\LARGE X = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}$$", mathjax=True,
                             style={'textAlign': 'center'}),
                html.Button('Apply X', id='btn-x', n_clicks=0, style={'marginTop': '8px', 'padding': '8px 14px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(r"$$\LARGE Y = \begin{bmatrix}0 & -i\\i & 0\end{bmatrix}$$", mathjax=True,
                             style={'textAlign': 'center'}),
                html.Button('Apply Y', id='btn-y', n_clicks=0, style={'marginTop': '8px', 'padding': '8px 14px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(r"$$\LARGE Z = \begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}$$", mathjax=True,
                             style={'textAlign': 'center'}),
                html.Button('Apply Z', id='btn-z', n_clicks=0, style={'marginTop': '8px', 'padding': '8px 14px'})
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'})
        ]
    )
        ])

@app.callback(
    Output('graph', 'figure'),
    Output('math-display', 'children'),
    Output('selected-gate', 'data'),
    Input('theta', 'value'),
    Input('phi', 'value'),
    Input('btn-h', 'n_clicks'),
    Input('btn-x', 'n_clicks'),
    Input('btn-y', 'n_clicks'),
    Input('btn-z', 'n_clicks'),
    State('selected-gate', 'data')
)
def update_bloch(theta_deg, phi_deg, h_clicks, x_clicks, y_clicks, z_clicks, stored_gate):

    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    state = get_quantum_state(theta, phi)
    transformation_matrix = get_transformation_matrix(theta, phi)

    gate_symbol = stored_gate if stored_gate in {'I', 'H', 'X', 'Y', 'Z'} else 'I'
    if ctx.triggered_id == 'btn-h':
        gate_symbol = 'H'
    elif ctx.triggered_id == 'btn-x':
        gate_symbol = 'X'
    elif ctx.triggered_id == 'btn-y':
        gate_symbol = 'Y'
    elif ctx.triggered_id == 'btn-z':
        gate_symbol = 'Z'

    selected_gate = {
        'I': np.eye(2, dtype=complex),
        'H': H,
        'X': PX,
        'Y': PY,
        'Z': PZ,
    }[gate_symbol]

    state2 = apply_gate(selected_gate, state)

    probs = get_probs(state2)
    new_fig = go.Figure(base_fig)
    plot_state(state, new_fig)
    draw_state_arc(ket_0, state, new_fig)
    plot_state(state2, new_fig, color="red")
    draw_state_arc(state, state2, new_fig)

    math_text = build_math_text(transformation_matrix, gate_symbol, state2, probs)
    return new_fig, math_text, gate_symbol

if __name__ == "__main__":
    app.run(debug=True)
