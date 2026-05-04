import numpy as np
import plotly.graph_objects as go
from dash import Dash, dcc, html, ctx, ALL
from dash import Input, Output, State

from qc_logic import get_quantum_state, ket_0, get_probs, get_transformation_matrix, H, PX, PY, PZ, apply_gate, state_to_cartesian
from visualization import draw_state_arc, plot_sphere, plot_state

base_fig = plot_sphere()

app = Dash(__name__)

# ── Constants ────────────────────────────────────────────────────────────────

MAX_GATES = 8

GATE_MAP = {'H': H, 'X': PX, 'Y': PY, 'Z': PZ}

# One distinct colour per circuit step (initial + up to MAX_GATES results)
STEP_COLORS = [
    '#1d4ed8',  # 0 – initial state      (blue)
    '#ea580c',  # 1 – after gate 1       (orange)
    '#16a34a',  # 2 – after gate 2       (green)
    '#9333ea',  # 3 – after gate 3       (purple)
    '#dc2626',  # 4 – after gate 4       (red)
    '#0891b2',  # 5 – after gate 5       (cyan)
    '#b45309',  # 6 – after gate 6       (amber)
    '#be185d',  # 7 – after gate 7       (pink)
    '#065f46',  # 8 – after gate 8       (teal)
]

# ── Helpers ──────────────────────────────────────────────────────────────────

def latex_complex(z):
    real = round(float(np.real(z)), 3)
    imag = round(float(np.imag(z)), 3)
    if imag == 0:
        return f"{real:.3f}"
    sign = "+" if imag >= 0 else "-"
    return f"{real:.3f} {sign} {abs(imag):.3f}\\mathrm{{i}}"


def build_math_text(gates, final_state, probs):
    """Render the gate equation and measurement probabilities as LaTeX."""
    s0 = latex_complex(final_state[0])
    s1 = latex_complex(final_state[1])
    p0 = float(probs[0])
    p1 = float(probs[1])

    if not gates:
        eq = (
            f"\\lvert\\psi\\rangle = \\begin{{bmatrix}}{s0} \\\\ {s1}\\end{{bmatrix}}"
        )
    else:
        # Matrix product reads right-to-left, so reverse the list
        gate_str = "\\,".join(reversed(gates))
        eq = (
            f"{gate_str}\\,\\lvert\\psi\\rangle = "
            f"\\begin{{bmatrix}}{s0} \\\\ {s1}\\end{{bmatrix}}"
        )

    return (
        "$$\\Large \\begin{gathered}"
        + eq
        + f"\\\\[1em] P(|0\\rangle) = {p0:.3f},\\quad P(|1\\rangle) = {p1:.3f}"
        + "\\end{gathered}$$"
    )


def build_circuit_display(gates):
    """Return a list of Dash children representing the gate circuit strip."""
    items = []

    # Initial state label
    items.append(html.Span(
        "|ψ₀⟩",
        style={
            'fontSize': '1.05em', 'fontWeight': 'bold',
            'color': STEP_COLORS[0], 'padding': '4px 2px',
        }
    ))

    for i, gate in enumerate(gates):
        # Arrow between steps
        items.append(html.Span("→", style={
            'color': '#9ca3af', 'margin': '0 5px', 'fontSize': '1.1em',
        }))

        chip_color = STEP_COLORS[(i + 1) % len(STEP_COLORS)]
        items.append(html.Span(
            [
                html.Span(gate, style={'fontWeight': 'bold', 'marginRight': '5px'}),
                html.Button(
                    "×",
                    id={'type': 'remove-gate', 'index': i},
                    n_clicks=0,
                    title=f"Remove {gate} (step {i + 1})",
                    style={
                        'border': 'none',
                        'background': 'rgba(255,255,255,0.25)',
                        'color': 'white', 'borderRadius': '3px',
                        'cursor': 'pointer', 'fontSize': '0.85em',
                        'padding': '0 3px', 'lineHeight': '1.3',
                        'fontWeight': 'bold',
                    },
                ),
            ],
            style={
                'display': 'inline-flex', 'alignItems': 'center',
                'backgroundColor': chip_color, 'color': 'white',
                'borderRadius': '6px', 'padding': '4px 8px',
                'fontSize': '0.9em', 'userSelect': 'none',
            }
        ))

    if gates:
        final_color = STEP_COLORS[min(len(gates), len(STEP_COLORS) - 1)]
        items.append(html.Span("→", style={
            'color': '#9ca3af', 'margin': '0 5px', 'fontSize': '1.1em',
        }))
        items.append(html.Span(
            "|ψ_f⟩",
            style={'fontSize': '1.05em', 'fontWeight': 'bold', 'color': final_color}
        ))
    else:
        items.append(html.Span(
            " — add gates above to build a circuit",
            style={
                'color': '#9ca3af', 'fontSize': '0.88em',
                'fontStyle': 'italic', 'marginLeft': '6px',
            }
        ))

    return items


def build_matrix_text(theta, phi):
    """Render U(θ,φ) as a LaTeX matrix."""
    U = get_transformation_matrix(theta, phi)
    u00 = latex_complex(U[0, 0])
    u01 = latex_complex(U[0, 1])
    u10 = latex_complex(U[1, 0])
    u11 = latex_complex(U[1, 1])
    return (
        "$$\\large U(\\theta,\\phi) = \\begin{bmatrix}"
        + f"{u00} & {u01} \\\\\\\\ {u10} & {u11}"
        + "\\end{bmatrix}$$"
    )


def gate_btn_style(color, disabled=False):
    base = {
        'padding': '7px 20px', 'borderRadius': '6px',
        'border': 'none', 'color': 'white',
        'fontWeight': 'bold', 'cursor': 'pointer' if not disabled else 'not-allowed',
        'fontSize': '1em', 'boxShadow': '0 1px 3px rgba(0,0,0,0.15)',
        'backgroundColor': color if not disabled else '#94a3b8',
        'opacity': '1' if not disabled else '0.55',
    }
    return base


# ── Layout ───────────────────────────────────────────────────────────────────

app.layout = html.Div([
    dcc.Store(id='gate-circuit', data={'gates': []}),

    html.H2(
        "Bloch Sphere Quantum Gate Simulator",
        style={
            'textAlign': 'center', 'marginBottom': '4px',
            'marginTop': '12px', 'color': '#1e293b',
        }
    ),

    # ── Main row: left controls + right sphere ──────────────────────────────
    html.Div(style={
        'display': 'flex', 'width': '96%', 'margin': '0 auto',
        'alignItems': 'stretch', 'gap': '2%',
    }, children=[

        # Left panel – math display + sliders
        html.Div(style={
            'width': '34%', 'height': '80vh', 'display': 'flex',
            'flexDirection': 'column', 'alignItems': 'center',
            'justifyContent': 'center', 'paddingLeft': '28px',
        }, children=[
            html.Div(
                dcc.Markdown(id='math-display', mathjax=True,
                             style={'textAlign': 'center'}),
                style={
                    'width': '100%', 'padding': '8px 0 10px 0',
                    'display': 'flex', 'justifyContent': 'center',
                    'alignItems': 'center',
                }
            ),
            html.Div(
                dcc.Markdown(id='matrix-display', mathjax=True,
                             style={'textAlign': 'center', 'marginBottom': '10px'}),
                style={'width': '100%', 'display': 'flex', 'justifyContent': 'center'},
            ),
            html.Div(style={
                'width': '68%', 'display': 'flex',
                'flexDirection': 'column', 'gap': '18px', 'marginTop': '8px',
            }, children=[
                html.Div([
                    dcc.Markdown("$\\theta$", mathjax=True, style={
                        'fontSize': '1em', 'width': '34px',
                        'textAlign': 'center', 'marginRight': '10px',
                        'lineHeight': '1',
                    }),
                    html.Div(
                        dcc.Slider(0, 360, value=45, id='theta',
                                   vertical=False, updatemode='mouseup'),
                        style={'flex': '1'},
                    ),
                ], style={
                    'display': 'flex', 'alignItems': 'center',
                    'transform': 'scale(1.15)', 'transformOrigin': 'center',
                }),
                html.Div([
                    dcc.Markdown("$\\phi$", mathjax=True, style={
                        'fontSize': '1em', 'width': '34px',
                        'textAlign': 'center', 'marginRight': '10px',
                        'lineHeight': '1',
                    }),
                    html.Div(
                        dcc.Slider(0, 360, value=45, id='phi',
                                   vertical=False, updatemode='mouseup'),
                        style={'flex': '1'},
                    ),
                ], style={
                    'display': 'flex', 'alignItems': 'center',
                    'transform': 'scale(1.15)', 'transformOrigin': 'center',
                }),
            ]),
        ]),

        # Right panel – Bloch sphere
        html.Div([
            dcc.Graph(id='graph', figure=go.Figure(base_fig),
                      style={"height": "80vh"}),
        ], style={'width': '64%'}),
    ]),

    # ── Gate circuit panel ─────────────────────────────────────────────────
    html.Div(style={
        'width': '60%', 'margin': '14px auto 6px auto',
        'border': '1.5px solid #e2e8f0', 'borderRadius': '14px',
        'padding': '24px 36px', 'background': '#f8fafc',
        'boxShadow': '0 2px 8px rgba(0,0,0,0.07)',
    }, children=[

        # Header row
        html.Div(style={
            'display': 'flex', 'justifyContent': 'space-between',
            'alignItems': 'center', 'marginBottom': '14px',
        }, children=[
            html.Span("Build Gate Circuit", style={
                'fontWeight': '600', 'fontSize': '1.05em', 'color': '#334155',
            }),
            html.Button(
                "↺  Reset",
                id='btn-reset', n_clicks=0,
                title="Clear all gates and start over",
                style={
                    'padding': '6px 18px', 'borderRadius': '6px',
                    'border': '1.5px solid #ef4444', 'color': '#ef4444',
                    'background': 'white', 'cursor': 'pointer',
                    'fontWeight': 'bold', 'fontSize': '0.95em',
                }
            ),
        ]),

        # Add-gate buttons
        html.Div(style={
            'display': 'flex', 'gap': '10px',
            'marginBottom': '16px', 'flexWrap': 'wrap',
        }, children=[
            html.Button(
                "+ H", id='btn-add-h', n_clicks=0,
                title="Hadamard gate – creates superposition",
                style=gate_btn_style(STEP_COLORS[1]),
            ),
            html.Button(
                "+ X", id='btn-add-x', n_clicks=0,
                title="Pauli-X gate – bit flip (NOT gate)",
                style=gate_btn_style(STEP_COLORS[2]),
            ),
            html.Button(
                "+ Y", id='btn-add-y', n_clicks=0,
                title="Pauli-Y gate – bit+phase flip",
                style=gate_btn_style(STEP_COLORS[3]),
            ),
            html.Button(
                "+ Z", id='btn-add-z', n_clicks=0,
                title="Pauli-Z gate – phase flip",
                style=gate_btn_style(STEP_COLORS[4]),
            ),
            html.Span(
                id='gate-limit-msg',
                style={
                    'alignSelf': 'center', 'fontSize': '0.85em',
                    'color': '#64748b', 'fontStyle': 'italic',
                }
            ),
        ]),

        # Circuit strip
        html.Div(
            id='circuit-display',
            style={
                'display': 'flex', 'alignItems': 'center',
                'flexWrap': 'wrap', 'gap': '4px',
                'minHeight': '36px', 'maxHeight': '44px',
                'padding': '4px 12px',
                'background': 'white', 'borderRadius': '8px',
                'border': '1px solid #e2e8f0', 'overflowX': 'auto',
            }
        ),
    ]),

    # ── State equation ─────────────────────────────────────────────────────
    dcc.Markdown(
        r"$$\large \lvert\psi\rangle = "
        r"\cos\!\left(\tfrac{\theta}{2}\right)\lvert 0\rangle + "
        r"e^{i\phi}\sin\!\left(\tfrac{\theta}{2}\right)\lvert 1\rangle$$",
        mathjax=True,
        style={'textAlign': 'center', 'padding': '10px 0 4px 0'},
    ),

    # ── Gate reference matrices ────────────────────────────────────────────
    html.Div(
        style={
            'display': 'flex', 'justifyContent': 'center',
            'gap': '30px', 'padding': '8px 0 18px 0',
        },
        children=[
            html.Div([
                dcc.Markdown(
                    r"$$H = \frac{1}{\sqrt{2}}\begin{bmatrix}1 & 1\\1 & -1\end{bmatrix}$$",
                    mathjax=True, style={'textAlign': 'center'},
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(
                    r"$$X = \begin{bmatrix}0 & 1\\1 & 0\end{bmatrix}$$",
                    mathjax=True, style={'textAlign': 'center'},
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(
                    r"$$Y = \begin{bmatrix}0 & -i\\i & 0\end{bmatrix}$$",
                    mathjax=True, style={'textAlign': 'center'},
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
            html.Div([
                dcc.Markdown(
                    r"$$Z = \begin{bmatrix}1 & 0\\0 & -1\end{bmatrix}$$",
                    mathjax=True, style={'textAlign': 'center'},
                ),
            ], style={'display': 'flex', 'flexDirection': 'column', 'alignItems': 'center'}),
        ]
    ),
])


# ── Callbacks ────────────────────────────────────────────────────────────────

@app.callback(
    Output('gate-circuit', 'data'),
    Input('btn-add-h', 'n_clicks'),
    Input('btn-add-x', 'n_clicks'),
    Input('btn-add-y', 'n_clicks'),
    Input('btn-add-z', 'n_clicks'),
    Input('btn-reset', 'n_clicks'),
    Input({'type': 'remove-gate', 'index': ALL}, 'n_clicks'),
    State('gate-circuit', 'data'),
    prevent_initial_call=True,
)
def update_circuit(h, x, y, z, reset, remove_clicks, circuit_data):
    """Add, remove, or clear gates in the circuit store."""
    gates = list(circuit_data.get('gates', []))
    triggered = ctx.triggered_id

    if triggered == 'btn-add-h' and len(gates) < MAX_GATES:
        gates.append('H')
    elif triggered == 'btn-add-x' and len(gates) < MAX_GATES:
        gates.append('X')
    elif triggered == 'btn-add-y' and len(gates) < MAX_GATES:
        gates.append('Y')
    elif triggered == 'btn-add-z' and len(gates) < MAX_GATES:
        gates.append('Z')
    elif triggered == 'btn-reset':
        gates = []
    elif isinstance(triggered, dict) and triggered.get('type') == 'remove-gate':
        idx = triggered['index']
        if 0 <= idx < len(gates):
            gates.pop(idx)

    return {'gates': gates}


@app.callback(
    Output('graph', 'figure'),
    Output('math-display', 'children'),
    Output('matrix-display', 'children'),
    Output('circuit-display', 'children'),
    Output('gate-limit-msg', 'children'),
    Input('theta', 'value'),
    Input('phi', 'value'),
    Input('gate-circuit', 'data'),
)
def update_bloch(theta_deg, phi_deg, circuit_data):
    """Re-render the Bloch sphere, math panel, and circuit strip."""
    theta = np.deg2rad(theta_deg)
    phi = np.deg2rad(phi_deg)
    gates = circuit_data.get('gates', [])

    # Build the full state sequence: initial state + one per gate
    initial_state = get_quantum_state(theta, phi)
    states = [initial_state]
    for gate_sym in gates:
        states.append(apply_gate(GATE_MAP[gate_sym], states[-1]))

    final_state = states[-1]
    probs = get_probs(final_state)

    # ── Figure ──────────────────────────────────────────────────────────────
    new_fig = go.Figure(base_fig)

    # Arc from |0⟩ to initial state
    draw_state_arc(ket_0, initial_state, new_fig)

    for i, state in enumerate(states):
        color = STEP_COLORS[i % len(STEP_COLORS)]
        plot_state(state, new_fig, color=color)

        # Marker dot at vector tip for hover labels
        sx, sy, sz = state_to_cartesian(state)
        if i == 0:
            hover_label = "|ψ₀⟩ (initial)"
        else:
            hover_label = "After " + " → ".join(gates[:i])
        new_fig.add_trace(go.Scatter3d(
            x=[sx], y=[sy], z=[sz],
            mode='markers',
            marker=dict(size=6, color=color, symbol='circle'),
            hovertemplate=(
                f"<b>{hover_label}</b><br>"
                f"x={sx:.3f}  y={sy:.3f}  z={sz:.3f}"
                "<extra></extra>"
            ),
            showlegend=False,
        ))

        # Arc between consecutive states
        if i > 0:
            draw_state_arc(states[i - 1], state, new_fig)

    # ── Math display ─────────────────────────────────────────────────────────
    math_text = build_math_text(gates, final_state, probs)

    # ── Transformation matrix (U from sliders) ───────────────────────────────
    matrix_text = build_matrix_text(theta, phi)

    # ── Circuit strip ────────────────────────────────────────────────────────
    circuit_children = build_circuit_display(gates)

    # ── Gate-limit hint ──────────────────────────────────────────────────────
    limit_msg = f"Maximum {MAX_GATES} gates reached." if len(gates) >= MAX_GATES else ""

    return new_fig, math_text, matrix_text, circuit_children, limit_msg


if __name__ == "__main__":
    app.run(debug=True)
