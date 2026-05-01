import plotly.graph_objects as go
import numpy as np
current_fig = None

def build_sphere(r=1):
    phi = np.linspace(0, 2*np.pi, 80)
    theta = np.linspace(0, np.pi, 80)
    x = r * np.outer(np.cos(phi), np.sin(theta))
    y = r * np.outer(np.sin(phi), np.sin(theta))
    z = r * np.outer(np.ones(np.size(phi)), np.cos(theta))
    return x, y, z

def convert_to_polar(x,y,z):
    p = np.sqrt((x*x)+(y*y)+(z*z))
    theta = np.arctan2(y,x)
    phi = np.arccos(z / p)
    return p, theta, phi

def convert_to_cartesian(p, theta, phi):
    x = p * np.sin(phi)*np.cos(theta)
    y = p*np.sin(phi)*np.sin(theta)
    z = p * np.cos(phi)
    return x,y,z

def plot_sphere():
    global current_fig
    fig = go.Figure()
    u = np.linspace(0, 2 * np.pi, 40) 
    v = np.linspace(0, np.pi, 40)    
    x = np.outer(np.cos(u), np.sin(v))
    y = np.outer(np.sin(u), np.sin(v))
    z = np.outer(np.ones(np.size(u)), np.cos(v))

    fig.add_surface(x=x, y=y, z=z,opacity=0.14,showscale=False,colorscale=[[0, 'skyblue'], [1, 'skyblue']],surfacecolor=np.zeros_like(x))

    fig.add_surface(x=x, y=y, z=z,opacity=0.15,showscale=False,colorscale='RdPu',surfacecolor=z)

    fig.add_trace(go.Scatter3d(x=[0, 0], y=[0, 0], z=[-1, 1], mode='lines', line=dict(color='blue', width=5)))
    fig.add_trace(go.Scatter3d(x=[-1, 1], y=[0, 0], z=[0, 0], mode='lines', line=dict(color='red', width=5)))
    fig.add_trace(go.Scatter3d(x=[0, 0], y=[-1, 1], z=[0, 0], mode='lines', line=dict(color='green', width=5)))

    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[1.18], mode='text', text=['|0>'], textfont=dict(size=14)))
    fig.add_trace(go.Scatter3d(x=[0], y=[0], z=[-1.25], mode='text', text=['|1>'], textfont=dict(size=14)))
    fig.add_trace(go.Scatter3d(x=[1.18], y=[0], z=[0], mode='text', text=['|+>'], textfont=dict(size=14)))
    fig.add_trace(go.Scatter3d(x=[-1.28], y=[0], z=[0], mode='text', text=['|->'], textfont=dict(size=14)))
    fig.add_trace(go.Scatter3d(x=[0], y=[1.18], z=[0], mode='text', text=['|+i>'], textfont=dict(size=14)))
    fig.add_trace(go.Scatter3d(x=[0], y=[-1.32], z=[0], mode='text', text=['|-i>'], textfont=dict(size=14)))

    fig.update_layout(scene=dict(
                        xaxis=dict(visible=False),
                        yaxis=dict(visible=False),
                        zaxis=dict(visible=False),
                        aspectmode='cube',
                        camera=dict(eye=dict(x=1.55, y=1.1, z=0.85))),
                    showlegend=False,
                    margin=dict(l=0, r=0, b=0, t=0),
                    uirevision='constant')

    current_fig = fig
    return fig

def plot_vector(theta, phi, ax, color='black', linewidth=2.5):
    x,y,z = convert_to_cartesian(1,theta,phi)
    x0, y0, z0 = convert_to_cartesian(0, 0, 0)
    ax.add_trace(go.Scatter3d(x=[x0, x], y=[y0, y], z=[z0, z],mode='lines',line=dict(color=color, width=linewidth)))
    return ax

def plot_state(state, ax, color="blue"):
    from qc_logic import state_to_polar
    theta, phi = state_to_polar(state)
    plot_vector(phi, theta, ax, color=color)

def draw_state_arc(state1, state2, fig):
    from qc_logic import state_to_cartesian
    x1, y1, z1 = state_to_cartesian(state1)
    x2, y2, z2 = state_to_cartesian(state2)
    draw_arc(x1, y1, z1, x2, y2, z2, fig)

def draw_arc(x1, y1, z1, x2, y2, z2, fig):
  current_fig=fig
  start = np.array([x1, y1, z1], dtype=float)
  end = np.array([x2, y2, z2], dtype=float)

  start = start / np.linalg.norm(start)
  end = end / np.linalg.norm(end)

  dot = np.clip(np.dot(start, end), -1.0, 1.0)
  omega = np.arccos(dot)
  t_vals = np.linspace(0.0, 1.0, 60)

  arc_points = []
  if np.isclose(omega, 0.0):
    for _ in t_vals:
      arc_points.append(start.copy())
  else:
    sin_omega = np.sin(omega)
    for t in t_vals:
      w1 = np.sin((1.0 - t) * omega) / sin_omega
      w2 = np.sin(t * omega) / sin_omega
      point = w1 * start + w2 * end
      arc_points.append(point)

  arc = np.array(arc_points)

  ax = current_fig if current_fig is not None else go.Figure()
  ax.add_trace(go.Scatter3d(
      x=arc[:, 0], y=arc[:, 1], z=arc[:, 2],
      mode='lines',
      line=dict(color='purple', width=4, dash='dash')))
  current_fig = ax
  return ax

