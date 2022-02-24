import numpy as np
import matplotlib.pyplot as plt
import plotly.graph_objects as go


def calc_force(r):
    """
    Calculate force based on distance

    Parameters
    ----------
    r : float
        distance between particles.

    Returns
    -------
    float
    """
    return 24*(2*np.power(r, -13) - np.power(r, -7))


def calc_position(r_prev, p_prev, dt):
    """
    Calculate new position

    Parameters
    ----------
    r_prev : float
        previous particle distance.
    p_prev : float
        previous particle momentum.
    dt : float
        time step.

    Returns
    -------
    float
        new particle position.
    """
    return r_prev + p_prev*dt + 0.5*calc_force(r_prev)*dt**2


def calc_momentum(p_prev, f, f_prev, dt):
    """
    Calculate new momentum

    Parameters
    ----------
    p_prev : float
        previous particle momentum.
    f : float
        current force on particle.
    f_prev : float
        previous force on particle.
    dt : float
        time step.

    Returns
    -------
    float
        new particle momentm.
    """
    return p_prev + 0.5*(f + f_prev)*dt


def plot_data(x, y, x_axis_title='', y_axis_title='', title=''):
    """
    Plot data
    """
    fig = go.Figure(data=go.Scatter(
        x=x,
        y=y,
        mode='lines'
        ))

    fig.update_layout(
    title={
        'text': title,
        'x':0.5,
        'xanchor': 'center',
        'yanchor': 'top'})
    fig.update_xaxes(title_text=x_axis_title)
    fig.update_yaxes(title_text=y_axis_title)

    fig.show()


# Parameters
dt = 0.01
n_final = 1000
r = np.zeros(n_final)
p = np.zeros(n_final)
F = np.zeros(n_final)

# Initial Conditions
t0 = 0.0
t = np.linspace(t0, dt*(n_final-1), n_final)
r[0] = 1.3
p[0] = 0.0
F[0] = calc_force(r[0])

for n in range(1, n_final):
    r[n] = calc_position(r[n-1], p[n-1], dt)
    F[n] = calc_force(r[n])
    p[n] = calc_momentum(p[n-1], F[n], F[n-1], dt)

K = 0.5*np.power(p, 2)
U = 4*(np.power(r, -12) - np.power(r, -6))
E = K + U

K_avg = np.average(K)
U_avg = np.average(U)

print(K_avg)
print(U_avg)

# Can make dt and n_final adjustable
# Should be able enter any value for r0, p0

plot_data(t, r, x_axis_title='time', y_axis_title='position', title='Position vs. Time')
# plot_data(t, p, x_axis_title='time', y_axis_title='momenum')
# plot_data(t, F, x_axis_title='time', y_axis_title='force')
# plot_data(t, K, x_axis_title='time', y_axis_title='force')
# plot_data(t, U, x_axis_title='time', y_axis_title='force')
# plot_data(t, E, x_axis_title='time', y_axis_title='force')
# plot_data(r, p, x_axis_title='position', y_axis_title='momentum')
