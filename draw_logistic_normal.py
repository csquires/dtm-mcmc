import numpy as np
from plotly.offline import init_notebook_mode, plot


def logistic(vec):
    return np.exp(vec) / sum(np.exp(vec))


SIGMA0 = 1
SIGMA = .1
T = 10000

ns = []
n = np.random.normal(0, SIGMA0, size=3)
ns.append(n)
for i in range(T):
    n = n + np.random.normal(0, SIGMA, size=3)
    ns.append(n)
ps = [logistic(n) for n in ns[0:T:100]]

PLOT = True
if PLOT:
    init_notebook_mode(connected=True)
    marker = {'color': 'red', 'size': 20}
    frames = [{'data': [{
        'x': [p[0]],
        'y': [p[1]],
        'mode': 'markers',
        'marker': marker
    }, {'x': [1, 0], 'y': [0, 1]}]} for p in ps]
    data = frames[0]['data']
    figure = {
        'data': data,
        'layout': {
            'xaxis': {'range': [0, 1], 'autorange': False},
            'yaxis': {'range': [0, 1], 'autorange': False},
        },
        'frames': frames[1:]
    }
    plot(figure)
