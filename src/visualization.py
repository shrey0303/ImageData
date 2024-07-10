import plotly.io as pio
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import math

pio.renderers.default = "notebook"


def plot_digits(data_disp: list, nb: int) -> None:
    """
    data_disp : a list of digit images represented as arrays
    nb : is such that nb**2 = N = number of images in total
    """
    fig = make_subplots(rows=nb, cols=nb)
    traces = list(
        map(lambda i: go.Heatmap(z=data_disp[0][i]), range(len(data_disp[0])))
    )

    for i in range(len(data_disp[0])):
        fig.add_trace(traces[i], row=math.floor(i / nb) + 1, col=i % nb + 1)
    fig.update_layout(height=600, width=600, title_text="Sample of digit images")
    fig.show()
