import numpy as np
import plotly.graph_objs as go

fig = go.Figure()
fig.add_scatter(x=np.random.rand(100), y=np.random.rand(100), mode='markers',
                marker={'size': 30, 'color': np.random.rand(100), 'opacity': 0.6,
                        'colorscale': 'Viridis'});


def show(fig):
    import io
    import plotly.io as pio
    from PIL import Image
    buf = io.BytesIO()
    pio.write_image(fig, buf)
    img = Image.open(buf)
    img.show()


show(fig)
