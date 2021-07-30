import plotly.graph_objects as go

radius=[3.5, 1.5, 2.5, 4.5, 4.5, 4, 3]
angle=[65, 15, 210, 110, 312.5, 180, 270]
breath=[20, 15, 10, 20, 15, 30, 15, ]

fig = go.Figure(go.Barpolar(
    r=radius,
    theta=angle,
    width=breath,
    marker_color=["#85EBFF", '#405CFF',"#85EBFF", '#405CFF', "#85EBFF", '#405CFF', '#B6FFB4'],
    marker_line_color="black",
    marker_line_width=2,
    opacity=0.8
))

fig.update_layout(
    template=None,
    polar=dict(
        radialaxis=dict(range=[0, 5], showticklabels=False, ticks=''),
        angularaxis=dict(showticklabels=False, ticks='')
    )
)

fig.show()
