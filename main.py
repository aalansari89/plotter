import dash
from dash import html, dcc, Input, Output, State
import dash_bootstrap_components as dbc
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from scipy.interpolate import griddata
import io
import base64
import plotly.express as px

# Import colorscale list
colorscales = px.colors.named_colorscales()

# Setup chart settings form
chart_settings = html.Div(
    [
        html.H2("Chart Controls"),
        html.Hr(),
        html.H4("Chart Format", style=dict(color="yellow")),
        # Colorscale selector
        dbc.Row([
            dbc.Col(
                html.P("Select Colorscale:"), width=1
            ),
            dbc.Col(
                dcc.Dropdown(
                    id='color_select',
                    options=colorscales,
                    value='viridis',
                    style={'color': 'black'}
                ), width=4
            ),
            dbc.Col(
                html.P("Chart Title:"), width=1
            ),
            dbc.Col(
                dbc.Input(
                    id='title',
                    placeholder="Chart Title",
                    type="text",
                    style={'color': 'black'}
                ), width=6
            ),
        ]
        ),

        # Colorscale selector
        dbc.Row([
            dbc.Col(
                html.P("X-Axis Title:"), width=1
            ),
            dbc.Col(
                dbc.Input(
                    id='x_title',
                    placeholder="Title of X-Axis",
                    type="text",
                    style={'color': 'black'}
                ), width=5
            ),
            dbc.Col(
                html.P("Y-Axis Title:"), width=1
            ),
            dbc.Col(
                dbc.Input(
                    id='y_title',
                    placeholder="Title of Y-Axis",
                    type="text",
                    style={'color': 'black'}
                ), width=5
            ),

        ]
        ),
        html.H4("Contour Settings", style=dict(color="yellow")),
        dbc.Row(
            [
                dbc.Col(
                    dbc.Checklist(
                        options=[
                            {"label": "Connect Gaps", "value": True},
                        ],
                        value=[False],
                        id="gaps",
                        inline=True,
                    ), width=2
                ),
                dbc.Col(html.P("Smoothing"), width=1),
                dbc.Col(
                    dcc.Slider(min=0, max=1.3, step=0.1, value=0, id='smoos'), width=3
                ),
                dbc.Col(
                    html.P("Grid Method:"), width=1
                ),
                dbc.Col(
                    dbc.Select(
                        id="grid_method",
                        options=[
                            {"label": "Nearest", "value": "nearest"},
                            {"label": "Linear", "value": "linear"},
                            {"label": "Cubic", "value": "cubic", },
                        ],
                        value="cubic",
                    ),
                    width=3
                ),

            ]
        ),
        dbc.Row([
            dbc.Col(
                html.P("Select Number of Contours:"), width=2
            ),
            dbc.Col(
                dbc.Input(type="number", min=5, max=30, step=1, value=15, id="contour-slider"), width=3
            ),
            dbc.Col(
                html.P("Select Mesh Levels:"), width=2
            ),
            dbc.Col(
                dbc.Input(type="number", min=10, max=200, step=1, value=100, id="mesh-slider"), width=3
            ),
        ]),

        dbc.Row(
            [dbc.Col(html.Button("Download Excel", id="btn_xlsx"), ),
             dbc.Col(dcc.Download(id="download-dataframe-xlsx"), ), ]
        ),

    ]
)

# Initialize the Dash app
app = dash.Dash(__name__, external_stylesheets=[dbc.themes.DARKLY])
app.title = 'Contour Plotter'
server = app.server

# App layout
app.layout = dbc.Container([
    dcc.Store(id='stored-data'),
    html.H1(children=["Contour Plotter"]),
    html.Plaintext(
        children=["     A dashboard for creating contour plots from XYZ data . Created by ",
                  html.A("Amir Alansari", href='https://www.amiralansari.com', target="_blank")],
        className='text-muted'),
    html.Hr(style=dict(border="1px solid white")),
    dbc.Row(
        dbc.Col(dcc.Upload(
            id='upload-data',
            children=html.Div(['Drag and Drop or ', html.A('Select Files')]),
            style={
                'width': '100%',
                'height': '60px',
                'lineHeight': '60px',

                'borderWidth': '1px',
                'borderStyle': 'dashed',
                'borderRadius': '5px',
                'textAlign': 'center',
                'margin': '10px'

            },
            multiple=False
        ), width="Auto")
    ),
    dbc.Row(
        dbc.Col(dcc.Graph(id='contour-plot'), width=12),
    ),
    html.Hr(),
    html.Br(),
    dbc.Row(
        dbc.Col(chart_settings, width=12)
    )
],
    fluid=True,
    style={'width': '80%', }

)


# Callback for file upload
@app.callback(
    Output('contour-plot', 'figure'),
    Output('stored-data', 'data'),
    Input('upload-data', 'contents'),
    Input('color_select', 'value'),
    Input('title', 'value'),
    Input('x_title', 'value'),
    Input('y_title', 'value'),
    Input('gaps', 'value'),
    Input('smoos', 'value'),
    Input('contour-slider', 'value'),
    Input('mesh-slider', 'value'),
    Input('grid_method', 'value'),
    State('upload-data', 'filename'),
)
def update_output(contents, scale, title, x_title, y_title, gaps, smoos, num_contours, meshes, grid, filename, ):
    if contents is not None:
        gaps = gaps[0]
        content_type, content_string = contents.split(',')
        decoded = base64.b64decode(content_string)
        df = pd.read_excel(io.BytesIO(decoded))
        df = df.dropna()
        # Data processing
        x = np.array(df['X'].tolist())
        y = np.array(df['Y'].tolist())
        z = np.array(df['Z'].tolist())

        xi = np.linspace(np.min(x), np.max(x), meshes)
        yi = np.linspace(np.min(y), np.max(y), meshes)

        # Make XYZ grid
        X, Y = np.meshgrid(xi, yi)
        Z = griddata((x, y), z, (X, Y), method=grid, rescale=True)

        X_flat = X.flatten()
        Y_flat = Y.flatten()
        Z_flat = Z.flatten()

        user_df = pd.DataFrame({
            'X': X_flat,
            'Y': Y_flat,
            'Z': Z_flat
        })

        user_df = user_df.pivot(index='Y', columns='X', values='Z')
        user_df = user_df.reset_index()
        user_df = user_df.rename(columns={'index': 'Y'})

        data_to_store = user_df.to_dict('records')

        # Plot
        fig = go.Figure(go.Contour(
            x=xi,
            y=yi,
            z=Z,
            colorscale=scale,
            connectgaps=gaps,
            line_smoothing=smoos,
            ncontours=num_contours,
        )
        )

        fig.update_layout(
            height=900,
            title=title,
        )

        fig.update_xaxes(
            title=x_title,
            ticks='outside',
            mirror=True,
            linecolor='black',
            titlefont=dict(size=20),
            tickfont=dict(size=15),
        )

        fig.update_yaxes(
            title=y_title,
            ticks='outside',
            mirror=True,
            linecolor='black',
            titlefont=dict(size=20),
            tickfont=dict(size=15),
        )

        return fig, data_to_store

    return go.Figure(), None


@app.callback(
    Output("download-dataframe-xlsx", "data"),
    Input("btn_xlsx", "n_clicks"),
    State('stored-data', 'data'),
    prevent_initial_call=True,
)
def func(n_clicks, stored_data):
    if stored_data is not None:
        # Convert the stored data back to DataFrame
        df_to_download = pd.DataFrame(stored_data)
        return dcc.send_data_frame(df_to_download.to_excel, "mydf.xlsx", )


# Run the app
if __name__ == '__main__':
    app.run_server(debug=True)
