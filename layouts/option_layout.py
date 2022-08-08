import dash_bootstrap_components as dbc
from dash import dcc, html

option_layout = html.Div(
  className='three columns div-user-controls',
  children=[
    html.H3('Interactive Data Visualizer'),
    html.P('Data Analysis and Visualization for Scientific Experimentation and Evaluation'),
    html.Div(
        className='div-for-dropdown',
            children=[
                dbc.Row([
                    dcc.Upload(
                        children=html.Div(['Drag and Drop or ', dbc.Button('Upload File')]), 
                        style={'width': '100%','height': '60px','lineHeight': '60px',
                                'borderWidth': '1px','borderStyle': 'dashed','borderRadius': '5px',
                              'textAlign': 'center'},
                        id="upload-data",
                        accept=".csv") 
                    ],
                    style={'padding': '10px 0px 10px 0px'}
                ),
                dbc.Row([
                    html.Div([
                        # do not display file-list
                        html.Ul(id="file-list", style= {'display': 'none'}),
                    ]),
                  ],
                ),
                dbc.Row([
                    html.Div([
                        dcc.Dropdown(id='dropdown-csv-separator',
                                    style = {'width': '100%',
                                             'color': '#212121',
                                            },
                                    options = [{"label": ";", "value": ";"},
                                               {"label": ",", "value": ","}],
                                    value = ",",
                                    placeholder="CSV separator (default: ,)",
                                    searchable=False,
                                    clearable=False,
                                    multi=False),
                        ],
                        className="mb-3 gap-2 d-md-flex justify-content-md-center",
                    ),
                  ],
                  style={'padding': '10px 0px 10px 0px'}
                ),
                dbc.Row([
                    html.Div([
                        dcc.Dropdown(id='dropdown-select-file',   
                                    style = {'width': '100%',
                                             'color': '#212121',
                                            },
                                    options = [],
                                    value = "",
                                    placeholder="Select file",
                                    searchable=True,
                                    clearable=True,
                                    multi=False),
                        dbc.Button("Select", color="primary", className="me-1", id="select-button", n_clicks=0),
                        dbc.Button("Delete", color="danger", className="me-1", id="delete-button", n_clicks=0),
                        ],
                        className="mb-3 gap-2 d-md-flex justify-content-md-center",
                    ),
                  ],
                  style={'padding': '10px 0px 10px 0px'}
                ),
                dbc.Row([
                    html.Div([
                        html.P("Selected files:"),
                        html.Ul(id="output-selected-file")
                        ],
                        className="align-self-center",
                        style= {'display': 'none'}
                      ),
                    ],
                ),
                dbc.Row([
                    html.Div([
                        dbc.Form([
                            dbc.Label("Groups"),
                            dbc.Row([
                              dbc.Col([
                                dcc.Dropdown(id='dropdown-groups',   
                                              style = {'width': '100%',
                                                       'color': '#212121',
                                                      },   
                                              options = [],
                                              value = [],
                                              placeholder="Select group",
                                              searchable=True,
                                              clearable=True,
                                              multi=True
                                ),
                              ])
                            ]),
                            dbc.Row([
                              dbc.Col([
                                dbc.Label("Motions"),
                                dbc.Checklist(
                                    options=[{'label': 'left', 'value':'left'},
                                             {'label': 'straight', 'value':'straight'},
                                             {'label': 'right', 'value':'right'}
                                            ],
                                    value=["left", "straight", "right"],
                                    id="checklist-motions",
                                    inline=False,
                                    switch=True,
                                ),
                                dbc.Checklist(
                                    options=[{'label': 'All', 'value':'All'}],
                                    value=['All'],
                                    id="checklist-motions-all",
                                    inline=True,
                                    switch=True,
                                ),
                              ]),
                              dbc.Col([
                                dbc.Label("Weights"),
                                dbc.Checklist(
                                    options=[{'label': 'small', 'value':'small'},
                                             {'label': 'medium', 'value':'medium'},
                                             {'label': 'large', 'value':'large'},
                                            ],
                                    value=["small", "medium", "large"],
                                    id="checklist-weights",
                                    inline=False,
                                    switch=True,
                                ),
                                dbc.Checklist(
                                    options=[{'label': 'All', 'value':'All'}],
                                    value=['All'],
                                    id="checklist-weights-all",
                                    inline=True,
                                    switch=True,
                                ),
                              ]),
                            ],
                            style={'padding': '10px 0px 10px 0px'}
                            ),
                        ]),
                        html.P(id="form-checklist-button"),
                    ],
                  )],
                  style={'padding': '10px 0px 10px 0px'}
                ),
            ],
    ),
  ]
)
