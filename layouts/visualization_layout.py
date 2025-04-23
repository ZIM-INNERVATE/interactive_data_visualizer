import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html
import utils.utils as utils

visualization_layout = html.Div(
        className='nine columns div-for-charts bg-white',
        children=[
            # Shared store for all triggers and lines
            dcc.Store(id='shared_triggers',
                      storage_type='session',
                      data={
                           "triggers": {},   # For vertical lines
                           "dragging": None
                      }),
            dcc.Store(id='shared_limits',
                      storage_type='session',
                      data={
                           "limits": {},   
                           "dragging": None
                      }),
            dcc.Tabs(
              id="tabs-for-plots",
              value="tab-data",
              parent_className='custom-tabs',
              className='custom-tabs-container',
              children=[
                   dcc.Tab(label='HDF5 Data', value='tab-data-hdf5',
                      children=[
                          html.Div([
                              html.Label("HDF5 Data Tables", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dcc.Tabs(
                                  id='hdf5-data-tabs',
                                  className='custom-tabs-container',
                              ),
                          ]),
                          html.Div(id='hdf5-data-tables-container')
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Metadata', value='tab-metadata',
                      children=[
                          html.Div([
                              html.Label("Group Metadata", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dash_table.DataTable(
                                  id="metadata-data",
                                  style_data={'color': 'black'},
                                  style_cell={'textAlign': 'center'},
                                  style_data_conditional=[
                                      {
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(220, 220, 220)',
                                          'textAlign': 'center'
                                      }
                                  ],
                                  page_size=25,
                                  style_table={'height': '120px', 'overflowY': 'auto'},
                                  style_header={
                                      'backgroundColor': 'rgb(210, 210, 210)',
                                      'color': 'black',
                                      'fontWeight': 'bold',
                                      'textAlign': 'center'
                                  }
                              ),
                              dash_table.DataTable(
                                  id="triggers-data",
                                  style_data={'color': 'black'},
                                  style_cell={'textAlign': 'center'},
                                  style_data_conditional=[
                                      {
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(220, 220, 220)',
                                          'textAlign': 'center'
                                      }
                                  ],
                                  page_size=25,
                                  style_table={'height': '120px', 'overflowY': 'auto'},
                                  style_header={
                                      'backgroundColor': 'rgb(210, 210, 210)',
                                      'color': 'black',
                                      'fontWeight': 'bold',
                                      'textAlign': 'center'
                                  }
                              ),
                          ],
                          #style={'display': 'inline-block', 'padding': '10px 10px 10px 10px'}
                          ),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Area calculation', value='tab-area',
                      children=[
                          dcc.Graph(id='area_under_radius',
                                    config=utils.create_graph_config(),
                                    animate=False,
                                    figure={'layout': utils.create_graph_layout()},
                                    style={'height': '2400px'}
                                    ),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Moving Average for Radius', value='tab-avg-radius',
                      children=[
                          dcc.Graph(id='mov_avg_radius',
                                    config=utils.create_graph_config(),
                                    animate=False,
                                    figure={'layout': utils.create_graph_layout()},
                                    style={'height': '2400px'}
                                    ),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Feature Extraction', value='tab-feature-extraction',
                      children=[
                         html.Div(id='feature-tab-content') 
                        #   html.Div(
                            #   children=[
                            #       dcc.Graph(id='features_extraction_{key}',
                            #             figure=fig,
                            #             config=create_graph_config(),
                            #             animate=False,
                            #             # style={'height': '2400px'}
                            #             )
                            #             for key, fig in features_plots.items()
                            #         ]
                            #     )
                            ],
                        className='custom-tab',
                        selected_className='custom-tab--selected',
                  ),
              ],
              style={'color': '#ffffff'}
            ),
            # Add modal for function editing
            dbc.Modal([
                dbc.ModalHeader("Edit Line Functions"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Select Subplot:"),
                            dcc.Dropdown(
                                id='subplot-selector-function', 
                                placeholder="Select subplot to edit",
                                className="mb-3",
                                options = [],
                                value = "",
                                searchable=True,
                                clearable=True,
                                multi=False
                            ),
                        ]),
                        dbc.Row([
                            dbc.Label("Select Line:"),
                            dcc.Dropdown(
                                id='horizontal-line-selector',
                                options=[
                                    {'label': 'Target Line', 'value': 'target'},
                                    {'label': 'Warning High', 'value': 'warning_high'},
                                    {'label': 'Warning Low', 'value': 'warning_low'},
                                    {'label': 'Alarm High', 'value': 'alarm_high'},
                                    {'label': 'Alarm Low', 'value': 'alarm_low'}
                                ],
                                placeholder="Select line to edit",
                                className="mb-3"
                            ),
                        ]),
                        dbc.Row([
                            dbc.Label("Function:"),
                            dbc.Input(
                                id="function-input",
                                placeholder="Enter function (e.g., y = 41 or y = 0.001*x + 40)",
                                type="text",
                                className="mb-3"
                            ),
                        ]),
                    ]),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Apply", id="apply-changes-function-modal", color="primary"),
                    dbc.Button("Close", id="close-function-modal", className="ml-2"),
                ]),
            ], id="function-modal"),

            # Add modal for triggers editing
            dbc.Modal([
                dbc.ModalHeader("Edit Triggers"),
                dbc.ModalBody([
                    dbc.Form([
                        dbc.Row([
                            dbc.Label("Select Subplot:"),
                            dcc.Dropdown(
                                id='subplot-selector-triggers',
                                placeholder="Select subplot to edit",
                                className="mb-3",
                                options = [],
                                value = "",
                                searchable=True,
                                clearable=True,
                                multi=False
                            ),
                            dbc.Button("Select", id="select-subplot-triggers", color="primary"),
                        ]),
                        dbc.Row([
                            dbc.Label("Star Trigger Value:"),
                            dbc.Input(
                                id="start-trigger-value-input",
                                placeholder="Enter a value",
                                type="number",
                                value = "",
                                className="mb-3"
                            ),
                            dbc.Label("Stop Trigger Value:"),
                            dbc.Input(
                                id="stop-trigger-value-input",
                                placeholder="Enter a value",
                                type="number",
                                value = "",
                                className="mb-3"
                            ),
                        ]),
                    ]),
                ]),
                dbc.ModalFooter([
                    dbc.Button("Apply", id="apply-changes-triggers-modal", color="primary"),
                    dbc.Button("Close", id="close-triggers-modal", className="ml-2"),
                ]),
            ], id="triggers-modal")
        ]
)
