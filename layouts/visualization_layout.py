import dash_bootstrap_components as dbc
from dash import dash_table, dcc, html

visualization_layout = html.Div(
        className='nine columns div-for-charts bg-white',
        children=[
            dcc.Tabs(
              id="tabs-for-plots",
              value="tab-data",
              parent_className='custom-tabs',
              className='custom-tabs-container',
              children=[
                  dcc.Tab(label='Data', value='tab-data',
                      children=[
                          html.Div([
                              html.Label("Experimentation Data", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dash_table.DataTable(
                                  id="experimentation-data",
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
                                  style_table={'height': '720px', 'overflowY': 'auto'},
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
                  dcc.Tab(label='Scatter Plot', value='tab-scatter',
                      children=[
                          dcc.Graph(id='xy-scatter',
                                    #figure=dict(layout=dict(autosize=False)),
                                    config={'displayModeBar': True},
                                    # config={'displayModeBar': True, 'responsive': True},
                                    animate=True,
                                    ),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Distribution and Histogram', value='tab-histogram',
                      children=[
                          dcc.Graph(id='pca-all-fig',
                                    config={'displayModeBar': True},
                                    animate=True),
                          dcc.Graph(id='pca-group-fig',
                                    config={'displayModeBar': True},
                                    animate=True),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
                  dcc.Tab(label='Normality Test', value='tab-normality-test',
                      children=[
                          html.Div([
                              html.Label("Normality test results with lilliefors", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dash_table.DataTable(
                                  id="normality-test-table-lilliefors",
                                  style_data={'color': 'black'},
                                  style_cell={'textAlign': 'center'},
                                  style_data_conditional=[
                                      {
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(220, 220, 220)',
                                      }
                                  ],
                                  style_header={
                                      'backgroundColor': 'rgb(210, 210, 210)',
                                      'color': 'black',
                                      'fontWeight': 'bold',
                                      'textAlign': 'center'
                                  }
                              ),
                          ], style={'display': 'inline-block', 'padding': '10px 10px 10px 10px'}
                          ),
                          html.Div([
                              html.Label("Normality test results with Shapiro-Wilk", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dash_table.DataTable(
                                  id="normality-test-table-shapiro",
                                  style_data={'color': 'black'},
                                  style_cell={'textAlign': 'center'},
                                  style_data_conditional=[
                                      {
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(220, 220, 220)',
                                      }
                                  ],
                                  style_header={
                                      'backgroundColor': 'rgb(210, 210, 210)',
                                      'color': 'black',
                                      'fontWeight': 'bold',
                                      'textAlign': 'center'
                                  }
                              ),
                          ],
                          style={'display': 'inline-block', 'padding': '10px 10px 10px 10px'}
                          ),
                          html.Div([
                              html.Label("Normality test results with chi-square", 
                                      style = {'color': 'black',
                                              'textAlign': 'center',
                                              'fontWeight': 'bold'},
                                    ),
                              dash_table.DataTable(
                                  id="normality-test-table-chi2",
                                  style_data={'color': 'black'},
                                  style_cell={'textAlign': 'center'},
                                  style_data_conditional=[
                                      {
                                          'if': {'row_index': 'odd'},
                                          'backgroundColor': 'rgb(220, 220, 220)',
                                      }
                                  ],
                                  style_header={
                                      'backgroundColor': 'rgb(210, 210, 210)',
                                      'color': 'black',
                                      'fontWeight': 'bold',
                                      'textAlign': 'center'
                                  }
                              ),
                          ],
                          style={'display': 'inline-block', 'padding': '10px 10px 10px 10px'}
                          ),
                          dcc.Graph(id='normality-test-fig',
                                    config={'displayModeBar': True},
                                    animate=True),
                      ],
                      className='custom-tab',
                      selected_className='custom-tab--selected',
                  ),
              ],
              style={'color': '#ffffff'}
            ),
        ]
)
