import dash_bootstrap_components as dbc
from dash import html

from layouts.option_layout import *
from layouts.visualization_layout import *

layout = dbc.Container([
          html.Div(#className='row', #this causes artifact
                   children=[
                      option_layout,
                      visualization_layout
                    ])
          ],
          fluid=True
)
