import dash_bootstrap_components as dbc
from dash import html

from layout.option_layout import *
from layout.visualization_layout import *

layout = dbc.Container([
          html.Div(#className='row', #this causes artifact
                   children=[
                      option_layout,
                      visualization_layout
                    ])
          ],
          fluid=True
)
