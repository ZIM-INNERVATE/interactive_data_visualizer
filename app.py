import dash
import dash_bootstrap_components as dbc
import flask
from flask_caching import Cache

from layout.main_layout import layout

server = flask.Flask(__name__)

app = dash.Dash(
    __name__, 
    server=server,
    external_stylesheets=[dbc.themes.BOOTSTRAP,
                          "./assets/style.css",
                          'https://codepen.io/chriddyp/pen/bWLwgP.css'
                          ],
    suppress_callback_exceptions=True
)

cache = Cache(app.server, 
              config={'CACHE_TYPE': 'filesystem',
                      'CACHE_DIR': 'cache-directory'})

app.title = "Interactive Data Visualizer for Scientific Experimentation and Evaluation"
app.layout = layout

server = app.server
