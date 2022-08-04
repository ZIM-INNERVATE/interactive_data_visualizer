import os

from app import app, server
from callbacks import callbacks


if __name__ == '__main__':
    if not os.environ.get("HOST"):
        os.environ["HOST"] = "0.0.0.0"
    if not os.environ.get("PORT"):
        os.environ["PORT"] = 8085
    if not os.environ.get("DEBUG"):
        os.environ["DEBUG"] = False

    app.run_server(host = os.environ.get("HOST"),
                   port = int(os.environ.get("PORT")),
                   debug = bool(os.environ.get("DEBUG")))
