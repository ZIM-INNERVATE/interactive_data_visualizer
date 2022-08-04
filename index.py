import os

from dotenv import load_dotenv

from app import app, server
from callbacks import callbacks

dotenv_path = os.path.join(os.path.dirname(__file__), os.getenv("ENVIRONMENT_FILE"))
load_dotenv(dotenv_path=dotenv_path, override=True)

if __name__ == '__main__':
    app.run_server(host = os.environ.get("HOST"),
                   port = int(os.environ.get("PORT")),
                   debug = bool(os.environ.get("DEBUG")))
