from starlette.applications import Starlette
from src.main.routes.starlette_routes import routes

app = Starlette(routes=routes)

# uvicorn run:app --host 0.0.0.0 --port 8000
# https://uvicorn.dev/deployment/docker/#quickstart