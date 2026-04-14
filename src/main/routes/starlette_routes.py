#pylint: disable=unused-argument
from starlette.routing import Mount, Route
from starlette.responses import RedirectResponse
from src.main.shiny_apps.rag_bi_app import RAGBIApp

async def home(request):
    return RedirectResponse(url='/rag-bi')

routes = [
    Route("/", endpoint=home),
    Mount("/rag-bi", app=RAGBIApp(), name="rag-bi"),
]
