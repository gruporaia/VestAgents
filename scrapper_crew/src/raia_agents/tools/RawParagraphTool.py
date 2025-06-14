# raw_paragraph_tool.py
from typing import Type

from crewai.tools import BaseTool
from pydantic import BaseModel, Field


class RawParagraphInput(BaseModel):
    url: str = Field(..., description="URL da página a extrair")


class RawParagraphTool(BaseTool):
    # → ANOTAÇÃO de tipo obrigatória
    name: str = "raw_paragraphs"
    description: str = (
        "Dada uma URL, retorna todos os parágrafos <p> na íntegra, sem limpar "
        "ou alterar nada no texto original."
    )
    args_schema: Type[BaseModel] = RawParagraphInput

    def _run(self, url: str) -> dict:
        import requests
        from bs4 import BeautifulSoup

        resp = requests.get(url)
        resp.raise_for_status()
        soup = BeautifulSoup(resp.text, "html.parser")
        paras = [str(p) for p in soup.find_all("p")]
        return {"raw_html_paragraphs": paras}

    async def _arun(self, url: str) -> dict:
        # se seu framework chamar async
        return self._run(url)
