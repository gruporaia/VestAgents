from typing import ClassVar, Set

from crewai_tools import SerperDevTool


class BlacklistSerperDevTool(SerperDevTool):
    """
    Extensão de SerperDevTool que insere operadores '-site:' automaticamente,
    bloqueando domínios indesejados na busca.
    """

    # Lista de domínios a bloquear (sem 'https://' ou 'www.')
    DOMINIOS_BLOQUEADOS: ClassVar[Set[str]] = {
        "wikipedia.es",
        "wikipedia.de",
        "wikipedia.pe",
        "wikipedia.mx",
        "wikipedia.ru",
        "wikipedia.ar",
        "wikipedia.it",
        "wikipedia.com.br",
        "infoescola.com",
        "wikipedia.pt",
        "wikipedia.com",
        "wikipedia.org",
        "wikipedia.cl",
        "wikipedia.fr",
        "wikipedia.co",
        "todamateria.com.br",
        "wikipedia.org.br",
        "brasilescola.uol.com.br",
        "mundoeducacao.uol.com.br",
        "teachy.com.br",
    }

    def _build_query(self, original_query: str) -> str:
        """
        Constrói a query final adicionando '-site:dominio' para cada domínio bloqueado.
        """
        exclusoes = " ".join(f"-site:{dominio}" for dominio in self.DOMINIOS_BLOQUEADOS)
        return f"{original_query} {exclusoes}".strip()

    # Sobrescrevemos o método _run para aceitar kwargs e inserir a blacklist
    def _run(self, **kwargs) -> dict:
        # Extrai a query original
        original_query = kwargs.get("search_query", "")
        # Insere a blacklist na query
        query_com_blacklist = self._build_query(original_query)
        # Chama o SerperDevTool original com a query ajustada
        return super()._run(search_query=query_com_blacklist)

    async def _arun(self, **kwargs) -> dict:
        original_query = kwargs.get("search_query", "")
        query_com_blacklist = self._build_query(original_query)
        return await super()._arun(search_query=query_com_blacklist)
