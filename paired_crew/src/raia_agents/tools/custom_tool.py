from typing import Type

import faiss
from crewai.tools import BaseTool
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

# Load the FAISS index from a file
index = faiss.IndexFlatL2(3080)
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


class EmptyToolResultError(Exception):
    """Raised when the tool returns an empty result."""


class Retriever:
    def __init__(self, folder_path: str = "artifacts/"):
        self.vector_store = FAISS.load_local(
            folder_path,
            embeddings,
            allow_dangerous_deserialization=True,
        )

    def invoke(self, topic, amount_to_retrieve, threshold):
        retrieve = self.vector_store.similarity_search_with_relevance_scores(
            k=amount_to_retrieve,
            query=topic,
            score_threshold=threshold,
        )

        return retrieve


class RetrieveQuestoesToolSchema(BaseModel):
    """Schema para pegar o tópico de interesse do usuário para poder gerar questões"""

    pergunta: str = Field(
        ...,
        description="A pergunta educacional que deve ser usada para recuperar questões similares.",
    )
    topic: str = Field(
        description="Tópico para fazer a busca por similaridade e gerar as questões atreladas a esse assunto"
    )
    amount_to_retrieve: int = Field(
        description="Quantidade de trechos de similaridade para ser usado na busca"
    )
    threshold: float = Field(
        description="Limite de similaridade para considerar um trecho relevante. Um valor razoável é 0.2, mas pode ser ajustado conforme necessário."
    )


# Ferramenta que será utilizada pelo agente few shot para procurar 5 questões na base de dados e utilizar o resultado para ajudar no prompt do agente gerador de questões
class RetrieveQuestoesTool(BaseTool):
    name: str = "Retriever"
    description: str = (
        "Function that retrieves data of a database, based on user topic of interest, so that it helps to generate questions."
    )
    args_schema: Type[BaseModel] = RetrieveQuestoesToolSchema

    def _run(
        self, pergunta: str, topic: str, amount_to_retrieve: int, threshold: float = 0.2
    ) -> str:
        retrieve = Retriever(folder_path="artifacts/questions_faiss_v2").invoke(
            topic, amount_to_retrieve, threshold
        )

        if not retrieve or len(retrieve) == 0:
            # raise EmptyToolResultError("No documents found for the given topic.")
            return "No documents found for the given topic."
        examples = [
            {
                "pergunta": doc.page_content,
                "resposta": doc.metadata.get("correct_answer", ""),
            }
            for doc, __ in retrieve
        ]

        example_prompt = PromptTemplate(
            input_variables=["pergunta", "resposta"],
            template="Pergunta: {pergunta}\n Alternativa correta: {resposta}",
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Foi solicitado o seguinte\n{input}. \nAqui estão algumas questões semelhantes ao que foi solicitado:",
            suffix="Com base nas questões de exemplo e utilizando o seu conhecimento, gere uma nova questão para atender o pedido.",
            input_variables=["input"],
        )
        prompt = few_shot_prompt.format(input=pergunta)
        return prompt


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
