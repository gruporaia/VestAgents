import getpass
import os
from typing import List, Optional, Tuple, Type, Union

import faiss
from crewai.tools import BaseTool
from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_core.documents import Document
from langchain_openai import OpenAIEmbeddings
from pydantic import BaseModel, Field

load_dotenv()

if not os.environ.get("OPENAI_API_KEY"):
    os.environ["OPENAI_API_KEY"] = getpass.getpass("Enter API key for OpenAI: ")


class EmptyToolResultError(Exception):
    """Raised when the tool returns an empty result."""

    pass


embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")

index = faiss.IndexFlatL2()

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)


class RagToolSchema(BaseModel):
    """Schema para pegar o tópico de interesse do usuário para poder gerar questões"""

    topic: str = Field(
        description="Tópico para fazer a busca por similaridade e gerar as questões atreladas a esse assunto"
    )
    amount_to_retrieve: int = Field(
        description="Quantidade de trechos de similaridade para ser usado na busca"
    )
    threshold: float = Field(
        description="Limite de similaridade para considerar um trecho relevante. Um valor razoável é 0.2, mas pode ser ajustado conforme necessário."
    )


class QuestionsRagToolSchema(RagToolSchema):
    """Schema para pegar o tópico de interesse do usuário para poder gerar questões"""

    threshold: float = Field(
        description="Limite de similaridade para considerar um trecho relevante. Um valor razoável é 0.3, mas pode ser ajustado conforme necessário."
    )


class CategoriesRagToolSchema(RagToolSchema):
    """Schema para pegar o tópico de interesse do usuário para poder gerar questões"""

    threshold: float = Field(
        description="Limite de similaridade para considerar um trecho relevante. Um valor razoável é 0.6, mas pode ser ajustado conforme necessário."
    )


class SingleRagToolSchema(BaseModel):
    """Schema para pegar o tópico de interesse do usuário para poder gerar questões"""

    topic: str = Field(
        description="Tópico para fazer a busca por similaridade e gerar as questões atreladas a esse assunto"
    )
    amount_to_retrieve_question: int = Field(
        description="Quantidade de trechos de similaridade para ser usado na busca por questões, considere um valor razoável de 4, mas pode ser ajustado conforme necessário."
    )
    threshold_question: Optional[float] = Field(
        description="Limite de similaridade para considerar um trecho relevante na busca por questões. Um valor razoável é 0.2."
    )
    amount_to_retrieve_category: int = Field(
        description="Quantidade de trechos de similaridade para ser usado na busca por categorias, considere um valor razoável de 4, mas pode ser ajustado conforme necessário."
    )
    threshold_category: Optional[float] = Field(
        description="Limite de similaridade para considerar um trecho relevante na busca por categorias. Um valor razoável é 0.7."
    )


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


class CategoryRAGTool(BaseTool):
    name: str = "Category Retriever"
    description: str = (
        "Function that retrieves data of a database, based on user topic of interest, so that it helps to generate questions."
    )
    args_schema: Type[BaseModel] = CategoriesRagToolSchema

    def _run(
        self, topic: str, amount_to_retrieve: int, threshold: float = 0.2
    ) -> Union[List[Tuple[Document, float]], str]:
        retrieved = Retriever(folder_path="artifacts/questions_faiss").invoke(
            topic, amount_to_retrieve, threshold
        )

        if not retrieved or len(retrieved) == 0:
            return "No documents found for the given topic."
        return retrieved


class QuestionsRAGTool(BaseTool):
    name: str = "Questions Retriever"
    description: str = (
        "Function that retrieves data of a database, based on user topic of interest, so that it helps to generate questions."
    )
    args_schema: Type[BaseModel] = QuestionsRagToolSchema

    def _run(
        self, topic: str, amount_to_retrieve: int, threshold: float = 0.2
    ) -> Union[List[Tuple[Document, float]], str]:
        retrieved = Retriever(folder_path="artifacts/questions_faiss").invoke(
            topic, amount_to_retrieve, threshold
        )

        if not retrieved or len(retrieved) == 0:
            return "No documents found for the given topic."
        return retrieved


class SingleRagTool(BaseTool):
    name: str = "Single Retriever"
    description: str = (
        "Function that retrieves data of a database, based on user topic of interest, so that it helps to generate questions."
    )
    args_schema: Type[BaseModel] = SingleRagToolSchema

    def _run(
        self,
        topic,
        amount_to_retrieve_question,
        threshold_question,
        amount_to_retrieve_category,
        threshold_category,
    ) -> Union[List[Tuple[Document, float]], str]:
        topic = topic
        amount_to_retrieve = amount_to_retrieve_question or 4
        threshold = threshold_question or 0.2
        retrieved_questions = CategoryRAGTool()._run(
            topic=topic,
            amount_to_retrieve=amount_to_retrieve,
            threshold=threshold,
        )

        if not retrieved_questions or len(retrieved_questions) == 0:
            return "No question documents found for the given topic."
        if isinstance(retrieved_questions, str):
            return retrieved_questions

        amount_to_retrieve = amount_to_retrieve_category or 4
        threshold = threshold_category or 0.7
        retrieved_categories = Retriever(folder_path="artifacts/category_faiss").invoke(
            topic, amount_to_retrieve, threshold
        )
        if not retrieved_categories or len(retrieved_categories) == 0:
            return "No category documents found for the given topic."

        seen_ids = set()
        unique_docs = []
        for doc, score in [*retrieved_questions, *retrieved_categories]:
            question_id = doc.metadata.get("question_id")
            if question_id is None or question_id not in seen_ids:
                unique_docs.append((doc, score))
            if question_id is not None:
                seen_ids.add(question_id)
        return unique_docs
