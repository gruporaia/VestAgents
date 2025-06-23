from typing import Type

import faiss
from crewai.tools import BaseTool
from langchain.prompts import FewShotPromptTemplate, PromptTemplate
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
from litellm import BaseModel, Field

index = faiss.IndexFlatL2()
embeddings = OpenAIEmbeddings(model="text-embedding-ada-002")
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
        description="Limite de similaridade para considerar um trecho relevante. Um valor razoável é 0.3 e no máximo 0.5, mas pode ser ajustado conforme necessário."
    )


class RetrieveQuestoesTool(BaseTool):
    name: str = "Questions Retriever"
    description: str = (
        "Function that retrieves data of a database, based on user topic of interest, so that it helps to generate questions."
    )
    args_schema: Type[BaseModel] = RagToolSchema

    def _run(
        self, topic: str, amount_to_retrieve: int = 5, threshold: float = 0.3
    ) -> str:
        vector_store = FAISS.load_local(
            "artifacts/questions_faiss_v2/",
            embeddings,
            allow_dangerous_deserialization=True,
        )
        docs = vector_store.similarity_search_with_relevance_scores(
            k=amount_to_retrieve,
            query=topic,
            score_threshold=threshold,
        )

        examples = [
            {
                "questao":
                    f"{doc.page_content.split('Alternatives')[0].strip()}\n\n" +
                    (
                        f"Alternativas:\n{doc.page_content.split('Alternatives:')[1].strip()}"
                        if "Alternatives:" in doc.page_content
                        else "Alternativas não encontradas."
                    )
            }
            for doc, _ in docs
        ]

        example_prompt = PromptTemplate(
            input_variables=["questao"], template="Questão: \n{questao}\n\n"
        )

        few_shot_prompt = FewShotPromptTemplate(
            examples=examples,
            example_prompt=example_prompt,
            prefix="Aqui estão exemplos de questões no tópico fornecido:",
            suffix="Gere uma nova questão de acordo com o tema: {input}",
            input_variables=["input"],
        )

        prompt = few_shot_prompt.format(input=topic)
        return prompt
