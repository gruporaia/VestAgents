from dotenv import load_dotenv
from langchain_community.docstore.in_memory import InMemoryDocstore
import faiss
from langchain_community.vectorstores import FAISS
from langchain_openai import OpenAIEmbeddings
import pandas as pd

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


user_inputs = [
    "Quero uma questão de ENEM sobre funções exponenciais.",
    "Me envie uma questão de história sobre a Revolução Francesa.",
    "Pode gerar uma questão de física sobre leis de Newton?",
    "Quero uma questão de biologia sobre fotossíntese.",
    "Gere uma questão de química sobre ligações químicas.",
    "Quero uma questão de geografia sobre mudanças climáticas.",
    "Manda uma questão de sociologia sobre desigualdade social.",
    "Me dá uma questão de filosofia sobre Sócrates.",
    "Quero uma questão de português sobre análise sintática.",
    "Gera uma questão de matemática sobre probabilidade.",
    "Quero uma questão de física sobre eletromagnetismo.",
    "Me envie uma questão de história do Brasil sobre a Era Vargas.",
    "Quero uma questão de biologia sobre genética.",
    "Pode gerar uma questão de química sobre pH e soluções?",
    "Quero uma questão de geografia sobre globalização.",
    "Me dá uma questão de filosofia sobre empirismo.",
    "Gere uma questão de português sobre interpretação de texto.",
    "Quero uma questão de matemática sobre matrizes.",
    "Manda uma questão de física sobre termodinâmica.",
    "Preciso de uma questão de sociologia sobre movimentos sociais.",
    "Me envie uma questão de história sobre a Guerra Fria.",
    "Quero uma questão de biologia sobre sistema imunológico.",
    "Gera uma questão de química sobre tabela periódica.",
    "Quero uma questão de geografia sobre urbanização.",
    "Me dá uma questão de filosofia sobre ética.",
    "Manda uma questão de português sobre funções da linguagem.",
    "Preciso de uma questão de matemática sobre análise combinatória.",
    "Quero uma questão de física sobre óptica geométrica.",
    "Me envie uma questão de história sobre o Império Romano.",
    "Gera uma questão de biologia sobre ciclo do carbono.",
    "Quero uma questão de química sobre reações de oxirredução.",
    "Me dá uma questão de geografia sobre migração populacional.",
    "Pode gerar uma questão de sociologia sobre cultura e identidade?",
    "Quero uma questão de filosofia sobre Kant.",
    "Me envie uma questão de português sobre crase.",
    "Gere uma questão de matemática sobre geometria espacial.",
    "Quero uma questão de física sobre movimento uniformemente variado.",
    "Me dá uma questão de história sobre o Iluminismo.",
    "Manda uma questão de biologia sobre evolução das espécies.",
    "Preciso de uma questão de química sobre gases.",
    "Quero uma questão de geografia sobre fontes de energia renovável.",
    "Gere uma questão de sociologia sobre trabalho e sociedade.",
    "Me envie uma questão de filosofia sobre niilismo.",
    "Quero uma questão de português sobre figuras de linguagem.",
    "Pode gerar uma questão de matemática sobre estatística descritiva?",
    "Me dá uma questão de física sobre leis de Kepler.",
    "Quero uma questão de história sobre o período colonial brasileiro.",
    "Manda uma questão de biologia sobre respiração celular.",
    "Gera uma questão de química sobre equilíbrio químico.",
    "Preciso de uma questão de geografia sobre biomas brasileiros."
]

index = faiss.IndexFlatL2()

vector_store = FAISS(
    embedding_function=embeddings,
    index=index,
    docstore=InMemoryDocstore(),
    index_to_docstore_id={},
)
breakpoint()
db = vector_store.load_local(
    folder_path="artifacts/questions_faiss",
    embeddings=embeddings,
    allow_dangerous_deserialization=True,
)

questions_dict = {'input': [], 'output': []}
for user_input in user_inputs:

    documents = db.similarity_search(
        query=user_input,
        k=3,
        filter={"university": "ENEM"},
    )
    # add prints
    
    question = "No relevant document found."
    if documents:
        question = documents[0].page_content
    # breakpoint()
    print(f"User input: {user_input}")
    print(f"Documents found: {question}")
    questions_dict['input'].append(user_input)
    questions_dict['output'].append(question)
    # breakpoint()


questions_df = pd.DataFrame.from_dict(questions_dict)
print("DataFrame from column-oriented dictionary:")
print(questions_df)

# breakpoint()
questions_df.to_csv("./csvs/enem_questions.csv", index=False)