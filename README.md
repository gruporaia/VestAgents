# VestAgents

Welcome to VestAgents! This project aims to create a tool for generating original entrance-exam questions for Brazilian vestibular exams using Large Language Models (LLMs). The development is supported by the distance-education company [PICO](https://www.usepico.com.br/) and explores alternatives to the company’s existing question-generation tool.

We implement 4 strategies:
- **Retrieval-generator**: A system with inbuilt RAG (Retrieval-Augmented Generation) to search for relevant questions.
- **Few-shot**: A system employing a Few-Shot Prompt generator using the 5 most relevant questions.
- **Scraping**: Searches for data on the web to include as guiding text.
- **Paired Crew**: A crew employing both web scraping and the few-shot approach.

## Retrieval-generator:

Instead of relying solely on the LLM's internal knowledge, our tool enables an AI agent to consult past exam questions before creating a new one. This consultation is performed on a specialized database containing thousands of questions, previously cataloged by knowledge area (e.g., Mathematics, Physics) and subcategory (e.g., Kinematics, Thermodynamics).

### Solution Architecture: RAG with Similarity Search

To allow the AI agent to "search" through a text-based database, we implemented a similarity search system. The process works in two main stages: **creating a vector knowledge base** and **retrieving relevant information** in real-time.

![Rag Agent Flowchart](assets/flowchart_rag_agent_en.png)

#### 1. Embeddings: Converting Text into Vectors

The first step was to transform our entire collection of exam questions into a format that a computer could understand and numerically compare. To do this, we used a technique called **embeddings**.

* **What are Embeddings?** 🧠
    An embedding model is a neural network that converts words, sentences, or entire documents into high-dimensional numerical vectors. The key advantage of this technique is that the generated vectors capture the **semantic meaning** of the text. This means that questions with similar topics or concepts, even if written with different words, will have very close vector representations in the vector space.

#### 2. Vector Store and FAISS: Efficient Storage and Search

Once all the questions were converted into embeddings, the next challenge was to store and search them efficiently. This is where vector stores and the FAISS library come in.

* **What is a Vector Store?** 🗄️
    A vector store (or vector database) is a type of database specifically optimized for storing and querying embedding vectors. Unlike a traditional database that searches for exact text matches, a vector store finds the vectors closest to a query vector, performing a **similarity search**.

* **How does FAISS work?** ⚡
    To manage our vector store, we use the **FAISS (Facebook AI Similarity Search)** library. FAISS is a highly efficient open-source library for similarity search in dense sets of vectors. It creates a data structure (an index) that organizes the question vectors for optimal performance. When a user requests a new question, we first convert that request into an embedding vector. Then, we use FAISS to rapidly compare this vector against the millions of indexed question vectors, returning the most similar ones in milliseconds.

#### 3. Multi-Index Strategy

To improve the relevance of our search results, we did not just create a single index. Instead, we generated separate embeddings and vector stores for different aspects of the questions:

* **Raw Text Embeddings:** Allow finding questions with similar conceptual content.
* **Category and Subcategory Embeddings:** Help refine the search, ensuring that the retrieved examples belong to the correct knowledge area as requested by the user.

---

### Generation Flow

With this architecture, the AI agent responsible for creating questions follows an intelligent workflow:

1.  **Receives the Request:** The user specifies the topic for the desired question (e.g., "a question about Newton's Second Law").
2.  **Similarity Search:** The tool converts the request into an embedding and uses FAISS to search its vector store for the most semantically similar questions.
3.  **Contextualization:** The retrieved questions (e.g., 5 examples of questions about Newton's Laws) are provided to the LLM as context or "reference examples."
4.  **Augmented Generation:** Based on these high-quality examples, the LLM generates a **new and original question** that follows the style, difficulty, and format of real university entrance exam questions on that topic.

## Few-shot

This system makes use of a RAG-based tool to perform a few-shot approach to generate new questions. Instead of relying solely on the LLM's internal knowledge, our tool enables an AI agent to consult the 5 past exam questions most related to the user's prompt so that a new question can be created. This consultation is performed on a specialized database containing thousands of questions, previously cataloged by knowledge area (e.g., Mathematics, Physics) and subcategory (e.g., Kinematics, Thermodynamics).

The architecture is similar to the Retrieval-generator model, only changing the generation flow to include a post-process of the retrieval to generate a few-shot prompt. This is spread by separating the flow into two crews, where the second one gets the outputs from the previous one and embeds the few-shot prompt into the generator agent.

### Generation Flow

With this architecture, the AI agent responsible for creating questions follows an intelligent workflow:

1.  **Receives the Request:** The user specifies the topic for the desired question (e.g., "a question about Newton's Second Law").
2.  **Similarity Search:** The tool converts the request into an embedding and uses FAISS to search its vector store for the most semantically similar questions.
3.  **Contextualization:** The retrieved questions (e.g., 5 examples of questions about Newton's Laws) are provided to the LLM as context or "reference examples."
4.  **Augmented Generation:** Based on these high-quality examples, the LLM generates a **new and original question** that follows the style, difficulty, and format of real university entrance exam questions on that topic.

## Scraping

This system implements a **Web Scraping–based** approach. Instead of relying solely on the LLM’s internal knowledge, our tool allows an AI agent to query specialized websites on topics covered in traditional Brazilian vestibular exams. We perform searches via the Serper API to find the most relevant sites and then use Beautiful Soup to extract the HTML content, which provides the contextual text for our question-generation agent.

---

### 1. Preparation

The process has two main stages: **Searching for specialized sites** and **Extracting content**. An upstream agent first analyzes the user’s prompt to identify relevant details—such as subject and subtopic—needed to guide the search. For example:

```

I want to generate mathematics questions on probability for São Paulo vestibular exams

````

Here, the agent would extract **Mathematics** as the subject and **Probability** as the subtopic, then pass those parameters into the Web Scraping pipeline.

### 2. Web Search with Serper

**Serper** is a programmatic search API that returns structured JSON—titles, snippets, URLs, and metadata—instead of raw HTML meant for browsers. This allows fast, secure, and easily integrated searches into our Web Scraping pipeline.

Example usage in our agent:
```python
@agent
def search_and_extract(self) -> Agent:
    return Agent(
        config=self.agents_config['search_and_extract'],
        tools=[SerperDevTool(), RawParagraphTool()],
        verbose=True,
        memory=True
    )
````

A typical Serper API response looks like:

```json
{
  "title": "Oxidation Reactions – Organic Chemistry | Example.edu",
  "snippet": "In organic chemistry, oxidation is defined as …",
  "link": "[https://example.edu/oxidation](https://example.edu/oxidation)",
  "displayed_link": "example.edu"
}
```

Once we have these URLs, we feed them to Beautiful Soup to fetch and parse the raw HTML.

### 3\. Content Extraction with Beautiful Soup

**What is Beautiful Soup?** 🥣
Beautiful Soup is a Python library for parsing HTML and XML. It converts an HTML string into a tree of Python objects, making it easy to locate and extract specific elements without wrestling with regex or brittle DOM parsing.

**How we use it in VestAgents:**

1.  **Load the HTML**
    After retrieving the raw HTML (via `web.open`), we instantiate Beautiful Soup:
    ```python
    from bs4 import BeautifulSoup
    soup = BeautifulSoup(html_content, 'html.parser')
    ```
2.  **Extract Paragraphs**
    We collect all `<p>` tags:
    ```python
    paragraphs = [str(p) for p in soup.find_all("p")]
    ```
    This yields a list of strings, each preserving the original HTML markup, accents, and formatting.
3.  **Pipeline Integration**
    These paragraphs are supplied to the `search_and_extract` agent, which determines which snippets are most relevant to serve as “contextual passages” for the question-generation agent. By preserving the original structure of each paragraph, we ensure fidelity to the source material.

-----

Below is a diagram showing how this approach fits into our overall architecture:

1.  **Receive Request**: The user specifies the desired question topic (e.g., “a math question on probability”).
2.  **Prompt Refinement**: Intermediate agents refine the initial prompt to identify subject and subtopic (e.g., “Mathematics | Probability”).
3.  **Search & Extraction**: We query specialized sites and extract relevant text for context.
4.  **Question Generation**: Using the user’s request and the extracted passages, an agent generates a tailored vestibular-style question.

## Paired Crew

This strategy first implements the few-shot builder agent, then a second crew executes the web scraping agent, and finally merges both results to generate a new question.

-----

# Installation

Ensure you have Python **3.10–3.12** installed. This project uses [UV](https://docs.astral.sh/uv/) for dependency and package management, giving you a smooth setup and run experience.

First, if you don’t already have it, install UV:

```bash
pip install uv
```

Then, from the project root, install dependencies:

```bash
crewai install
```

### Customization

1.  Add your `OPENAI_API_KEY` to a `.env` file.
2.  Modify `src/raia_agents/config/agents.yaml` to configure your agents.
3.  Modify `src/raia_agents/config/tasks.yaml` to define tasks.
4.  Update `src/raia_agents/crew.py` to add custom logic, tools, or arguments.
5.  Update `src/raia_agents/main.py` to include custom inputs for your agents and tasks.

### Running the Project

To start your AI agent crew and execute tasks, run:

```bash
crewai run
```

This command initializes the `raia_agents` crew and dispatches agents according to your configuration.

-----

-----

# VestAgents

Bem-vindo ao VestAgents\! Este projeto visa criar uma ferramenta de geração de questões inéditas para vestibulares brasileiros utilizando Modelos de Linguagem Grandes (LLMs). O desenvolvimento conta com o apoio da empresa de educação a distância [PICO](https://www.usepico.com.br/) e explora alternativas para a ferramenta de geração de questões existente na empresa.

Implementamos 4 estratégias:

  - **Retrieval-generator**: Um sistema com busca RAG (Geração Aumentada por Recuperação) integrada para pesquisar questões relevantes.
  - **Few-shot**: Um sistema que emprega um gerador de Prompts Few-Shot usando as 5 questões mais relevantes.
  - **Scraping**: Realiza buscas na web para incluir dados como texto de apoio.
  - **Paired Crew**: Uma equipe que emprega tanto a web scraping quanto a abordagem few-shot.

## Retrieval-generator:

Em vez de depender apenas do conhecimento interno do LLM, nossa ferramenta permite que um agente de IA consulte questões de vestibulares anteriores antes de criar uma nova. Essa consulta é realizada em um banco de dados especializado que contém milhares de questões, previamente catalogadas por área de conhecimento (ex: Matemática, Física) e subcategoria (ex: Cinemática, Termodinâmica).

### Arquitetura da Solução: RAG com Busca por Similaridade

Para permitir que o agente de IA "pesquise" em um banco de dados textual, implementamos um sistema de busca por similaridade. O processo funciona em duas etapas principais: **criação de uma base de conhecimento vetorial** e **recuperação de informações relevantes** em tempo real.

#### 1\. Embeddings: Convertendo Texto em Vetores

O primeiro passo foi transformar toda a nossa coleção de questões de vestibular em um formato que um computador pudesse entender e comparar numericamente. Para isso, usamos uma técnica chamada **embeddings**.

  * **O que são Embeddings?** 🧠
    Um modelo de embedding é uma rede neural que converte palavras, frases ou documentos inteiros em vetores numéricos de alta dimensão. A principal vantagem dessa técnica é que os vetores gerados capturam o **significado semântico** do texto. Isso significa que questões com tópicos ou conceitos similares, mesmo que escritas com palavras diferentes, terão representações vetoriais muito próximas no espaço vetorial.

#### 2\. Vector Store e FAISS: Armazenamento e Busca Eficiente

Depois que todas as questões foram convertidas em embeddings, o próximo desafio era armazená-las e pesquisá-las de forma eficiente. É aqui que entram os vector stores e a biblioteca FAISS.

  * **O que é um Vector Store?** 🗄️
    Um vector store (ou banco de dados vetorial) é um tipo de banco de dados otimizado especificamente para armazenar e consultar vetores de embedding. Diferente de um banco de dados tradicional que busca por correspondências exatas de texto, um vector store encontra os vetores mais próximos a um vetor de consulta, realizando uma **busca por similaridade**.

  * **Como o FAISS funciona?** ⚡
    Para gerenciar nosso vector store, usamos a biblioteca **FAISS (Facebook AI Similarity Search)**. FAISS é uma biblioteca de código aberto altamente eficiente para busca por similaridade em conjuntos densos de vetores. Ela cria uma estrutura de dados (um índice) que organiza os vetores das questões para um desempenho otimizado. Quando um usuário solicita uma nova questão, primeiro convertemos essa solicitação em um vetor de embedding. Em seguida, usamos o FAISS para comparar rapidamente esse vetor com milhões de vetores de questões indexados, retornando os mais similares em milissegundos.

#### 3\. Estratégia de Múltiplos Índices

Para melhorar a relevância dos nossos resultados de busca, não criamos apenas um único índice. Em vez disso, geramos embeddings e vector stores separados para diferentes aspectos das questões:

  * **Embeddings de Texto Bruto:** Permitem encontrar questões com conteúdo conceitual semelhante.
  * **Embeddings de Categoria e Subcategoria:** Ajudam a refinar a busca, garantindo que os exemplos recuperados pertençam à área de conhecimento correta, conforme solicitado pelo usuário.

-----

### Fluxo de Geração

Com essa arquitetura, o agente de IA responsável por criar as questões segue um fluxo de trabalho inteligente:

1.  **Recebe a Solicitação:** O usuário especifica o tema da questão desejada (ex: "uma questão sobre a Segunda Lei de Newton").
2.  **Busca por Similaridade:** A ferramenta converte a solicitação em um embedding e usa o FAISS para buscar em seu vector store as questões semanticamente mais similares.
3.  **Contextualização:** As questões recuperadas (ex: 5 exemplos de questões sobre as Leis de Newton) são fornecidas ao LLM como contexto ou "exemplos de referência".
4.  **Geração Aumentada:** Com base nesses exemplos de alta qualidade, o LLM gera uma **questão nova e original** que segue o estilo, a dificuldade e o formato das questões reais de vestibular sobre aquele tópico.

## Few-shot

Este sistema utiliza uma ferramenta baseada em RAG para aplicar uma abordagem few-shot na geração de novas questões. Em vez de depender apenas do conhecimento interno do LLM, nossa ferramenta permite que um agente de IA consulte as 5 questões de vestibulares passados mais relacionadas ao prompt do usuário para que uma nova questão possa ser criada. Essa consulta é realizada em um banco de dados especializado que contém milhares de questões, previamente catalogadas por área de conhecimento (ex: Matemática, Física) e subcategoria (ex: Cinemática, Termodinâmica).

A arquitetura é semelhante ao modelo Retrieval-generator, alterando apenas o fluxo de geração, que inclui um pós-processamento da recuperação para gerar um prompt few-shot. Isso é distribuído ao separar o fluxo em duas equipes (crews), onde a segunda recebe os resultados da primeira e incorpora o prompt few-shot no agente gerador.

### Fluxo de Geração

Com essa arquitetura, o agente de IA responsável por criar as questões segue um fluxo de trabalho inteligente:

1.  **Recebe a Solicitação:** O usuário especifica o tema da questão desejada (ex: "uma questão sobre a Segunda Lei de Newton").
2.  **Busca por Similaridade:** A ferramenta converte a solicitação em um embedding e usa o FAISS para buscar em seu vector store as questões semanticamente mais similares.
3.  **Contextualização:** As questões recuperadas (ex: 5 exemplos de questões sobre as Leis de Newton) são fornecidas ao LLM como contexto ou "exemplos de referência".
4.  **Geração Aumentada:** Com base nesses exemplos de alta qualidade, o LLM gera uma **questão nova e original** que segue o estilo, a dificuldade e o formato das questões reais de vestibular sobre aquele tópico.

## Scraping

Este sistema implementa uma abordagem baseada em **Web Scraping**. Em vez de depender apenas do conhecimento interno do LLM, nossa ferramenta permite que um agente de IA consulte sites especializados em assuntos abordados nos vestibulares tradicionais do Brasil. Realizamos buscas via API Serper para encontrar os sites mais relevantes e, em seguida, usamos o Beautiful Soup para extrair o conteúdo HTML, que fornece o texto contextual para nosso agente gerador de questões.

-----

### 1\. Preparação

O processo funciona em duas etapas principais: a **Busca de sites especializados** e a **extração de conteúdo**. Um agente, anterior à etapa de busca e extração, analisa o prompt do usuário para identificar detalhes relevantes — como matéria e subtópico — necessários para guiar a busca. Por exemplo:

```
Quero gerar questões de matemática de probabilidade para vestibulares paulistas
```

Aqui, o agente extrairia **Matemática** como a matéria e **Probabilidade** como o subtópico, e então passaria esses parâmetros para o pipeline de Web Scraping.

### 2\. Busca na Web com Serper

O **Serper** é uma API de busca programática que retorna JSON estruturado — títulos, snippets, URLs e metadados — em vez de HTML bruto destinado a navegadores. Isso permite buscas rápidas, seguras e facilmente integráveis ao nosso pipeline de Web Scraping.

Exemplo de uso em nosso agente:

```python
@agent
def pesquisa_e_extracao(self) -> Agent:
    return Agent(
        config=self.agents_config['pesquisa_e_extracao'],
        tools=[SerperDevTool(), RawParagraphTool()],
        verbose=True,
        memory=True
    )
```

Uma resposta típica da API do Serper se parece com:

```json
{
  "title": "Reações de oxidação – Química Orgânica | Exemplo.edu",
  "snippet": "Na química orgânica, oxidação é definida como ...",
  "link": "[https://exemplo.edu/oxidacao](https://exemplo.edu/oxidacao)",
  "displayed_link": "exemplo.edu"
}
```

Assim que temos essas URLs, nós as fornecemos ao Beautiful Soup para buscar e analisar o HTML bruto.

### 3\. Extração de Conteúdo com Beautiful Soup

  * **O que é o Beautiful Soup?** 🥣
    O Beautiful Soup é uma biblioteca Python para fazer parsing de documentos HTML e XML. Ela converte uma string HTML em uma árvore de objetos Python, facilitando a localização e extração de elementos específicos sem a necessidade de lidar com regex ou com as particularidades do DOM de cada site.

  * **Como usamos no VestAgents:**

    1.  **Carregar o HTML**
        Após obter o HTML bruto (via `web.open`), instanciamos o Beautiful Soup:
        ```python
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html_content, 'html.parser')
        ```
    2.  **Extrair Parágrafos**
        Coletamos todas as tags `<p>`:
        ```python
        paragraphs = [str(p) for p in soup.find_all("p")]
        ```
        Isso resulta em uma lista de strings, cada uma preservando a marcação HTML, acentos e formatação originais.
    3.  **Integração ao Pipeline**
        Esses parágrafos são fornecidos ao agente `pesquisa_e_extracao`, que determina quais trechos são mais relevantes para servir como "passagens de apoio" para o agente de geração de questões. Ao preservar a estrutura original de cada parágrafo, garantimos fidelidade ao material de origem.

-----

Abaixo, um diagrama mostrando como essa abordagem se encaixa em nossa arquitetura geral:

1.  **Recebe a Solicitação**: O usuário especifica o tema da questão desejada (ex: "uma questão de matemática sobre probabilidade").
2.  **Refinamento do Prompt**: Agentes intermediários refinam o prompt inicial para identificar a matéria e o subtópico (ex: "Matemática | Probabilidade").
3.  **Busca e Extração**: Consultamos sites especializados e extraímos texto relevante para o contexto.
4.  **Geração da Questão**: Usando a solicitação do usuário e as passagens extraídas, um agente gera uma questão personalizada no estilo vestibular.

## Paired Crew

Esta abordagem implementa primeiro o agente construtor de few-shot, em seguida, para a segunda equipe, executa inicialmente o agente de web scraping e, por fim, mescla ambos os resultados para gerar uma nova questão.

-----

# Instalação

Certifique-se de que você tenha o Python **3.10 a 3.12** instalado. Este projeto utiliza o [UV](https://docs.astral.sh/uv/) para gerenciamento de dependências e pacotes, oferecendo uma experiência de configuração e execução fluida.

Primeiro, se você ainda não tiver, instale o UV:

```bash
pip install uv
```

Em seguida, a partir da raiz do projeto, instale as dependências:

```bash
crewai install
```

### Personalização

1.  Adicione sua `OPENAI_API_KEY` a um arquivo `.env`.
2.  Modifique `src/raia_agents/config/agents.yaml` para configurar seus agentes.
3.  Modifique `src/raia_agents/config/tasks.yaml` para definir suas tarefas.
4.  Atualize `src/raia_agents/crew.py` para adicionar sua própria lógica, ferramentas ou argumentos.
5.  Atualize `src/raia_agents/main.py` para incluir entradas personalizadas para seus agentes e tarefas.

### Executando o Projeto

Para iniciar sua equipe de agentes de IA e executar as tarefas, rode:

```bash
crewai run
```

Este comando inicializa a equipe `raia_agents` e despacha os agentes de acordo com sua configuração.
