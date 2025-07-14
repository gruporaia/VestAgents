
# VestAgents

Este projeto visa criar uma ferramenta de geração de questões inéditas para vestibulares brasileiros utilizando Large Language Models e sistemas Multiagentes (LLMs). O desenvolvimento conta com o apoio da empresa de educação a distância [PICO](https://www.usepico.com.br/) e explora alternativas à ferramenta de geração de questões já existente na empresa.

## ⚙️ Funcionamento

O VestAgents implementa quatro estratégias complementares para geração de questões:

- **Retrieval-generator**: Sistema com RAG (Geração Aumentada por Recuperação) integrado para pesquisar questões semelhantes em uma base vetorial.
- **Few-shot**: Geração baseada em prompt few-shot com as 5 questões mais semelhantes (derivado do módulo de retrieval).
- **Scraping**: Realiza busca na web via Serper API e extrai textos com BeautifulSoup como base para a geração.
- **Paired Crew**: Combina few-shot e scraping em sequência para geração aumentada com múltiplas fontes.

### Arquitetura Geral

#### Retrieval-generator

Utiliza embeddings e FAISS para construir e consultar um banco de dados vetorial de questões passadas. Isso permite que o modelo tenha exemplos semanticamente semelhantes como contexto para gerar novas questões.

#### Few-shot

Extensão do retrieval: após buscar questões semelhantes, constrói-se um prompt few-shot que serve de base para o modelo gerar uma nova questão.

#### Scraping

Busca conteúdos relevantes na web sobre o tema solicitado (como "Probabilidade no Enem"), extrai parágrafos de páginas especializadas e os insere como contexto para geração.

#### Paired Crew

Estrutura em duas crews: uma para recuperar exemplos (few-shot), outra para buscar conteúdo online (scraping). Os dois conjuntos são unidos para criar a questão final.

## 🛠️ Como rodar

### Requisitos

- Python 3.10 a 3.12
- [UV](https://docs.astral.sh/uv/) como gerenciador de pacotes

### Instalando dependências

```bash
pip install uv
crewai install
```

### Configurando

1. Entre no diretório do projeto que deseja testar. 
Exemplo:
```bash
cd fewshot_crews
```
2. Crie o arquivo `.env` com sua `OPENAI_API_KEY`.
3. Configure os agentes em:  
   `src/raia_agents/config/agents.yaml`
4. Configure as tarefas em:  
   `src/raia_agents/config/tasks.yaml`
5. Edite `src/raia_agents/crew.py` e `src/raia_agents/main.py` para customizações.

### Executando o projeto

```bash
crewai run
```

Esse comando inicia a crew de agentes conforme definido em `main.py`.

## 📊 Resultados

O VestAgents já foi utilizado para gerar centenas de questões inéditas com qualidade semelhante a vestibulares tradicionais. Entre os diferenciais observados:

- Maior alinhamento com o estilo do Enem e de vestibulares paulistas.
- Diversidade de tópicos e níveis de dificuldade.
- Geração contextualizada com exemplos e textos atualizados.

## 📑 Referências

- FAISS - Facebook AI Similarity Search  
- Serper API - Busca estruturada na web  
- Beautiful Soup - Extração de conteúdo HTML  
- [PICO](https://www.usepico.com.br/)

## 💻 Quem somos

| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) | Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](https://instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

### Desenvolvedores

- **Álvaro Lopes**
- **Artur de Vlieger**
- **Fabrício Salomon**
- **Leticia Bossatto Marchezi**  
- **Luis Felipe Jorge**
- **Luísa Shimabucoro**
- **Otávio Coletti**
- **Pedro Monteiro**
