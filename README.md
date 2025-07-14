
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

### Próximos passos

- Integração com interface gráfica via Streamlit.
- Avaliação com professores parceiros da PICO.
- Implementação de avaliação automática da qualidade da questão gerada.
- Inclusão de suporte a diferentes idiomas e níveis educacionais.

## 📑 Referências

- [FAISS - Facebook AI Similarity Search](https://engineering.fb.com/2017/03/29/data-infrastructure/faiss-a-library-for-efficient-similarity-search/)  
- [Serper API - Busca estruturada na web](https://serper.dev/)  
- [Beautiful Soup - Extração de conteúdo HTML](https://pypi.org/project/beautifulsoup4/)
- [CrewAI](https://www.crewai.com/)  
- [PICO](https://www.usepico.com.br/)

## 💻 Quem somos

| ![LogoRAIA](https://github.com/user-attachments/assets/ce3f8386-a900-43ff-af84-adce9c17abd2) | Este projeto foi desenvolvido pelos membros do **RAIA (Rede de Avanço de Inteligência Artificial)**, uma iniciativa estudantil do Instituto de Ciências Matemáticas e de Computação (ICMC) da USP - São Carlos. Somos estudantes que compartilham o objetivo de criar soluções inovadoras utilizando inteligência artificial para impactar positivamente a sociedade. Para saber mais, acesse [nosso site](https://gruporaia.vercel.app/) ou [nosso Instagram](https://instagram.com/grupo.raia)! |
|------------------|-------------------------------------------|

## 💻 Patrocínio
| <img width="240" height="240" alt="pico" src="https://github.com/user-attachments/assets/bf9fb6e9-d978-409a-9cbb-fc38538c0c5a" /> | Este projeto foi desenvolvido com a Pico, uma plataforma de aprendizado digital com gameficação, que implementa inteligência artificial para estimular o aprendizado. Para saber mais, acesse [use Pico](https://www.usepico.com.br/) |
|------------------|-------------------------------------------|


### Desenvolvedores

- **Álvaro Lopes**. [Linkedin](https://www.linkedin.com/in/alvaro-jose-lopes/) e [Github](https://github.com/AlvaroJoseLopes)
- **Artur de Vlieger**  [Linkedin](https://www.linkedin.com/in/artur-de-vlieger-336829252/) e [Github](https://github.com/Deflyer)
- **Fabrício Salomon** [Linkedin](https://www.linkedin.com/in/fabr%C3%ADcio-salomon/) e [Github](https://github.com/FabricioLRSalomon)
- **Leticia Bossatto Marchezi**   [Linkedin](https://www.linkedin.com/in/letmarchezi/) e [Github](https://github.com/letMarchezi)
- **Luis Felipe Jorge** [Linkedin](https://www.linkedin.com/in/luis-felipe-jorge/) e [Github](https://github.com/LuisFelipeJorge)
- **Otávio Coletti** [Linkedin](https://www.linkedin.com/in/ot%C3%A1viocoletti-012/) e [Github](https://github.com/otaviofcoletti)
