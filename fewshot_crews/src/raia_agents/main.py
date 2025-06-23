import warnings

from raia_agents.crew import RaiaAgents, RaiaRedacaoCrew
import time
warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = [
        {"input_user": "Quero uma questão de ENEM sobre funções exponenciais."},
        #{"input_user": "Me envie uma questão de história sobre a Revolução Francesa."},
        # {"input_user": "Pode gerar uma questão de física sobre leis de Newton?"},
        # {"input_user": "Quero uma questão de biologia sobre fotossíntese."},
        # {"input_user": "Gere uma questão de química sobre ligações químicas."},
        # {"input_user": "Quero uma questão de geografia sobre mudanças climáticas."},
        # {"input_user": "Manda uma questão de sociologia sobre desigualdade social."},
        # {"input_user": "Me dá uma questão de filosofia sobre Sócrates."},
        # {"input_user": "Quero uma questão de português sobre análise sintática."},
        # {"input_user": "Gera uma questão de matemática sobre probabilidade."},
        # {"input_user": "Quero uma questão de física sobre eletromagnetismo."},
        # {"input_user": "Me envie uma questão de história do Brasil sobre a Era Vargas."},
        # {"input_user": "Quero uma questão de biologia sobre genética."},
        # {"input_user": "Pode gerar uma questão de química sobre pH e soluções?"},
        # {"input_user": "Quero uma questão de geografia sobre globalização."},
        # {"input_user": "Me dá uma questão de filosofia sobre empirismo."},
        # {"input_user": "Gere uma questão de português sobre interpretação de texto."},
        # {"input_user": "Quero uma questão de matemática sobre matrizes."},
        # {"input_user": "Manda uma questão de física sobre termodinâmica."},
        # {"input_user": "Preciso de uma questão de sociologia sobre movimentos sociais."},
        # {"input_user": "Me envie uma questão de história sobre a Guerra Fria."},
        # {"input_user": "Quero uma questão de biologia sobre sistema imunológico."},
        # {"input_user": "Gera uma questão de química sobre tabela periódica."},
        # {"input_user": "Quero uma questão de geografia sobre urbanização."},
        # {"input_user": "Me dá uma questão de filosofia sobre ética."},
        # {"input_user": "Manda uma questão de português sobre funções da linguagem."},
        # {"input_user": "Preciso de uma questão de matemática sobre análise combinatória."},
        # {"input_user": "Quero uma questão de física sobre óptica geométrica."},
        # {"input_user": "Me envie uma questão de história sobre o Império Romano."},
        # {"input_user": "Gera uma questão de biologia sobre ciclo do carbono."},
        # {"input_user": "Quero uma questão de química sobre reações de oxirredução."},
        # {"input_user": "Me dá uma questão de geografia sobre migração populacional."},
        # {"input_user": "Pode gerar uma questão de sociologia sobre cultura e identidade?"},
        # {"input_user": "Quero uma questão de filosofia sobre Kant."},
        # {"input_user": "Me envie uma questão de português sobre crase."},
        # {"input_user": "Gere uma questão de matemática sobre geometria espacial."},
        # {"input_user": "Quero uma questão de física sobre movimento uniformemente variado."},
        # {"input_user": "Me dá uma questão de história sobre o Iluminismo."},
        # {"input_user": "Manda uma questão de biologia sobre evolução das espécies."},
        # {"input_user": "Preciso de uma questão de química sobre gases."},
        # {"input_user": "Quero uma questão de geografia sobre fontes de energia renovável."},
        # {"input_user": "Gere uma questão de sociologia sobre trabalho e sociedade."},
        # {"input_user": "Me envie uma questão de filosofia sobre niilismo."},
        # {"input_user": "Quero uma questão de português sobre figuras de linguagem."},
        # {"input_user": "Pode gerar uma questão de matemática sobre estatística descritiva?"},
        # {"input_user": "Me dá uma questão de física sobre leis de Kepler."},
        # {"input_user": "Quero uma questão de história sobre o período colonial brasileiro."},
        # {"input_user": "Manda uma questão de biologia sobre respiração celular."},
        # {"input_user": "Gera uma questão de química sobre equilíbrio químico."},
        # {"input_user": "Preciso de uma questão de geografia sobre biomas brasileiros."},
    ]
    try:
        for idx, request in enumerate(inputs):
            start_time = time.time()
            busca_crew = RaiaAgents(idx, request.get("input_user")).crew()
            busca_crew.kickoff(inputs=request)

            with open(f"./resultados/{idx}/few_shot_prompt.json", encoding="utf-8") as f:
                few_shot_prompt = f.read()

            with open(f"./resultados/{idx}/topicos_questao.json", encoding="utf-8") as f:
                topicos = f.read()

            with open(f"./resultados/{idx}/prompt_melhorado.json", encoding="utf-8") as f:
                solicitacao_melhorada = f.read()

            outputs = {
                "solicitacao_melhorada": f"{solicitacao_melhorada}",
                "topicos": f"{topicos}",
                "few_shot_prompt": f"{few_shot_prompt}",
            }
            redacao_crew = RaiaRedacaoCrew(idx).crew()
            redacao_crew.kickoff(inputs=outputs)
            end_time = time.time()
            
            metrics_busca = busca_crew.usage_metrics
            metrics_redacao = redacao_crew.usage_metrics
            total_tokens = metrics_busca.prompt_tokens + metrics_busca.completion_tokens + metrics_redacao.prompt_tokens + metrics_redacao.completion_tokens
            print("Total tokens used:", total_tokens)
            print(f"Execution time for crew {idx}: {end_time - start_time} seconds")
            
            with open(f"./resultados/{idx}/usage.json", "w") as log_file:
                log_file.write("{\n"+f'"token_usage": {total_tokens},\n')
                log_file.write(f'"execution_time": {end_time - start_time} seconds\n'+ "}")

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")
