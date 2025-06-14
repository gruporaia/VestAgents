#!/usr/bin/env python
import sys
import warnings

from raia_agents.crew import RaiaAgents, RaiaRedacaoCrew

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """
    inputs = [
        {"input_user": "Quero uma questão de ENEM sobre funções exponenciais."},
        {"input_user": "Me envie uma questão de história sobre a Revolução Francesa."},
        {"input_user": "Pode gerar uma questão de física sobre leis de Newton?"},
        {"input_user": "Quero uma questão de biologia sobre fotossíntese."},
        {"input_user": "Gere uma questão de química sobre ligações químicas."},
        {"input_user": "Quero uma questão de geografia sobre mudanças climáticas."},
        {"input_user": "Manda uma questão de sociologia sobre desigualdade social."},
        {"input_user": "Me dá uma questão de filosofia sobre Sócrates."},
        {"input_user": "Quero uma questão de português sobre análise sintática."},
        {"input_user": "Gera uma questão de matemática sobre probabilidade."},
        {"input_user": "Quero uma questão de física sobre eletromagnetismo."},
        {"input_user": "Me envie uma questão de história do Brasil sobre a Era Vargas."},
        {"input_user": "Quero uma questão de biologia sobre genética."},
        {"input_user": "Pode gerar uma questão de química sobre pH e soluções?"},
        {"input_user": "Quero uma questão de geografia sobre globalização."},
        {"input_user": "Me dá uma questão de filosofia sobre empirismo."},
        {"input_user": "Gere uma questão de português sobre interpretação de texto."},
        {"input_user": "Quero uma questão de matemática sobre matrizes."},
        {"input_user": "Manda uma questão de física sobre termodinâmica."},
        {"input_user": "Preciso de uma questão de sociologia sobre movimentos sociais."},
        {"input_user": "Me envie uma questão de história sobre a Guerra Fria."},
        {"input_user": "Quero uma questão de biologia sobre sistema imunológico."},
        {"input_user": "Gera uma questão de química sobre tabela periódica."},
        {"input_user": "Quero uma questão de geografia sobre urbanização."},
        {"input_user": "Me dá uma questão de filosofia sobre ética."},
        {"input_user": "Manda uma questão de português sobre funções da linguagem."},
        {"input_user": "Preciso de uma questão de matemática sobre análise combinatória."},
        {"input_user": "Quero uma questão de física sobre óptica geométrica."},
        {"input_user": "Me envie uma questão de história sobre o Império Romano."},
        {"input_user": "Gera uma questão de biologia sobre ciclo do carbono."},
        {"input_user": "Quero uma questão de química sobre reações de oxirredução."},
        {"input_user": "Me dá uma questão de geografia sobre migração populacional."},
        {"input_user": "Pode gerar uma questão de sociologia sobre cultura e identidade?"},
        {"input_user": "Quero uma questão de filosofia sobre Kant."},
        {"input_user": "Me envie uma questão de português sobre crase."},
        {"input_user": "Gere uma questão de matemática sobre geometria espacial."},
        {"input_user": "Quero uma questão de física sobre movimento uniformemente variado."},
        {"input_user": "Me dá uma questão de história sobre o Iluminismo."},
        {"input_user": "Manda uma questão de biologia sobre evolução das espécies."},
        {"input_user": "Preciso de uma questão de química sobre gases."},
        {"input_user": "Quero uma questão de geografia sobre fontes de energia renovável."},
        {"input_user": "Gere uma questão de sociologia sobre trabalho e sociedade."},
        {"input_user": "Me envie uma questão de filosofia sobre niilismo."},
        {"input_user": "Quero uma questão de português sobre figuras de linguagem."},
        {"input_user": "Pode gerar uma questão de matemática sobre estatística descritiva?"},
        {"input_user": "Me dá uma questão de física sobre leis de Kepler."},
        {"input_user": "Quero uma questão de história sobre o período colonial brasileiro."},
        {"input_user": "Manda uma questão de biologia sobre respiração celular."},
        {"input_user": "Gera uma questão de química sobre equilíbrio químico."},
        {"input_user": "Preciso de uma questão de geografia sobre biomas brasileiros."},
    ]
    try:
        for idx, request in enumerate(inputs):

            RaiaAgents(idx, request.get("input_user")).crew().kickoff(inputs=request)

            f = open(f"./resultados/{idx}/selecionar_questoes.txt")
            questoes_selecionadas = f.read()
            f.close()

            f = open(f"./resultados/{idx}/topicos_questao.json")
            topicos = f.read()
            f.close()

            f = open(f"./resultados/{idx}/prompt_melhorado.json")
            questao_melhorada = f.read()
            f.close()
            outputs = {
                "questao_melhorada": f"{questao_melhorada}",
                "topicos": f"{topicos}",
                "input_user": f"Com base nesses exemplos \n {questoes_selecionadas} \n Gere uma questão para mim seguindo os exemplos e a temática.",
            }
            RaiaRedacaoCrew(idx).crew().kickoff(inputs=outputs)

    except Exception as e:
        raise Exception(f"An error occurred while running the crew: {e}")

        RaiaAgents().crew().test(
            n_iterations=int(sys.argv[1]), eval_llm=sys.argv[2], inputs=inputs
        )
