#!/usr/bin/env python
import json
import warnings

from raia_agents.crew import RaiaAgents

warnings.filterwarnings("ignore", category=SyntaxWarning, module="pysbd")


def run():
    """
    Run the crew.
    """

    questions = [
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
        "Preciso de uma questão de geografia sobre biomas brasileiros.",
    ]
    inputs = []
    for question in questions:
        inputs.append(
            {
                "input_user": question,
            }
        )

    for idx, question in enumerate(inputs):
        RaiaAgents(question_id=idx).crew().kickoff(inputs=question)
