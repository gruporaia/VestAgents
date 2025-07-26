"""
This script is used to evaluate the quality of the questions based on the ENEM Matrix.
It uses the GEval library to evaluate the questions.

How to execute:
python evaluate.py <csv_file_name> <outputdir>


Example:

The script is called evaluate.py and the csv file is called enem_questoes_teste.csv. Ommit the .csv.
python evaluate.py enem_questoes_teste enem

The output will be saved in the eval_result/<optional dir name> directory with the name of the csv file.
The output will be a JSON file with the evaluation results.
"""

import os
import json
import argparse
from typing import Optional
from dotenv import load_dotenv
import csv
import pathlib

from deepeval.evaluate.evaluate import assert_test
from deepeval.metrics.g_eval.g_eval import GEval
from deepeval.test_case.llm_test_case import LLMTestCase, LLMTestCaseParams


load_dotenv()
def escape_latex_backslashes(text: str) -> str:
    # Escapes all single backslashes to double backslashes (\\) to avoid JSON issues
    return text.replace("\\", "\\\\") if text else text


eixos_cognitivos_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "eixos_cognitivos — Avalie se a questão envolve ao menos um dos seguintes processos: Dominar Linguagens (DL), Compreender Fenômenos (CF), Enfrentar Situações-Problema (SP), Construir Argumentação (CA), Elaborar Propostas (EP).",
]
habilidade_enem_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "habilidade_enem — A questão deve estar corretamente alinhada a uma competência e habilidade específicas da matriz do ENEM (ex: H15 de Linguagens).",
]
enunciado_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "enunciado — O enunciado deve ser claro, objetivo e contextualizado, sem ambiguidades ou linguagem rebuscada.",
]
alternativa_correta_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "alternativa_correta — A resposta correta deve ser inequívoca, fundamentada e compatível com o conteúdo previsto.",
]
distratores_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "distratores — As alternativas erradas devem ser verossímeis, coerentes com o tema e úteis para avaliar compreensão.",
]
exigencia_cognitiva_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "exigencia_cognitiva — A questão deve exigir interpretação, raciocínio, análise ou aplicação de conceitos (não apenas memorização).",
]
linguagem_inclusiva_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "linguagem_inclusiva — A linguagem deve ser neutra, acessível, livre de vieses ou estereótipos.",
]
sem_erro_conceitual_steps = [
    "Avalie a qualidade da seguinte questão de vestibular gerada por IA com base na Matriz de Referência do ENEM.",
    "sem_erro_conceitual — A questão deve estar livre de erros conceituais, ambiguidade ou imprecisão científica ou histórica.",
]

eixos_cognitivos_metric = GEval(
    name="eixos_cognitivos",
    evaluation_steps=eixos_cognitivos_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
habilidade_enem_metric = GEval(
    name="habilidade_enem",
    evaluation_steps=habilidade_enem_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
enunciado_metric = GEval(
    name="enunciado",
    evaluation_steps=enunciado_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
alternativa_correta_metric = GEval(
    name="alternativa_correta",
    evaluation_steps=alternativa_correta_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
distratores_metric = GEval(
    name="distratores",
    evaluation_steps=distratores_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
exigencia_cognitiva_metric = GEval(
    name="exigencia_cognitiva",
    evaluation_steps=exigencia_cognitiva_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
linguagem_inclusiva_metric = GEval(
    name="linguagem_inclusiva",
    evaluation_steps=linguagem_inclusiva_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)
sem_erro_conceitual_metric = GEval(
    name="sem_erro_conceitual",
    evaluation_steps=sem_erro_conceitual_steps,
    evaluation_params=[
        LLMTestCaseParams.INPUT,
        LLMTestCaseParams.ACTUAL_OUTPUT,
        LLMTestCaseParams.RETRIEVAL_CONTEXT,
    ],
    threshold=0.5,
)


def main(
    csv_file_name: str,
    output_dir: Optional[str] = None,
) -> None:
    """Evaluate questions listed in a CSV file and write JSON results.

    The function expects a CSV file located in ``csvs/``. Each row should contain at least two
    columns: ``input`` (the user prompt/question) and ``output`` (the
    generated question to be evaluated). Optionally, it may include a
    ``retrieval_context`` column with additional context.

    Parameters
    ----------
    csv_file_name : str
        Name of the CSV file *without* the ``.csv`` extension.
    output_dir : Optional[str]
        Directory where the evaluation JSON will be stored. Defaults to
        ``<current_module>/eval_result``.
    """
    if output_dir is not None:
        output_dir = os.path.join(os.path.dirname(__file__), "eval_result", output_dir)
    else:
        output_dir = os.path.join(os.path.dirname(__file__), "eval_result")

    # Normalise paths -------------------------------------------------------------
    csv_path = os.path.join("csvs", f"{csv_file_name}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found at {csv_path}")

    os.makedirs(output_dir, exist_ok=True)

    # Prepare container for all evaluation results --------------------------------
    aggregated_results: list[dict[str, object]] = []
    json_output_path = os.path.join(output_dir, "enem", f"eval_{csv_file_name}.json")
    indexes_to_reavaluate = [5, 15, 17, 22, 24, 39, 45]
    starting_index = 0

    # Read CSV and perform evaluation -------------------------------------------
    with open(csv_path, newline="", encoding="utf-8-sig") as csvfile:
        reader = csv.DictReader(csvfile)

        for idx, row in enumerate(reader):
            if len(indexes_to_reavaluate) == 0 and idx < starting_index:
                print(
                    f"Skipping row {idx + 1} (before starting index {starting_index})"
                )
                continue
            print(f"Processing row {idx + 1}...")
            print(f"Row {row}...")

            if os.path.exists(json_output_path):
                with open(json_output_path, "r", encoding="utf-8") as jf:
                    try:
                        aggregated_results = json.load(jf)
                    except json.JSONDecodeError:
                        aggregated_results = []
            else:
                aggregated_results = []

            if len(indexes_to_reavaluate) > 0:
                if idx not in indexes_to_reavaluate:
                    print(
                        f"Skipping row {idx + 1} as it is not in the re-evaluation list."
                    )
                    continue
                if idx in indexes_to_reavaluate and aggregated_results:
                    for i, result in enumerate(aggregated_results):
                        if result.get("index") == idx:
                            aggregated_results.pop(i)
                            print(f"Re-evaluating result at index {idx}...")
                            break
            user_input = (
                row.get("input")
                or row.get("user_input")
                or row.get("prompt")
                or row.get("question")
            )
            output = (
                row.get("output")
                or row.get("generated")
                or row.get("answer")
                or row.get("question_generated")
            )
            
            output = escape_latex_backslashes(output)
            retrieval_context = row.get("retrieval_context", "")

            if not user_input or not output:
                print(user_input)
                print(output)
                # Skip rows that don't have the essential information.
                continue

            # Build LLM test case ------------------------------------------------
            llm_test_case = LLMTestCase(
                input=user_input,
                actual_output=output,
                expected_output=user_input,  # As per original logic
                retrieval_context=[retrieval_context],
            )

            # Edit the assert_test method inside your virtual environment
            # Click on the method to open it and then add:
            #     ```
            #     return test_result
            #     ```
            # right before:
            #     ```
            #     if not test_result.success:
            #     ```
            # Run evaluation -----------------------------------------------------
            response = assert_test(
                llm_test_case,
                [
                    eixos_cognitivos_metric,
                    habilidade_enem_metric,
                    enunciado_metric,
                    alternativa_correta_metric,
                    distratores_metric,
                    exigencia_cognitiva_metric,
                    linguagem_inclusiva_metric,
                    sem_erro_conceitual_metric,
                ],
            )
            print(response)
            # Collect metrics ----------------------------------------------------
            metrics_info = []
            for metric in response.metrics_data:
                try:
                    parsed_details = json.loads(metric.reason)
                except Exception:
                    # In case JSON parsing fails, store raw text
                    parsed_details = {"raw_reason": metric.reason}

                metrics_info.append(
                    {
                        "metric_name": metric.name,
                        "score": metric.score,
                        "details": parsed_details,
                    }
                )

            current_result = {
                "index": (
                    len(aggregated_results)
                    if len(indexes_to_reavaluate) == 0 and starting_index == 0
                    else idx
                ),
                "input": user_input,
                "output": output,
                "metrics": metrics_info,
            }
            aggregated_results.append(current_result)
            aggregated_results.sort(key=lambda x: x.get("index", 0))
            with open(json_output_path, "w", encoding="utf-8") as jf:
                json.dump(aggregated_results, jf, ensure_ascii=False, indent=2)

            print(f"Updated results in: {json_output_path}")

    print(f"Evaluation finished. Results saved to: {json_output_path}")


# -----------------------------------------------------------------------------
# CLI ENTRY POINT
# -----------------------------------------------------------------------------


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate ENEM-style questions listed in a CSV file.",
    )
    parser.add_argument(
        "csv_file_name",
        help="CSV file name (without extension) located in the 'csvs' directory.",
    )
    parser.add_argument(
        "--output_dir",
        default=None,
        help="Directory to write evaluation JSON files. Defaults to <module>/eval_result.",
    )

    args = parser.parse_args()

    # Remove potential '.csv' extension from the provided name -------------------
    file_stem = os.path.splitext(args.csv_file_name)[0]

    main(file_stem, output_dir=args.output_dir)
