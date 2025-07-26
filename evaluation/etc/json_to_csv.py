import json
import pandas as pd

def convert_json_to_csv(json_file_path, csv_file_path):
    """
    Lê um arquivo JSON com uma estrutura aninhada, o achata e o converte para um arquivo CSV.

    Args:
        json_file_path (str): O caminho para o arquivo JSON de entrada.
        csv_file_path (str): O caminho para o arquivo CSV de saída.
    """
    try:
        
        metric_names = [
            "Eixos cognitivos",
            "Habilidades específicas",
            "Qualidade do enunciado",
            "Resposta fundamentada",
            "Distratores coerentes",
            "Exigencia cognitiva",
            "Linguagem Inclusiva",
            "Sem erro conceitual"
        ]
        # Etapa 1: Ler o arquivo JSON
        with open(json_file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        # Etapa 2: Processar os dados JSON e achatá-los
        processed_data = []
        # Itera sobre cada questão no arquivo JSON
        for item in data:
            # Para cada questão, itera sobre a lista de métricas
            for i, metric in enumerate(item['metrics']):
                metric_name = metric_names[i % len(metric_names)]
                # Cria um dicionário para cada linha do CSV
                row = {
                    'index': item.get('index'),
                    'input': item.get('input'),
                    'output': item.get('output'),
                    'metric_name': metric_name,
                    'score': int(metric.get('score')*100),
                    # Extrai o 'raw_reason' do dicionário aninhado 'details'
                    'raw_reason': metric.get('details', {}).get('raw_reason')
                }
                processed_data.append(row)

        # Etapa 3: Criar um DataFrame do pandas a partir dos dados processados
        df = pd.DataFrame(processed_data)

        # Etapa 4: Salvar o DataFrame em um arquivo CSV
        # 'encoding='utf-8-sig'' é usado para garantir a compatibilidade com caracteres especiais
        # 'index=False' evita que o pandas escreva o índice do DataFrame no arquivo
        df.to_csv(csv_file_path, index=False, encoding='utf-8-sig')

        print(f"Arquivo '{json_file_path}' convertido com sucesso para '{csv_file_path}'")

    except FileNotFoundError:
        print(f"Erro: O arquivo '{json_file_path}' não foi encontrado.")
    except json.JSONDecodeError:
        print(f"Erro: O arquivo '{json_file_path}' não é um JSON válido.")
    except Exception as e:
        print(f"Ocorreu um erro inesperado: {e}")

# --- Execução do Script ---
if __name__ == '__main__':
    # Defina o nome do seu arquivo JSON de entrada aqui
    # Certifique-se de que 'evalenem_questions.json' está na mesma pasta que este script
    # ou forneça o caminho completo para o arquivo.
    output_dir = 'enem'
    input_json_file = f'./eval_result/{output_dir}/eval_{output_dir}_questions.json'

    # Defina o nome do arquivo CSV de saída desejado
    output_csv_file = f'./eval_result/{output_dir}/eval_{output_dir}_questions_output.csv'

    # Chama a função para realizar a conversão
    convert_json_to_csv(input_json_file, output_csv_file)
