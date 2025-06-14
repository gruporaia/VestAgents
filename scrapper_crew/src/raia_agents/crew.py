from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task
from src.raia_agents.tools.RawParagraphTool import RawParagraphTool
from src.raia_agents.tools.Serper import BlacklistSerperDevTool


@CrewBase
class RaiaAgents:
    """RaiaAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, question_id: int):
        self.question_id = question_id
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"

    @agent
    def melhorador_perguntas(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["melhorador_perguntas"],
            tools=[],
            verbose=True,
        )

    @agent
    def identificador_topicos(self) -> Agent:
        return Agent(
            config=self.agents_config["identificador_topicos"],
            tools=[],
            verbose=True,
        )

    @agent
    def pesquisa_e_extracao(self) -> Agent:
        return Agent(
            config=self.agents_config["pesquisa_e_extracao"],
            # Substituímos SerperDevTool() por BlacklistSerperDevTool()
            tools=[BlacklistSerperDevTool(), RawParagraphTool()],
            verbose=True,
        )

    @agent
    def redator_questao(self) -> Agent:
        return Agent(
            config=self.agents_config["redator_questao"],
            tools=[],
            verbose=True,
        )

    @agent
    def resolutor_questao(self) -> Agent:
        return Agent(
            config=self.agents_config["resolutor_questao"],
            tools=[],
            verbose=True,
        )

    @agent
    def revisor_questao(self) -> Agent:
        return Agent(
            config=self.agents_config["revisor_questao"], tools=[], verbose=True
        )

    @task
    def melhorar_pergunta(self) -> Task:
        return Task(
            config=self.tasks_config["melhorar_pergunta"],
            output_file=f"resultados/{self.question_id}/prompt_melhorado.json",
        )

    @task
    def filtrar_topicos(self) -> Task:
        return Task(
            config=self.tasks_config["filtrar_topicos"],
            output_file=f"resultados/{self.question_id}/topicos_questao.json",
        )

    @task
    def pesquisa_questao(self) -> Task:
        return Task(
            config=self.tasks_config["buscar_dados_para_questao"],
            output_file=f"resultados/{self.question_id}/buscar_dados_para_questao.json",
        )

    @task
    def geração_questao(self) -> Task:
        return Task(
            config=self.tasks_config["geração_questao"],
            output_file=f"resultados/{self.question_id}/questao_gerada.json",
        )

    @task
    def resolucao_questao(self) -> Task:
        return Task(
            config=self.tasks_config["resolucao_questao"],
            output_file=f"resultados/{self.question_id}/resolucao.json",
        )

    @task
    def correção_questao(self) -> Task:
        return Task(
            config=self.tasks_config["correção_questao"],
            output_file=f"resultados/{self.question_id}/questao_corrigida.json",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RaiaAgents crew"""
        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.json",
        )
