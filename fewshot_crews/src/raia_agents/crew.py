from typing import List

from crewai import Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from src.raia_agents.tools.custom_tool import RetrieveQuestoesTool


@CrewBase
class RaiaAgents:
    """RaiaAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, question_id, prompt):
        self.question_id = question_id
        self.prompt = prompt

    agents_config = "config/agents.yaml"
    tasks_config = "config/tasks.yaml"

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
    def few_shot_agent(self) -> Agent:
        return Agent(
            config=self.agents_config["few_shot_agent"],
            tools=[RetrieveQuestoesTool(result_as_answer=True)],
            verbose=True,
        )

    @task
    def melhorar_pergunta(self) -> Task:
        return Task(  # type: ignore
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
    def selecionar_questoes(self) -> Task:
        return Task(
            config=self.tasks_config["selecionar_questoes"],
            output_file=f"resultados/{self.question_id}/resultado_ferramenta.json",
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


@CrewBase
class RaiaRedacaoCrew:
    agents: List[BaseAgent]
    tasks: List[Task]
    agents_config = "config/agents2.yaml"
    tasks_config = "config/tasks2.yaml"

    def __init__(self, question_id):
        self.question_id = question_id

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
            config=self.agents_config["revisor_questao"],
            tools=[],
            verbose=True,
        )

    @task
    def geração_questao(self) -> Task:
        return Task(
            config=self.tasks_config["geração_questao"],
            output_file=f"resultados/{self.question_id}/questao_gerada.json",
            context=[],
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
        return Crew(
            agents=self.agents,
            tasks=self.tasks,
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.json",
        )
