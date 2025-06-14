from typing import List

from crewai import LLM, Agent, Crew, Process, Task
from crewai.agents.agent_builder.base_agent import BaseAgent
from crewai.project import CrewBase, agent, crew, task

from raia_agents.tools.RAGTool import SingleRagTool

llm = LLM(
    model="openai/gpt-4o", temperature=0.3, seed=42  # call model by provider/model_name
)

# If you want to run a snippet of code before or after the crew starts,
# you can use the @before_kickoff and @after_kickoff decorators
# https://docs.crewai.com/concepts/crews#example-crew-class-with-decorators


@CrewBase
class RaiaAgents:
    """RaiaAgents crew"""

    agents: List[BaseAgent]
    tasks: List[Task]

    def __init__(self, question_id: int):
        self.question_id = question_id
        self.agents_config = "config/agents.yaml"
        self.tasks_config = "config/tasks.yaml"

    # Learn more about YAML configuration files here:
    # Agents: https://docs.crewai.com/concepts/agents#yaml-configuration-recommended
    # Tasks: https://docs.crewai.com/concepts/tasks#yaml-configuration-recommended

    # If you would like to add tools to your agents, you can learn more about it here:
    # https://docs.crewai.com/concepts/agents#agent-tools

    @agent
    def melhorador_perguntas(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["melhorador_perguntas"],
            tools=[],
            verbose=True,
        )

    @agent
    def identificador_topicos(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["identificador_topicos"],
            tools=[],
            verbose=True,
        )

    @agent
    def redator_questao(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["redator_questao"],
            tools=[SingleRagTool()],
            verbose=True,
        )

    @agent
    def resolutor_questao(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["resolutor_questao"],
            tools=[],
            verbose=True,
        )

    @agent
    def revisor_questao(self) -> Agent:
        return Agent(  # type: ignore
            config=self.agents_config["revisor_questao"],
            tools=[],
            verbose=True,
        )

    # To learn more about structured task outputs,
    # task dependencies, and task callbacks, check out the documentation:
    # https://docs.crewai.com/concepts/tasks#overview-of-a-task

    @task
    def melhorar_pergunta(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["melhorar_pergunta"],
            output_file=f"resultados/{self.question_id}/prompt_melhorado.json",
        )

    @task
    def filtrar_topicos(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["filtrar_topicos"],
            output_file=f"resultados/{self.question_id}/topicos_questao.json",
        )

    @task
    def verificação_ferramenta(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["verificação_ferramenta"],
            output_file=f"resultados/{self.question_id}/resultado_ferramentas.json",
        )

    @task
    def geração_questao(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["geração_questao"],
            output_file=f"resultados/{self.question_id}/questao_gerada.json",
        )

    @task
    def resolucao_questao(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["resolucao_questao"],
            output_file=f"resultados/{self.question_id}/resolucao.json",
        )

    @task
    def correção_questao(self) -> Task:
        return Task(  # type: ignore
            config=self.tasks_config["correção_questao"],
            output_file=f"resultados/{self.question_id}/questao_corrigida.json",
        )

    @crew
    def crew(self) -> Crew:
        """Creates the RaiaAgents crew"""
        # To learn how to add knowledge sources to your crew, check out the documentation:
        # https://docs.crewai.com/concepts/knowledge#what-is-knowledge

        return Crew(
            agents=self.agents,  # Automatically created by the @agent decorator
            tasks=self.tasks,  # Automatically created by the @task decorator
            process=Process.sequential,
            verbose=True,
            output_log_file="logs.json",
            # memory=True
            # process=Process.hierarchical, # In case you wanna use that instead https://docs.crewai.com/how-to/Hierarchical/
        )
