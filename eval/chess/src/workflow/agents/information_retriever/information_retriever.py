from workflow.agents.agent import Agent

from workflow.agents.information_retriever.tool_kit.extract_keywords import ExtractKeywords
from workflow.agents.information_retriever.tool_kit.retrieve_entity import RetrieveEntity


class InformationRetriever(Agent):
    """
    Agent responsible for retrieving relevant entities from the question and hint.
    CHESS's retrieve_context tool (vector search over column descriptions) is removed: BEAVER has no column descriptions.
    """

    def __init__(self, config: dict):
        """Initialize the tools needed for information retrieval"""
        super().__init__(
            name="Information Retriever",
            task="retrieve the most important entities relevant to the keywords of the question, through extracting keywords and retrieving entities",
            config=config
        )

        self.tools = {
            "extract_keywords": ExtractKeywords(**config["tools"]["extract_keywords"]),
            "retrieve_entity": RetrieveEntity(**config["tools"]["retrieve_entity"]),
        }
