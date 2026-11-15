from typing import Dict

from workflow.system_state import SystemState
from workflow.agents.tool import Tool


class Agent:
    """
    Base class for agents. CHESS let an LLM pick each next tool from the tool names alone (the state is
    never shown to it), which amounts to the fixed order below at the cost of an extra call per step;
    the tools now run in their listed order.
    """

    def __init__(self, name: str, task: str, config: dict):
        self.name = name
        self.task = task
        self.config = config
        self.tools: Dict[str, Tool] = {}

    def workout(self, system_state: SystemState) -> SystemState:
        for tool in self.tools.values():
            tool(system_state)
        return system_state

    def __call__(self, system_state: SystemState) -> SystemState:
        return self.workout(system_state)
