from pydantic import BaseModel


class Task(BaseModel):
    """
    A question to answer. CHESS's gold SQL and difficulty fields are removed so the gold query can never
    reach the workflow; evidence holds the BEAVER hints (empty in setting 0).
    """
    question_id: str
    db_id: str
    question: str
    evidence: str
