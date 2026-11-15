import difflib
from typing import List, Dict, Any, Tuple, Optional

import numpy as np

from llm.engine import embed
from runner.database_manager import DatabaseManager
from workflow.system_state import SystemState
from workflow.agents.tool import Tool

class RetrieveEntity(Tool):
    """
    Tool for retrieving entities and columns similar to given keywords from the question and hint.
    """

    def __init__(self, embedding_model: str):
        super().__init__()
        # CHESS used OpenAI text-embedding-3-small; the model is configurable here (Qwen3-Embedding-8B by default)
        self.embedding_model = embedding_model
        self.edit_distance_threshold = 0.3
        self.embedding_similarity_threshold = 0.6

        self.retrieved_entities = []

    def _run(self, state: SystemState):
        """
        Executes the entity retrieval process.

        Args:
            state (SystemState): The current system state.
        """

        state.similar_columns = self._get_similar_columns(keywords=state.keywords, question=state.task.question, hint=state.task.evidence)

        state.schema_with_examples = self._get_similar_entities(keywords=state.keywords)

    ### Column name similarity ###

    def _get_similar_columns(self, keywords: List[str], question: str, hint: str) -> Dict[str, List[str]]:
        """
        Finds columns similar to given keywords based on question and hint.

        Args:
            keywords (List[str]): The list of keywords.
            question (str): The question string.
            hint (str): The hint string.

        Returns:
            Dict[str, List[str]]: A dictionary mapping table names to lists of similar column names.
        """
        selected_columns = {}
        similar_columns = self._get_similar_column_names(keywords=keywords, question=question, hint=hint)
        for table_name, column_name in similar_columns:
            if table_name not in selected_columns:
                selected_columns[table_name] = []
            if column_name not in selected_columns[table_name]:
                selected_columns[table_name].append(column_name)
        return selected_columns

    def _column_value(self, string: str) -> Tuple[Optional[str], Optional[str]]:
        """
        Splits a string into column and value parts if it contains '='.

        Args:
            string (str): The string to split.

        Returns:
            Tuple[Optional[str], Optional[str]]: The column and value parts.
        """
        if "=" in string:
            left_equal = string.find("=")
            first_part = string[:left_equal].strip()
            second_part = string[left_equal + 1:].strip() if len(string) > left_equal + 1 else None
            return first_part, second_part
        return None, None

    def _extract_paranthesis(self, string: str) -> List[str]:
        """
        Extracts strings within parentheses from a given string.

        Args:
            string (str): The string to extract from.

        Returns:
            List[str]: A list of strings within parentheses.
        """
        paranthesis_matches = []
        open_paranthesis = []
        for i, char in enumerate(string):
            if char == "(":
                open_paranthesis.append(i)
            elif char == ")" and open_paranthesis:
                start = open_paranthesis.pop()
                found_string = string[start:i + 1]
                if found_string:
                    paranthesis_matches.append(found_string)
        return paranthesis_matches

    def _does_keyword_match_column(self, keyword: str, column_name: str, threshold: float = 0.9) -> bool:
        """
        Checks if a keyword matches a column name based on similarity.

        Args:
            keyword (str): The keyword to match.
            column_name (str): The column name to match against.
            threshold (float, optional): The similarity threshold. Defaults to 0.9.

        Returns:
            bool: True if the keyword matches the column name, False otherwise.
        """
        keyword = keyword.lower().replace(" ", "").replace("_", "").rstrip("s")
        column_name = column_name.lower().replace(" ", "").replace("_", "").rstrip("s")
        similarity = difflib.SequenceMatcher(None, column_name, keyword).ratio()
        return similarity >= threshold

    def _get_similar_column_names(self, keywords: str, question: str, hint: str) -> List[Tuple[str, str]]:
        """
        Finds column names similar to given keywords based on question and hint.

        Args:
            keywords (str): The list of keywords.
            question (str): The question string.
            hint (str): The hint string.

        Returns:
            List[Tuple[str, str]]: A list of tuples containing table and column names.
        """
        potential_column_names = []
        for keyword in keywords:
            keyword = keyword.strip()
            potential_column_names.append(keyword)

            column, value = self._column_value(keyword)
            if column:
                potential_column_names.append(column)

            potential_column_names.extend(self._extract_paranthesis(keyword))

            if " " in keyword:
                potential_column_names.extend(part.strip() for part in keyword.split())
        schema = DatabaseManager().get_db_schema()

        # CHESS also embedded every column and the question+hint here to sort the matches by similarity,
        # then returned them as an unordered set, so the embedding call is skipped
        similar_column_names = []
        for table, columns in schema.items():
            for column in columns:
                for potential_column_name in potential_column_names:
                    if self._does_keyword_match_column(potential_column_name, column):
                        similar_column_names.append((table, column))

        table_column_pairs = list(set(similar_column_names))
        return table_column_pairs

    ### Entity similarity ###

    def _get_similar_entities(self, keywords: List[str]) -> Dict[str, Dict[str, List[str]]]:
        """
        Retrieves similar entities from the database based on keywords.

        Args:
            keywords (List[str]): The list of keywords.

        Returns:
            Dict[str, Dict[str, List[str]]]: A dictionary mapping table and column names to similar entities.
        """
        to_seartch_values = self._get_to_search_values(keywords)
        similar_entities_via_LSH = self._get_similar_entities_via_LSH(to_seartch_values)
        similar_entities_via_edit_distance = self._get_similar_entities_via_edit_distance(similar_entities_via_LSH)
        similar_entities_via_embedding = self._get_similar_entities_via_embedding(similar_entities_via_edit_distance)

        selected_values = {}
        for entity in similar_entities_via_embedding:
            table_name = entity["table_name"]
            column_name = entity["column_name"]
            if table_name not in selected_values:
                selected_values[table_name] = {}
            if column_name not in selected_values[table_name]:
                selected_values[table_name][column_name] = []
            selected_values[table_name][column_name].append(entity)
        for table_name, column_values in selected_values.items():
            for column_name, values in column_values.items():
                max_edit_distance_similarity = max(entity["edit_distance_similarity"] for entity in values)
                values = [entity for entity in values if entity["edit_distance_similarity"] >= 0.9*max_edit_distance_similarity]
                max_embedding_similarity = max(entity["embedding_similarity"] for entity in values)
                selected_values[table_name][column_name] = [entity['similar_value'] for entity in values if entity["embedding_similarity"] >= 0.9*max_embedding_similarity]

        return selected_values

    def _get_to_search_values(self, keywords: List[str]) -> List[str]:
        """
        Extracts values to search from the keywords.

        Args:
            keywords (List[str]): The list of keywords.

        Returns:
            List[str]: A list of values to search.
        """
        def get_substring_packet(keyword: str, substring: str) -> Dict[str, str]:
            return {"keyword": keyword, "substring": substring}

        to_search_values = []
        for keyword in keywords:
            keyword = keyword.strip()
            to_search_values.append(get_substring_packet(keyword, keyword))
            if " " in keyword:
                for i in range(len(keyword)):
                    if keyword[i] == " ":
                        first_part = keyword[:i]
                        second_part = keyword[i+1:]
                        to_search_values.append(get_substring_packet(keyword, first_part))
                        to_search_values.append(get_substring_packet(keyword, second_part))
            hint_column, hint_value = self._column_value(keyword)
            if hint_value:
                to_search_values.append(get_substring_packet(keyword, hint_value))
        to_search_values.sort(key=lambda x: (x["keyword"], len(x["substring"]), x["substring"]), reverse=True)
        return to_search_values

    def _get_similar_entities_via_LSH(self, substring_packets: List[Dict[str, str]]) -> List[Dict[str, Any]]:
        similar_entities_via_LSH = []
        for packet in substring_packets:
            keyword = packet["keyword"]
            substring = packet["substring"]
            unique_similar_values = DatabaseManager().query_lsh(keyword=substring, signature_size=100, top_n=10)
            for table_name, column_values in unique_similar_values.items():
                for column_name, values in column_values.items():
                    for value in values:
                        similar_entities_via_LSH.append({"keyword": keyword,
                                                "substring": substring,
                                                "table_name": table_name,
                                                "column_name": column_name,
                                                "similar_value": value})
        return similar_entities_via_LSH

    def _get_similar_entities_via_edit_distance(self, similar_entities_via_LSH: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        similar_entities_via_edit_distance_similarity = []
        for entity_packet in similar_entities_via_LSH:
            edit_distance_similarity = difflib.SequenceMatcher(None, entity_packet["substring"].lower(), entity_packet["similar_value"].lower()).ratio()
            if edit_distance_similarity >= self.edit_distance_threshold:
                entity_packet["edit_distance_similarity"] = edit_distance_similarity
                similar_entities_via_edit_distance_similarity.append(entity_packet)
        return similar_entities_via_edit_distance_similarity

    def _get_similar_entities_via_embedding(self, similar_entities_via_edit_distance: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        similar_values_dict = {}
        to_embed_strings = []
        for entity_packet in similar_entities_via_edit_distance:
            keyword = entity_packet["keyword"]
            substring = entity_packet["substring"]
            similar_value = entity_packet["similar_value"]
            if keyword not in similar_values_dict:
                similar_values_dict[keyword] = {}
            if substring not in similar_values_dict[keyword]:
                similar_values_dict[keyword][substring] = []
                to_embed_strings.append(substring)
            similar_values_dict[keyword][substring].append(entity_packet)
            to_embed_strings.append(similar_value)

        if not to_embed_strings:
            return []
        all_embeddings = embed(to_embed_strings, self.embedding_model, step=self.tool_name)
        similar_entities_via_embedding_similarity = []
        index = 0
        for keyword, substring_dict in similar_values_dict.items():
            for substring, entity_packets in substring_dict.items():
                substring_embedding = all_embeddings[index]
                index += 1
                similar_values_embeddings = all_embeddings[index:index+len(entity_packets)]
                index += len(entity_packets)
                similarities = np.dot(similar_values_embeddings, substring_embedding)
                for i, entity_packet in enumerate(entity_packets):
                    if similarities[i] >= self.embedding_similarity_threshold:
                        entity_packet["embedding_similarity"] = float(similarities[i])
                        similar_entities_via_embedding_similarity.append(entity_packet)
        return similar_entities_via_embedding_similarity

    def _get_updates(self, state: SystemState) -> Dict:
        return {"similar_columns": state.similar_columns,
                "schema_with_examples": state.schema_with_examples}
