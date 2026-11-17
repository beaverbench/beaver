from typing import Any, Dict


def get_config(model: str, embedding_model: str) -> Dict[str, Any]:
    """
    CHESS run/configs/CHESS_IR_SS_CG.yaml (information retriever -> schema selector -> candidate generator)
    with one model for every step; CHESS used gpt-4o-mini, and gpt-4o for selecting tables and columns.
    retrieve_context is dropped (BEAVER has no column descriptions), and filter_column judges the columns
    of one table per call instead of one column per call.
    """
    def engine(temperature: float) -> Dict[str, Any]:
        return {"engine_name": model, "temperature": temperature}

    return {
        "setting_name": "CHESS_IR_SS_CG",
        "team_agents": {
            "information_retriever": {
                "tools": {
                    "extract_keywords": {
                        "template_name": "extract_keywords",
                        "engine_config": engine(0.2),
                        "parser_name": "python_list_output_parser",
                    },
                    "retrieve_entity": {"embedding_model": embedding_model},
                },
            },
            "schema_selector": {
                "tools": {
                    "filter_column": {
                        "template_name": "filter_column",
                        "engine_config": engine(0.0),
                        "parser_name": "filter_column",
                        "max_columns_per_call": 50,
                    },
                    "select_tables": {
                        "mode": "ask_model",
                        "template_name": "select_tables",
                        "engine_config": engine(0.0),
                        "parser_name": "select_tables",
                    },
                    "select_columns": {
                        "mode": "ask_model",
                        "template_name": "select_columns",
                        "engine_config": engine(0.0),
                        "parser_name": "select_columns",
                    },
                },
            },
            "candidate_generator": {
                "tools": {
                    "generate_candidate": {
                        "generator_configs": [{
                            "template_name": "generate_candidate_one",
                            "engine_config": engine(0.01),
                            "parser_name": "generate_candidate_gemini_markdown_cot",
                            "sampling_count": 1,
                        }],
                    },
                    "revise": {
                        "template_name": "revise_one",
                        "engine_config": engine(0.0),
                        "parser_name": "revise_new",
                    },
                },
            },
        },
    }
