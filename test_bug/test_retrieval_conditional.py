"""
Test: get_retrieved_tables is only called when retrieval is actually needed.

Before the fix:
  get_retrieved_tables(dataset) was called unconditionally at line 328,
  BEFORE the preprocessing_option check. This caused AssertionError
  for Setting 1 (option=2) and Setting 2 (option=3) even though
  they don't need retrieval files.

After the fix:
  get_retrieved_tables(dataset) is only called when preprocessing_option == 1,
  which corresponds to Setting 0 (end-to-end mode requiring retrieval).
"""
import sys
import os
import json

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "eval", "reforce"))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

def test_setting_without_retrieval():
    """Setting 1 (option=2) should succeed without retrieval files."""
    from convert_beaver_to_reforce import convert_beaver_to_reforce
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_path = os.path.join(os.path.dirname(__file__), "test_opt2_output.json")
    
    convert_beaver_to_reforce(
        dataset="dw",
        beaver_questions_path=os.path.join(data_dir, "dw", "dev_sampled.json"),
        beaver_tables_path=os.path.join(data_dir, "dw", "dev_tables.json"),
        output_path=output_path,
        sampled_questions_path=output_path.replace(".json", "_sampled.json"),
        preprocessing_option=2,  # = Setting 1: gold tables, no retrieval needed
        db_id="dw",
    )
    
    assert os.path.exists(output_path), f"Output file {output_path} not created"
    with open(output_path) as f:
        data = json.load(f)
    assert len(data) > 0, "Output data is empty"
    print(f"[PASS] Setting 1 (option=2): {len(data)} entries processed successfully")
    os.remove(output_path)

def test_setting_without_retrieval_option3():
    """Setting 2 (option=3) should also succeed without retrieval files."""
    from convert_beaver_to_reforce import convert_beaver_to_reforce
    
    data_dir = os.path.join(os.path.dirname(__file__), "..", "data")
    output_path = os.path.join(os.path.dirname(__file__), "test_opt3_output.json")
    
    convert_beaver_to_reforce(
        dataset="dw",
        beaver_questions_path=os.path.join(data_dir, "dw", "dev_sampled.json"),
        beaver_tables_path=os.path.join(data_dir, "dw", "dev_tables.json"),
        output_path=output_path,
        sampled_questions_path=output_path.replace(".json", "_sampled.json"),
        preprocessing_option=3,  # = Setting 2: all hints, no retrieval needed
        db_id="dw",
    )
    
    assert os.path.exists(output_path)
    with open(output_path) as f:
        data = json.load(f)
    assert len(data) > 0
    print(f"[PASS] Setting 2 (option=3): {len(data)} entries processed successfully")
    os.remove(output_path)

if __name__ == "__main__":
    test_setting_without_retrieval()
    test_setting_without_retrieval_option3()
    print("\nAll tests passed!")
