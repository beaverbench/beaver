"""
Test: get_retrieved_tables is only called when retrieval is actually needed.

Affected files:
  - eval/reforce/convert_beaver_to_reforce.py
  - eval/dinsql/preprocessed_data/beaver_preprocess_v2.py
  - eval/dailsql/preprocessed_data/beaver_preprocess.py

Before the fix:
  get_retrieved_tables(dataset) was called unconditionally before
  the option/ setting check, causing AssertionError for Setting 1 (option=2)
  and Setting 2 (option=3) even though they use gold tables from data.

After the fix:
  get_retrieved_tables(dataset) is only called when option == 1
  (Setting 0 / end-to-end mode requiring retrieval).
"""
import sys
import os
import json

REPO_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, REPO_DIR)

DATA_DIR = os.path.join(REPO_DIR, "data")
OUTPUT_DIR = os.path.dirname(os.path.abspath(__file__))


def test_reforce_setting_1():
    """ReFoRCE: Setting 1 (option=2) without retrieval."""
    from eval.reforce.convert_beaver_to_reforce import convert_beaver_to_reforce
    
    out = os.path.join(OUTPUT_DIR, "reforce_opt2.json")
    sampled = out.replace(".json", "_sampled.json")
    convert_beaver_to_reforce(
        dataset="dw",
        beaver_questions_path=os.path.join(DATA_DIR, "dw", "dev_sampled.json"),
        beaver_tables_path=os.path.join(DATA_DIR, "dw", "dev_tables.json"),
        output_path=out,
        sampled_questions_path=sampled,
        preprocessing_option=2,
        db_id="dw",
    )
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert len(data) > 0
    os.remove(out)
    if os.path.exists(sampled):
        os.remove(sampled)
    print(f"  [PASS] ReFoRCE option=2: {len(data)} entries")


def test_dinsql_setting_1():
    """DIN-SQL: Setting 1 (option=2) without retrieval."""
    from eval.dinsql.preprocessed_data.beaver_preprocess_v2 import convert_beaver_questions_to_dinsql_format
    
    out = os.path.join(OUTPUT_DIR, "dinsql_opt2.json")
    convert_beaver_questions_to_dinsql_format(
        dataset="dw",
        beaver_questions_path=os.path.join(DATA_DIR, "dw", "dev_sampled.json"),
        output_path=out,
        option=2,
    )
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert len(data) > 0
    os.remove(out)
    print(f"  [PASS] DIN-SQL option=2: {len(data)} entries")


def test_dailsql_setting_1():
    """DAIL-SQL: Setting 1 (option=2) without retrieval."""
    from eval.dailsql.preprocessed_data.beaver_preprocess import convert_beaver_questions_to_dailsql_format
    
    out = os.path.join(OUTPUT_DIR, "dailsql_opt2.json")
    convert_beaver_questions_to_dailsql_format(
        dataset="dw",
        beaver_questions_path=os.path.join(DATA_DIR, "dw", "dev_sampled.json"),
        output_path=out,
        option=2,
    )
    assert os.path.exists(out)
    with open(out) as f:
        data = json.load(f)
    assert len(data) > 0
    os.remove(out)
    print(f"  [PASS] DAIL-SQL option=2: {len(data)} entries")


def test_setting_0_still_requires_retrieval():
    """Setting 0 (option=1) should still fail if retrieval files are missing."""
    from eval.reforce.convert_beaver_to_reforce import convert_beaver_to_reforce
    
    out = os.path.join(OUTPUT_DIR, "setting0_should_fail.json")
    sampled = out.replace(".json", "_sampled.json")
    try:
        convert_beaver_to_reforce(
            dataset="dw",
            beaver_questions_path=os.path.join(DATA_DIR, "dw", "dev_sampled.json"),
            beaver_tables_path=os.path.join(DATA_DIR, "dw", "dev_tables.json"),
            output_path=out,
            sampled_questions_path=sampled,
            preprocessing_option=1,
            db_id="dw",
        )
        print("  [FAIL] Setting 0 should have raised an error (no retrieval files)")
    except AssertionError:
        print("  [PASS] Setting 0 raises AssertionError as expected (no retrieval files)")
    
    if os.path.exists(out):
        os.remove(out)
    if os.path.exists(sampled):
        os.remove(sampled)


if __name__ == "__main__":
    print("=== Test: Non-retrieval settings without retrieval files ===\n")
    print("ReFoRCE:");    test_reforce_setting_1()
    print("DIN-SQL:");    test_dinsql_setting_1()
    print("DAIL-SQL:");   test_dailsql_setting_1()
    print("\n=== Test: Setting 0 still requires retrieval ===")
    test_setting_0_still_requires_retrieval()
    print("\n=== All tests completed! ===")
