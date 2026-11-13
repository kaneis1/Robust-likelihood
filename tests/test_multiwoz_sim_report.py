from robust_likelihood.multiwoz.sim_report import num, ordered, pct, table, with_ci


def test_table_escapes_pipes_in_cells():
    text = table(["Model", "value"], [["osim_mid|S", "1"]])
    assert "osim_mid\\|S" in text and text.count("\n") == 2


def test_number_formats():
    assert num(None) == "–" and num(1.234) == "1.23" and num(True) == "yes"
    assert pct(0.5) == "50.0%" and pct(None) == "–"
    assert with_ci({"median": 0.5, "ci95": [0.25, 0.75]}, "median", percent=True) == "50.0% [25.0%, 75.0%]"
    assert with_ci({"mean": None}, "mean") == "–"


def test_models_follow_the_report_order():
    runs = {"userlm|S": {}, "qwen25_inst|P0": {}, "osim_mid|S": {}, "extra|X": {}}
    assert ordered(runs) == ["qwen25_inst|P0", "osim_mid|S", "userlm|S", "extra|X"]
