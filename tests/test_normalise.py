from nlp2.normalise import Normaliser, NormParams, evaluate, h_candidates, naive_all

LEX = {"ukuthi": 50, "wathi": 30, "apha": 40, "kuba": 100, "nkosi": 20}


def test_candidates_insert_h_after_aspirable_consonants():
    assert "ukuthi" in h_candidates("ukuti", max_insert=2)
    assert all(c.count("h") == 1 for c in h_candidates("ukuti", max_insert=1))


def test_normaliser_restores_attested_forms_only():
    n = Normaliser(LEX, NormParams(min_freq=2, ratio=2.0, max_insert=2))
    assert n.word("ukuti") == "ukuthi"
    assert n.word("apa") == "apha"
    assert n.word("kuba") == "kuba"          # already modern, no h-candidate attested
    assert n.word("umntu") == "umntu"        # nothing attested -> unchanged


def test_case_is_preserved_in_running_text():
    n = Normaliser(LEX, NormParams(min_freq=2, ratio=2.0, max_insert=2))
    assert n.text("Wati, ukuti apa.") == "Wathi, ukuthi apha."


def test_naive_baseline_over_inserts():
    assert naive_all("kuba") == "khuba"


def test_evaluate_metrics():
    m = evaluate(pred=["ukuthi", "kuba", "apa"], gold_hist=["ukuti", "kuba", "apa"], gold_mod=["ukuthi", "kuba", "apha"])
    assert m["accuracy"] == 2 / 3
    assert m["rewrite_precision"] == 1.0
    assert m["rewrite_recall"] == 0.5
