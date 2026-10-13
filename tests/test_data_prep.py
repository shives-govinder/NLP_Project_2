from nlp2 import config
from nlp2.data_prep import assign_edition_splits, clean_text, finalise_segment, keep_segment, parse_filename, segment


def test_parse_filename():
    m = parse_filename("IMVO_1893-03-09_p004.txt")
    assert m["paper"] == "IMVO" and m["year"] == 1893 and m["page"] == 4
    assert m["edition"] == "IMVO_1893-03-09"


def test_clean_removes_headers_and_joins_hyphens():
    raw = "IMVO ZABANTSUNDU (NATIVE OPINION) 3\n\numsebenzi ba-\nbehle kanjalo\n10\n"
    out = clean_text(raw)
    assert "IMVO" not in out
    assert "babehle" in out
    assert "10" not in out


def test_illegible_words_are_flagged_then_removed():
    out = clean_text("ABO bati????????ngeniswe Amatshayina njalonjalo ukuba")
    seg, had = finalise_segment(segment(out)[0])
    assert had and "?" not in seg and config.ILLEGIBLE_TOKEN not in seg


def test_keep_segment_filters_short_and_noisy():
    assert not keep_segment("ewe kunjalo")
    assert not keep_segment("12 34 56 78 90 ;; -- ab")
    assert keep_segment("Wenjenje ukuyi pendula londawo u-Mr. Distin ubonisa ukuba")


def test_edition_splits_are_disjoint_and_cover_all():
    eds = [f"IMVO_1890-01-{d:02d}" for d in range(1, 29)] + [f"ISIG_1880-{m:02d}-01" for m in range(1, 13)]
    df = assign_edition_splits(eds)
    assert set(df["edition"]) == set(eds)
    assert df["edition"].is_unique
    assert set(df["split"]) == {"train", "validation", "test"}


def test_capitalised_continuation_keeps_hyphen():
    assert "u-Mxosa" in clean_text("ukuba u-\nMxosa uyenqena")


def test_curly_quotes_are_straightened():
    assert "ib'ingu" in clean_text("Inkabi yalo mazwi ib\u2019ingu Dunga.")


def test_broken_fragments_are_dropped():
    assert not keep_segment("Train drivers very close to going on strike, union warns. rain drivers are close to it")
    assert keep_segment("He met u-Mr. Distin at the station and they talked for a long time.")


def test_no_split_after_titles_or_initials():
    segs = segment(clean_text("Kusuke ingwevu ka Gqadushe u Rev. J. Mzimba. Wati emitetweni yake masingayityi."))
    assert segs[0] == "Kusuke ingwevu ka Gqadushe u Rev. J. Mzimba."
    assert len(segs) == 2
