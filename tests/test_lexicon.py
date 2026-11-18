from chatmod.lexicon import LexiconMatcher, build_lexicon, edit_distance


def test_edit_distance_basic_cases():
    assert edit_distance("", "abc") == 3
    assert edit_distance("abc", "abc") == 0
    assert edit_distance("kitten", "sitting") == 3
    assert edit_distance("salak", "slak") == 1  # deletion
    assert edit_distance("salak", "saalak") == 1  # insertion
    assert edit_distance("salak", "selak") == 1  # substitution


def test_matcher_flags_exact_and_misspelled_words():
    m = LexiconMatcher({"salak", "gerizekali"})
    assert m.predict("sen tam bir salak") == 1
    assert m.predict("GERİZEKALI misin") == 1  # normalized + asciified first
    assert m.predict("gerizekalli") == 1  # one edit away
    assert m.predict("s4l4k") == 1  # leetspeak decoded by normalize()
    assert m.predict("iyi oyundu tebrikler") == 0


def test_short_words_need_exact_match():
    # "kal" is one edit from "kel", but short words would cause too many false positives
    m = LexiconMatcher({"kel"})
    assert m.predict("kal burada") == 0
    assert m.predict("kel adam") == 1


def test_turkish_suffixes_are_tolerated_for_long_words():
    m = LexiconMatcher({"aptal", "am"}, prefix_match=True)
    assert m.predict("sen tam bir aptalsın") == 1  # aptal + -sın
    assert m.predict("aptallar") == 1
    assert m.predict("ama neden") == 0  # short lexicon words never match as prefix
    assert LexiconMatcher({"aptal"}, prefix_match=False).predict("aptalsın") == 0


def test_matched_words_are_reported():
    m = LexiconMatcher({"salak"})
    assert m.matches("sen slak misin") == [("slak", "salak", 1)]


def test_build_lexicon_keeps_words_typical_of_offensive_class():
    texts = ["salak adam"] * 6 + ["iyi adam"] * 6 + ["salak"] * 1
    labels = [1] * 6 + [0] * 6 + [1]
    lex = build_lexicon(texts, labels, min_count=3, min_ratio=0.8)
    assert lex == {"salak"}
