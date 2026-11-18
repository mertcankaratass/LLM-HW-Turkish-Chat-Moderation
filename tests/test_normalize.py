from chatmod.normalize import asciify, normalization_steps, normalize


def test_normalization_steps_name_the_rules_that_fired():
    assert normalization_steps("s4l4k mısın") == ["leetspeak"]
    assert normalization_steps("aptaaaal") == ["repeated letters"]
    assert normalization_steps("GERİZEKALI") == ["Turkish casing"]
    assert normalization_steps("S4L4AAAAK") == ["leetspeak", "repeated letters"]
    assert normalization_steps("AP7AL IIII") == ["Turkish casing", "leetspeak", "repeated letters"]
    assert normalization_steps("gg iyi oyundu") == []
    assert normalization_steps("en az 3000 kadro") == []


def test_turkish_aware_lowercase():
    assert normalize("İSTANBUL IŞIK") == "istanbul ışık"


def test_leetspeak_between_letters_is_decoded():
    assert normalize("s4l4k") == "salak"
    assert normalize("s@l@k misin") == "salak misin"
    assert normalize("gelm3ye") == "gelmeye"


def test_real_numbers_and_hashtags_are_kept():
    assert normalize("en az 3000 kadro") == "en az 3000 kadro"
    assert normalize("#yaşasın1mayıs tv8") == "#yaşasın1mayıs tv8"


def test_user_mentions_and_urls_become_placeholders():
    assert normalize("@USER @USER selam URL") == "@user @user selam url"


def test_repeated_letters_are_collapsed():
    assert normalize("aptaaaaal") == "aptal"
    assert normalize("çoooook güzel") == "çok güzel"
    # legitimate double letters are kept
    assert normalize("saat hakkında") == "saat hakkında"


def test_whitespace_is_squeezed():
    assert normalize("  merhaba    dünya  ") == "merhaba dünya"


def test_asciify_maps_turkish_letters():
    assert asciify("çığşöü") == "cigsou"
    assert asciify(normalize("GERİZEKALI")) == "gerizekali"
