from mt_training.text import normalize_for_nllb


def test_guillemets_become_straight_quotes_hugging_the_text():
    assert normalize_for_nllb("Il dit : « Je suis arrivé. »") == 'Il dit : "Je suis arrivé."'
    assert normalize_for_nllb("A yeelame : «M waame !»") == 'A yeelame : "M waame !"'


def test_apostrophes_quotes_dashes():
    assert normalize_for_nllb("C’est l‘état ʼa") == "C'est l'état 'a"
    assert normalize_for_nllb("“Hello” „x”") == '"Hello" "x"'
    assert normalize_for_nllb("a — b – c") == "a - b - c"


def test_greek_perispomeni_becomes_a_composed_tilde_vowel():
    assert normalize_for_nllb("ba͂") == "bã"


def test_plain_text_is_unchanged():
    text = 'Tõnd na n kẽnga yiri, t\'a sẽn yeel "ayo".'
    assert normalize_for_nllb(text) == text


def test_retroflex_l_is_the_capital_iota_of_all_caps_headings():
    assert normalize_for_nllb("BÕN-VɭɭLɭ KOGLG Tɭ") == "BÕN-VƖƖLƖ KOGLG TƖ"
