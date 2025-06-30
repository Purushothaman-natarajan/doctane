
import string

__all__ = ["VOCABS", "get_vocab"]

VOCABS: dict[str, str] = {
    # Arabic & Persian
    "arabic_diacritics": "ًٌٍَُِّْ",
    "arabic_digits": "٠١٢٣٤٥٦٧٨٩",
    "arabic_letters": "ءآأؤإئابةتثجحخدذرزسشصضطظعغـفقكلمنهوىي",
    "arabic_punctuation": "؟؛«»—",
    "persian_letters": "پچڢڤگ",

    # Bangla
    "bangla_digits": "০১২৩৪৫৬৭৮৯",
    "bangla_letters": "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহ়ঽািীুূৃেৈোৌ্ৎংঃঁ",

    # Cyrillic
    "generic_cyrillic_letters": "абвгдежзийклмнопрстуфхцчшщьюяАБВГДЕЖЗИЙКЛМНОПРСТУФХЦЧШЩЬЮЯ",
    "russian_cyrillic_letters": "ёыэЁЫЭ",
    "russian_signs": "ъЪ",

    # Greek
    "ancient_greek": "αβγδεζηθικλμνξοπρστυφχψωΑΒΓΔΕΖΗΘΙΚΛΜΝΞΟΠΡΣΤΥΦΧΨΩ",

    # Gujarati
    "gujarati_digits": "૦૧૨૩૪૫૬૭૮૯",
    "gujarati_vowels": "અઆઇઈઉઊઋએઐઓઔ",
    "gujarati_consonants": "કખગઘઙચછજઝઞટઠડઢણતથદધનપફબભમયરલવશસહળક્ષ",
    "gujarati_punctuation": "૰ઽ◌ંઃ॥ૐ઼ઁ૱",

    # Hindi / Marathi (Devanagari)
    "hindi_digits": "०१२३४५६७८९",
    "hindi_letters": "अआइईउऊऋॠऌॡएऐओऔंःकखगघङचछजझञटठडढणतथदधनपफबभमयरलवशषसह",
    "hindi_punctuation": "।,?!:्ॐ॰॥",
    "marathi_specials": "ऱळक़ख़ग़ज़ड़ढ़फ़य़ॲऑ",

    # Tamil
    "tamil_digits": "௦௧௨௩௪௫௬௭௮௯",
    "tamil_letters": "அஆஇஈஉஊஎஏஐஒஓஔஃகஙசஞடணதநபமயரலவழளறனஷஸஹஜ",
    "tamil_punctuation": "।ௐஂஃ",

    # Telugu
    "telugu_digits": "౦౧౨౩౪౫౬౭౮౯",
    "telugu_letters": "అఆఇఈఉఊఋౠఎఏఐఒఓఔఽకఖగఘఙచఛజఝఞటఠడఢణతథదధనపఫబభమయరఱలళవశషసహ",
    "telugu_punctuation": "।॥ఽంః",

    # Kannada
    "kannada_digits": "೦೧೨೩೪೫೬೭೮೯",
    "kannada_letters": "ಅಆಇಈಉಊಋೠಎಏಐಒಓಔಅಂಅಃಕಖಗಘಙಚಛಜಝಞಟಠಡಢಣತಥದಧನಪಫಬಭಮಯರಱಲಳವಶಷಸಹ",
    "kannada_punctuation": "।॥ೱಂಃ",

    # Malayalam
    "malayalam_digits": "൦൧൨൩൪൫൬൭൮൯",
    "malayalam_letters": "അആഇഈഉഊഋൠഎഏഐഒഓഔകഖഗഘങചഛജഝഞടഠഡഢണതഥദധനപഫബഭമയരറലളവശഷസഹ",
    "malayalam_punctuation": "।॥ംഃ",

    # Punjabi (Gurmukhi)
    "punjabi_digits": "੦੧੨੩੪੫੬੭੮੯",
    "punjabi_letters": "ਅਆਇਈਉਊਏਐਓਔਅੰਅਃਕਖਗਘਙਚਛਜਝਞਟਠਡਢਣਤਥਦਧਨਪਫਬਭਮਯਰਲਵਸ਼ਸਹਲ਼ੜੴ",
    "punjabi_punctuation": "।ੵ਼ਖ਼ਗ਼ਜ਼ਫ਼ਲ਼ੰੱਁੰੰਂ:?!",

    # Odia (Oriya)
    "odia_digits": "୦୧୨୩୪୫୬୭୮୯",
    "odia_letters": "ଅଆଇଈଉଊଋୠଌୡଏଐଓଔଂଃକଖଗଘଙଚଛଜଝଞଟଠଡଢଣତଥଦଧନପଫବଭମଯରଲଳଵଶଷସହଡ଼ଢ଼ୟ",
    "odia_punctuation": "।ଽ୰ୱଁ:?!",

    # Hebrew
    "hebrew_cantillations": "֑֖֛֢֣֤֥֦֧֪֚֭֮֒֓֔֕֗֘֙֜֝֞֟֠֡֨֩֫֬֯",
    "hebrew_letters": "אבגדהוזחטיךכלםמןנסעףפץצקרשת",
    "hebrew_specials": "ׯװױײיִﬞײַﬠﬡﬢﬣﬤﬥﬦﬧﬨ﬩שׁשׂשּׁשּׂאַאָאּבּגּדּהּוּזּטּיּךּכּלּמּנּסּףּפּצּקּרּשּתּוֹבֿכֿפֿﭏ",
    "hebrew_punctuation": "ֽ־ֿ׀ׁׂ׃ׅׄ׆׳״",
    "hebrew_vowels": "ְֱֲֳִֵֶַָׇֹֺֻ",

    # Latin
    "digits": string.digits,
    "ascii_letters": string.ascii_letters,
    "punctuation": string.punctuation,
    "currency": "£€¥¢฿",
}

# Latin-based languages
VOCABS["latin"] = VOCABS["digits"] + VOCABS["ascii_letters"] + VOCABS["punctuation"]
VOCABS["english"] = VOCABS["latin"] + "°" + VOCABS["currency"]

# Combine Hindi/Marathi (both Devanagari-based)
VOCABS["hindi"] = VOCABS["hindi_letters"] + VOCABS["hindi_digits"] + VOCABS["hindi_punctuation"]
VOCABS["marathi"] = VOCABS["hindi"] + VOCABS["marathi_specials"]

# Other Indian languages
VOCABS["bangla"] = VOCABS["bangla_letters"] + VOCABS["bangla_digits"]
VOCABS["gujarati"] = VOCABS["gujarati_vowels"] + VOCABS["gujarati_consonants"] + VOCABS["gujarati_digits"] + VOCABS["gujarati_punctuation"] + VOCABS["punctuation"]
VOCABS["tamil"] = VOCABS["tamil_letters"] + VOCABS["tamil_digits"] + VOCABS["tamil_punctuation"] + VOCABS["punctuation"]
VOCABS["telugu"] = VOCABS["telugu_letters"] + VOCABS["telugu_digits"] + VOCABS["telugu_punctuation"] + VOCABS["punctuation"]
VOCABS["kannada"] = VOCABS["kannada_letters"] + VOCABS["kannada_digits"] + VOCABS["kannada_punctuation"] + VOCABS["punctuation"]
VOCABS["malayalam"] = VOCABS["malayalam_letters"] + VOCABS["malayalam_digits"] + VOCABS["malayalam_punctuation"] + VOCABS["punctuation"]
VOCABS["punjabi"] = VOCABS["punjabi_letters"] + VOCABS["punjabi_digits"] + VOCABS["punjabi_punctuation"] + VOCABS["punctuation"]
VOCABS["odia"] = VOCABS["odia_letters"] + VOCABS["odia_digits"] + VOCABS["odia_punctuation"] + VOCABS["punctuation"]

# Western - Languages
VOCABS["czech"] = VOCABS["english"] + "áčďéěíňóřšťúůýžÁČĎÉĚÍŇÓŘŠŤÚŮÝŽ"
VOCABS["danish"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["dutch"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ"
VOCABS["french"] = VOCABS["english"] + "àâéèêëîïôùûüçÀÂÉÈÊËÎÏÔÙÛÜÇ"
VOCABS["legacy_french"] = VOCABS["latin"] + "°" + "àâéèêëîïôùûçÀÂÉÈËÎÏÔÙÛÇ" + VOCABS["currency"]
VOCABS["finnish"] = VOCABS["english"] + "äöÄÖ"
VOCABS["german"] = VOCABS["english"] + "äöüßÄÖÜẞ"
VOCABS["croatian"] = VOCABS["english"] + "ČčĆćĐđŠšŽž"
VOCABS["hebrew"] = (
    VOCABS["english"]
    + VOCABS["hebrew_letters"]
    + VOCABS["hebrew_vowels"]
    + VOCABS["hebrew_punctuation"]
    + VOCABS["hebrew_cantillations"]
    + VOCABS["hebrew_specials"]
    + "₪"
)
VOCABS["italian"] = VOCABS["english"] + "àèéìíîòóùúÀÈÉÌÍÎÒÓÙÚ"
VOCABS["norwegian"] = VOCABS["english"] + "æøåÆØÅ"
VOCABS["polish"] = VOCABS["english"] + "ąćęłńóśźżĄĆĘŁŃÓŚŹŻ"
VOCABS["portuguese"] = VOCABS["english"] + "áàâãéêíïóôõúüçÁÀÂÃÉÊÍÏÓÔÕÚÜÇ"
VOCABS["spanish"] = VOCABS["english"] + "áéíóúüñÁÉÍÓÚÜÑ" + "¡¿"
VOCABS["swedish"] = VOCABS["english"] + "åäöÅÄÖ"
VOCABS["vietnamese"] = (
    VOCABS["english"]
    + "áàảạãăắằẳẵặâấầẩẫậđéèẻẽẹêếềểễệóòỏõọôốồổộỗơớờởợỡúùủũụưứừửữựíìỉĩịýỳỷỹỵ"
    + "ÁÀẢẠÃĂẮẰẲẴẶÂẤẦẨẪẬĐÉÈẺẼẸÊẾỀỂỄỆÓÒỎÕỌÔỐỒỔỘỖƠỚỜỞỢỠÚÙỦŨỤƯỨỪỬỮỰÍÌỈĨỊÝỲỶỸỴ"
)

# Non-Latin
VOCABS["arabic"] = (
    VOCABS["digits"]
    + VOCABS["arabic_digits"]
    + VOCABS["arabic_letters"]
    + VOCABS["persian_letters"]
    + VOCABS["arabic_diacritics"]
    + VOCABS["arabic_punctuation"]
    + VOCABS["punctuation"]
)
VOCABS["russian"] = (
    VOCABS["generic_cyrillic_letters"]
    + VOCABS["russian_cyrillic_letters"]
    + VOCABS["russian_signs"]
    + VOCABS["digits"]
    + VOCABS["punctuation"]
    + "₽"
)
VOCABS["ukrainian"] = (
    VOCABS["generic_cyrillic_letters"] + VOCABS["digits"] + VOCABS["punctuation"] + VOCABS["currency"] + "ґіїєҐІЇЄ₴"
)

# Multilingual combo
VOCABS["multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["french"]
        + VOCABS["portuguese"]
        + VOCABS["spanish"]
        + VOCABS["german"]
        + VOCABS["czech"]
        + VOCABS["croatian"]
        + VOCABS["polish"]
        + VOCABS["dutch"]
        + VOCABS["italian"]
        + VOCABS["norwegian"]
        + VOCABS["danish"]
        + VOCABS["finnish"]
        + VOCABS["swedish"]
        + VOCABS["hindi"]
        + "§"
    )
)

# Indian-lingual (Latin + Indian major scripts)
VOCABS["indian_multilingual"] = "".join(
    dict.fromkeys(
        VOCABS["hindi"]
        + VOCABS["marathi"]
        + VOCABS["gujarati"]
        + VOCABS["punjabi"]
        + VOCABS["odia"]
        + VOCABS["tamil"]
        + VOCABS["telugu"]
        + VOCABS["kannada"]
        + VOCABS["malayalam"]
        + VOCABS["english"]
        + VOCABS["hebrew"]
    )
)

def get_vocab(language: str) -> str:
    """Returns a set of valid characters for the specified language."""
    language = language.lower()
    if language in VOCABS:
        return VOCABS[language]
    else:
        raise ValueError(f"Language '{language}' not supported. Available: {list(VOCABS.keys())}")
