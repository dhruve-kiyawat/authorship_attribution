from indic_transliteration import sanscript

def transliterate_hindi_to_english(hindi_text):
    return sanscript.transliterate(hindi_text, sanscript.DEVANAGARI, sanscript.ITRANS)

hindi_text = "मुझे तुम्हारे काम की सराहना करनी चाहिए।"
english_text = transliterate_hindi_to_english(hindi_text)
print("Hindi Text:", hindi_text)
print("English Text:", english_text)