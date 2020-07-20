"""For extracting direct quotes from Chinese text."""
from . import tokenization


# pairs that mark start and end of a quote
quote_start_markers = '“「〈《【'
quote_end_markers = {
    '“': '”',
    '「': '」',
    '〈': '〉',
    '《': '》',
    '【': '】',
}


def extract(text):
    """Extracts quotes from the text.

    Heuristics:
      - Anything (and only things) inside quote markers is a potential quote.
      - But, sometimes they quote concepts, perhaps sarcastically, but don't
        want to lose those. From observing usage, e.g.:
          不要以为擅自提前返校添的是“小乱”，疫情面前，要勿以“添小乱”而为之。
          (articleId: 90128009)
        heuristically set a threshold of > 2 tokenized words to be a quote.
    """
    quotes = []
    current = None
    in_quote = False
    start_char = None
    for c in text:
        if not in_quote:
            if c in quote_start_markers:
                in_quote = True
                start_char = c
                current = ''
        else:  # in quote
            if c == quote_end_markers[start_char]:
                tokens = tokenization.tokenize_jieba(current)
                if len(tokens) > 2:
                    quotes.append(current)
                in_quote = False
                start_char = None
                current = None
            else:
                current += c
    return quotes
