import string

import jieba

from .stopwords import zh_stopwords


zh_punct = '！”“＃＄％＆‘’（）＊＋，。 …※；：﹔‥．？‧「」〔〕｛｝〈〉＼｜～、' \
           '\r\n\t《》·／▌【】◆'
start_quotes = '”'
punct = string.punctuation + zh_punct
digits = string.digits + '一二三四五六七八九十' + '.%,'


def coerce_num(tok):
    if all(c in digits for c in tok):
        return 'NUM'
    else:
        return tok


def zh_sent_tokenize(text):
    line_breaks = [s.strip() for s in text.split('\n')]
    candidates = [s for s in line_breaks if '。' in s]
    sents = []
    for x in candidates:
        sents += [s.strip() for s in x.split('。')
                  if len(s) > 0]
    return sents


def tokenize_jieba(text):
    text = text.strip()
    jieba_toks = [t.strip() for t in jieba.cut(text)]
    toks = []
    for tok in jieba_toks:
        # skip if punctuation
        if all(c in punct for c in tok):
            continue
        # skip stopwords
        if tok in zh_stopwords:
            continue
        # coerce numbers to NUM
        if all(c in digits for c in tok):
            # toks.append('NUM')  # don't add it for topic modeling
            continue
        # drop things like "span" and "class" and non-zh
        if all(c in string.ascii_lowercase for c in tok.lower()):
            continue
        # if it's all alpha and digits, part of a messy link
        if all(c in string.ascii_lowercase or c in digits for c in tok.lower()):
            continue
        # latest heuristic: drop any single-character words
        if len(tok) < 2:
            continue
        toks.append(tok)
    return toks
