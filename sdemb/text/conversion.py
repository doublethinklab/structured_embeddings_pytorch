import opencc


cc_s2tw = opencc.OpenCC('s2tw.json')
cc_t2s = opencc.OpenCC('t2s.json')


def s2t(text):
    return cc_s2tw.convert(text)


def t2s(text):
    return cc_t2s.convert(text)
