from typing import List
import re
import json
from functools import wraps

import requests
from cachier import cachier

from hebrew import Niqqud
import hebrew


class DottingError(RuntimeError):
    pass


def split_string_by_length(text: str, maxlen) -> List[str]:
    return [''.join(s).strip() for s in hebrew.split_by_length(text, maxlen)]


def piecewise(maxlen):
    def inner(fetch):
        @wraps(fetch)
        def fetcher(text):
            return ' '.join(fetch(chunk) for chunk in split_string_by_length(text, maxlen))
        return fetcher
    return inner


@cachier()
@piecewise(75)  # estimated maximum for reasonable time
def fetch_snopi(text: str) -> str:
    # Add bogus continuation in case there's only a single word
    # so Snopi will not decide to answer with single-word-analysis
    text = text + ' 1'

    url = 'http://www.nakdan.com/GetResult.aspx'

    payload = {
        "txt": text,
        "ktivmale": 'true',
    }
    headers = {
        'Referer': 'http://www.nakdan.com/nakdan.aspx',
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    res = list(r.text.split('Result')[1][1:-2])
    items = list(hebrew.iterate_dotted_text(res))

    for i in range(len(text)):
        if text[i] != ' ' and items[i].letter == ' ':
            del items[i]
        elif text[i] != items[i].letter:
            items.insert(i, hebrew.HebrewItem(text[i], '', '', '', ''))
    res = hebrew.items_to_text(items)
    assert hebrew.remove_niqqud(res) == text, f'{repr(hebrew.remove_niqqud(res))}\n!=\n{repr(text)}'

    return res[:-2]


@cachier()
@piecewise(100)
def fetch_morfix(text: str) -> str:
    url = 'https://nakdan.morfix.co.il/nikud/NikudText'

    payload = {
        "text": text,
        "isLogged": 'false',
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return json.loads(r.json()['nikud'])['OutputText']


@cachier()
@piecewise(10000)
def fetch_dicta(text: str) -> str:
    text = '\n'.join(line for line in text.split('\n') if not line.startswith('https') and not line.startswith('#')).strip()
    def extract_word(k):
        if k['options']:
            res = k['options'][0][0]
            res = res.replace('|', '')
            res = res.replace(Niqqud.KUBUTZ + 'ו' + Niqqud.METEG, 'ו' + Niqqud.SHURUK)
            res = res.replace(Niqqud.HOLAM + 'ו' + Niqqud.METEG, 'ו' + Niqqud.HOLAM)
            res = res.replace(Niqqud.METEG, '')

            res = re.sub(Niqqud.KAMATZ + 'ו' + '(?=[א-ת])', 'ו' + Niqqud.HOLAM, res)
            res = res.replace(Niqqud.REDUCED_KAMATZ + 'ו', 'ו' + Niqqud.HOLAM)

            res = res.replace(hebrew.DAGESH_LETTER * 2, hebrew.DAGESH_LETTER)
            res = res.replace('\u05be', '-')
            res = res.replace('יְהוָֹה', 'יהוה')
            return res
        return k['word']

    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'

    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    result = ''.join(extract_word(k) for k in r.json())
    if len(hebrew.find_longest_undotted(result)) > 40:
        raise DottingError('Failed to dot')
    return result


@piecewise(10000)
def fetch_nakdimon(text: str) -> str:
    url = 'http://127.0.0.1:5000'

    payload = {
        "text": text,
        "model_name": 'final_model/final.h5'
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return r.text


@piecewise(10000)
def fetch_nakdimon_no_dicta(text: str) -> str:
    url = 'http://127.0.0.1:5000'

    payload = {
        "text": text,
        "model_name": 'models/without_dicta.h5'
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return r.text


@piecewise(10000)
def fetch_nakdimon_fullnew(text: str) -> str:
    url = 'http://127.0.0.1:5000'

    payload = {
        "text": text,
        "model_name": 'models/FullNewCleaned.h5'
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return r.text


@piecewise(10000)
def fetch_nakdimon_FinalWithShortStory(text: str) -> str:
    url = 'http://127.0.0.1:5000'

    payload = {
        "text": text,
        "model_name": 'models/FinalWithShortStory.h5'
    }
    headers = {
    }

    r = requests.post(url, data=payload, headers=headers)
    r.raise_for_status()
    return r.text


SYSTEMS = {
    'Snopi': fetch_snopi,  # Too slow
    'Morfix': fetch_morfix,  # terms-of-use issue
    'Dicta': fetch_dicta,
    'Nakdimon': fetch_nakdimon,
    'NakdimonNoDicta': fetch_nakdimon_no_dicta,
    'NakdimonFullNew': fetch_nakdimon_fullnew,
    'NakdimonFinalWithShortStory': fetch_nakdimon_FinalWithShortStory,
}

# fetch_nakdimon.clear_cache()
# fetch_dicta.clear_cache()


def fetch_dicta_count_ambiguity(text: str):
    url = 'https://nakdan-2-0.loadbalancer.dicta.org.il/api'

    payload = {
        "task": "nakdan",
        "genre": "modern",
        "data": text,
        # "addmorph": True,
        "keepqq": False,
        "nodageshdefmem": False,
        "patachma": False,
        "keepmetagim": True,
    }
    headers = {
        'content-type': 'text/plain;charset=UTF-8'
    }

    r = requests.post(url, json=payload, headers=headers)
    r.raise_for_status()
    return [len(set(token['options'])) for token in r.json() if not token['sep']]


if __name__ == '__main__':
    text = 'בית עם גינה'
    counts = fetch_dicta_count_ambiguity(text)
    print(counts)

fetch_dicta.clear_cache()
