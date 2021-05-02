from string import punctuation

from unidecode import unidecode


def parse_line(line):
    clean_line = unidecode(line).lower().strip()
    clean_no_punctuation = "".join([i if i not in punctuation else " " for i in clean_line])
    words = clean_no_punctuation.split(" ")

    return filter(lambda _w: len(_w) > 1, words)


def extract_words(*file_paths):
    for file_path in file_paths:
        with open(file_path, encoding="utf-8") as f:
            for line in f:
                for _word in parse_line(line):
                    yield _word
