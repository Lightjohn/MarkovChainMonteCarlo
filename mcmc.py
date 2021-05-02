import os
import string

import matplotlib.pyplot as plt
import numpy as np

from preprocessing import extract_words


def _update_table(current_char, next_char, result, indirection):
    current_char_index = indirection[current_char]
    next_char_index = indirection[next_char]
    result[current_char_index, next_char_index] += 1


def compute_char_relation(_word: str, result_table, indirection_table: dict):
    _update_table(" ", _word[0], result_table, indirection_table)
    for i in range(0, len(_word) - 1):
        _update_table(_word[i], _word[i + 1], result_table, indirection_table)
    _update_table(_word[-1], " ", result_table, indirection_table)


def create_indirection_table():
    data = [i for i in " " + string.ascii_lowercase + string.digits]
    return {j: i for i, j in enumerate(data)}


def create_simpler_indirection_table():
    data = [i for i in " " + string.ascii_lowercase]
    return {j: i for i, j in enumerate(data)}


def save_table(_table, indirection, show_digits=True):
    if not show_digits:
        _table = _table[:27, :27]
        indirection = {j[0]: j[1] for i, j in enumerate(indirection.items()) if i < 27}
    # Better coloring of output
    _table = _table ** 0.33
    plt.figure(figsize=(8, 8))
    plt.imshow(_table, interpolation='nearest', cmap='inferno')
    plt.axis('off')

    for ip, i in enumerate(indirection):
        plt.text(-1, ip, i, horizontalalignment='center', verticalalignment='center')
        plt.text(ip, -1, i, horizontalalignment='center', verticalalignment='center')
    plt.savefig("out.jpg")


def compute_stats(_word, _stats):
    for c in _word:
        _stats[c] += 1


def print_stats(_stats):
    for w in _sort_dict(_stats):
        print(f"'{w}' -> {_stats[w]}")


def get_plausibility(_input_string, probability_table, _bigrams):
    # I expect cleaned string as input i.e every char in _input_string should be in _bigrams
    proba = 0
    input_size = len(_input_string)
    for i in range(input_size - 1):
        x = _bigrams[_input_string[i]]
        y = _bigrams[_input_string[i + 1]]
        proba += probability_table[x][y]
    return proba / input_size


def _sort_dict(input_dict):
    return sorted(input_dict, key=input_dict.get, reverse=True)


def make_first_proposition(crypt_phrase, global_stats, _char_to_num):
    _stats = {i: 0 for i in _char_to_num}
    for i in crypt_phrase:
        _stats[i] += 1
    first_estimation = {}
    for i, j in zip(_sort_dict(_stats), _sort_dict(global_stats)):
        first_estimation[i] = j
    return first_estimation


if __name__ == '__main__':
    input_file = "MiserablesV1.txt"
    # input_file = "Swann.txt"
    chars = {}
    char_to_num = create_indirection_table()
    stats = {i: 0 for i in char_to_num}
    size = len(char_to_num)
    bigrams = np.zeros((size, size))
    for word in extract_words(os.path.join("data", input_file)):
        compute_char_relation(word, bigrams, char_to_num)
        compute_stats(word, stats)

    # Normalising table
    # will be errors if no char are found in all text like "5" in original "Swann.txt"
    bigrams = bigrams / bigrams.sum(axis=1)[:, None]

    save_table(bigrams, char_to_num, False)

    coded_phrase = "w0uow0kfikfjkzgorftkfzu0inkfktfg0i0tkfsgtokxkfg0flutjfskskfjkfikfw0kfpkf1goyf1u0yfvgxrkx"
    decrypt_guess = make_first_proposition(coded_phrase, stats, char_to_num)
    best_decoded = "".join([decrypt_guess[i] for i in coded_phrase])
    best_proba = get_plausibility(best_decoded, bigrams, char_to_num)

    print(f"INPUT: {coded_phrase}\n{best_proba:.3f} {best_decoded}")
    max_iter = 100000

    list_val = [i for i in char_to_num]
    count = 0

    while count < max_iter:
        # Switching decoding table
        guess_tmp = decrypt_guess
        i = list_val[np.random.randint(0, size)]
        j = list_val[np.random.randint(0, size)]

        guess_tmp[i], guess_tmp[j] = guess_tmp[j], guess_tmp[i]

        decoded = "".join([guess_tmp[i] for i in coded_phrase])
        proba = get_plausibility(decoded, bigrams, char_to_num)

        rand_acceptation = np.random.rand() < 0.05

        if proba > best_proba or rand_acceptation:
            if not rand_acceptation:
                best_proba = proba
                best_decoded = decoded
                print(f"{proba:.3f} {decoded} {count}")
            decrypt_guess = guess_tmp
        count += 1

    print(f"\n{best_proba:.3f} {best_decoded}")
