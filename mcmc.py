import os
import string

import matplotlib.pyplot as plt
import numpy as np

from Ciphers.CaesarCipher import CaesarCipher
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


def create_indirection_table(include_digits=True, include_uppercase=False):
    digits = string.digits if include_digits else ""
    uppercase = string.ascii_uppercase if include_uppercase else ""
    data = [i for i in " " + string.ascii_lowercase + uppercase + digits]
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
    probability = 0
    input_size = len(_input_string)
    for i in range(input_size - 1):
        x = _bigrams[_input_string[i]]
        y = _bigrams[_input_string[i + 1]]
        probability += probability_table[x][y]
    return probability / input_size


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
    # Creating stats on input text
    input_file = "MiserablesV1.txt"
    # input_file = "Swann.txt"
    chars = {}
    use_digits = False
    char_to_num = create_indirection_table(include_digits=use_digits)
    stats = {i: 0 for i in char_to_num}
    size = len(char_to_num)
    bigrams = np.zeros((size, size))
    for word in extract_words(os.path.join("data", input_file), include_digits=use_digits):
        compute_char_relation(word, bigrams, char_to_num)
        compute_stats(word, stats)

    # Normalising table
    # will be errors if no char are found in all text like "5" in original "Swann.txt"
    bigrams = bigrams / bigrams.sum(axis=1)[:, None]

    # Producing graph
    # save_table(bigrams, char_to_num, False)

    # Creating ciphered phrase
    original_phrase = "quoique ce detail ne touche en aucune maniere au fond meme de ce que je vais vous parler"
    original_phrase_size = len(original_phrase)
    caesar_cipher = CaesarCipher(8, include_digits=use_digits)
    coded_phrase = caesar_cipher.crypt(original_phrase)
    input_probability = get_plausibility(original_phrase, bigrams, char_to_num)

    decrypt_guess = make_first_proposition(coded_phrase, stats, char_to_num)

    best_decoded = "".join([decrypt_guess[i] for i in coded_phrase])
    best_probability = get_plausibility(best_decoded, bigrams, char_to_num)

    print(f"INPUT:\n{input_probability:.3f} {original_phrase}")
    print(f"DECODED: {coded_phrase}")
    print(f"{best_probability:.3f} {best_decoded}\n")

    max_iter = 1000000
    alpha = 2

    list_val = [i for i in char_to_num]
    count = 0

    current_probability = 0

    while count < max_iter:
        # Switching decoding table
        guess_tmp = decrypt_guess
        rand_i = list_val[np.random.randint(0, size)]
        rand_j = list_val[np.random.randint(0, size)]

        guess_tmp[rand_i], guess_tmp[rand_j] = guess_tmp[rand_j], guess_tmp[rand_i]

        decoded = "".join([guess_tmp[i] for i in coded_phrase])
        tmp_probability = get_plausibility(decoded, bigrams, char_to_num)

        # Test whether move should be accepted
        x = np.random.rand()
        p = alpha * (tmp_probability - current_probability) * original_phrase_size

        if p > x:
            decrypt_guess = guess_tmp.copy()
            current_probability = tmp_probability
            if tmp_probability > best_probability:
                best_probability = tmp_probability
                best_decoded = decoded
                print(f"{tmp_probability:.3f} {decoded} {count}")
        count += 1

    print(f"\n{best_probability:.3f} {best_decoded}")
