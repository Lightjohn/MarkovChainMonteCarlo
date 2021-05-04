import os
import string

import jellyfish
import matplotlib.pyplot as plt
import numpy as np
from numpy import exp, log

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


def get_min_hamming_distance(_input_word, _dictionary):
    min_hamming = 999
    for w in _dictionary:
        hamming_distance = jellyfish.hamming_distance(_input_word, w)
        if hamming_distance < min_hamming:
            min_hamming = hamming_distance
    return min_hamming


def get_dict_plausibility(_input_string, _dictionary):
    score = 0
    all_words = list(filter(None, _input_string.split(" ")))
    for _word in all_words:
        w_len = len(_word)
        w_score = get_min_hamming_distance(_word, _dictionary)
        # score is between 1 (no difference) and 0 (all different)
        score += max((w_len - w_score) / w_len, 0)
    return score / len(all_words)


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


def apply_permutation(code: dict, _size):
    code_copy = code.copy()
    rand_i = list_val[np.random.randint(0, _size)]
    rand_j = list_val[np.random.randint(0, _size)]

    code_copy[rand_i], code_copy[rand_j] = code_copy[rand_j], code_copy[rand_i]
    return code_copy


if __name__ == '__main__':
    # Creating stats on input text
    # input_file = "MiserablesV1.txt"
    input_file = "Swann.txt"
    input_file_path = os.path.join("data", input_file)
    chars = {}
    use_digits = False
    char_to_num = create_indirection_table(include_digits=use_digits)
    stats = {i: 0 for i in char_to_num}
    size_solution = len(char_to_num)
    bigrams = np.zeros((size_solution, size_solution))
    for word in extract_words(input_file_path, include_digits=use_digits):
        compute_char_relation(word, bigrams, char_to_num)
        compute_stats(word, stats)

    # Normalising table
    # will be errors if no char are found in all text like "5" in original "Swann.txt"
    EPSILON = 1e-6  # Epsilon for when 0
    bigrams = log(bigrams / bigrams.sum(axis=1)[:, None] + EPSILON)

    # Producing graph
    # save_table(bigrams, char_to_num, False)

    # Creating ciphered phrase
    original_phrase = "quoique ce detail ne touche en aucune maniere au fond meme de ce que je vais vous parler"
    original_phrase = "moi je ne crois pas qu il y ait de bonne ou de mauvaise situation"
    original_phrase_size = len(original_phrase)
    caesar_cipher = CaesarCipher(8, include_digits=use_digits)
    coded_phrase = caesar_cipher.crypt(original_phrase)
    input_probability = get_plausibility(original_phrase, bigrams, char_to_num)

    current_guess = make_first_proposition(coded_phrase, stats, char_to_num)

    best_guess = current_guess
    best_decoded = "".join([current_guess[i] for i in coded_phrase])
    best_probability = get_plausibility(best_decoded, bigrams, char_to_num)

    print(f"INPUT:\n{input_probability:.3f} {original_phrase}")
    print(f"CODED: {coded_phrase}")
    print(f"{best_probability:.3f} {best_decoded}\n")

    max_iter = 200000
    alpha = 1

    list_val = [i for i in char_to_num]
    count = 0

    current_probability = best_probability

    print(f"PHASE 1: Using char probability")
    while count < max_iter:
        # Switching decoding table
        guess_tmp = apply_permutation(current_guess, size_solution)

        decoded = "".join([guess_tmp[i] for i in coded_phrase])
        tmp_probability = get_plausibility(decoded, bigrams, char_to_num)

        # Test whether move should be accepted
        x = np.random.rand()
        p = exp(alpha * (tmp_probability - current_probability) * original_phrase_size)

        if p > x:
            current_guess = guess_tmp.copy()
            current_probability = tmp_probability
            if tmp_probability > best_probability:
                best_guess = current_guess
                best_probability = tmp_probability
                best_decoded = decoded
                print(f"{tmp_probability:.3f} {decoded} {count}")
        count += 1

    print(f"\nORIG: {original_phrase}")
    print(f"{best_probability:.3f} {best_decoded}")

    print(f"PHASE 2: Dictionary time")
    dictionary = set(extract_words(input_file_path, include_digits=use_digits))

    max_iter = 2000
    temperature = 0.05
    rho = 0.999
    count = 0
    GAMMA = 1
    # Starting from best
    current_guess = best_guess
    word_score = get_dict_plausibility(best_decoded, dictionary)
    cur_score = GAMMA * best_probability + word_score

    while count < max_iter:
        guess_tmp = apply_permutation(current_guess, size_solution)
        decoded = "".join([guess_tmp[i] for i in coded_phrase])
        # this compute the mean of incorrect char per words in the phrase

        # dict plausibility is between 0 and 1
        tmp_probability = get_dict_plausibility(decoded, dictionary)
        tmp_word_score = get_plausibility(decoded, bigrams, char_to_num)
        tmp_score = GAMMA * tmp_probability + tmp_word_score

        # Test whether move should be accepted
        x = np.random.rand()
        p = np.exp((tmp_score - cur_score) / temperature)
        temperature = temperature * rho

        if p > x:
            current_guess = guess_tmp.copy()
            current_probability = tmp_probability
            if tmp_probability > best_probability:
                best_guess = current_guess
                best_probability = tmp_probability
                best_decoded = decoded
                print(f"{tmp_probability:.3f} {decoded} {count}")
        count += 1

    # https://en.wikipedia.org/wiki/String_metric
    # Now apply jellyfish.hamming_distance(a,b) en replace words with distance one
    # create dictionary from input
    # check every words and for distance of 1 identify error, apply fix and rerun
