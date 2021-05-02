import string
from collections import Counter


class CaesarCypher:

    def __init__(self, key):
        self.alphabet = " " + string.ascii_lowercase + string.digits
        self.cryptor = {j: self.alphabet[(i + key) % len(self.alphabet)] for i, j in enumerate(self.alphabet)}
        self.decryptor = {j: i for i, j in self.cryptor.items()}

    def crypt(self, string_input):
        return "".join([self.cryptor[i] for i in string_input])

    def decrypt(self, string_input):
        return "".join([self.decryptor[i] for i in string_input])

    def crack(self, string_input):
        most_common = Counter(string_input).most_common(2)[0]
        # MIGHT works because we "know" that most common letter should be "e"
        # so compute distance between most common found and "e"
        key = self.alphabet.index(most_common[0]) - self.alphabet.index("e")
        # key = self.alphabet.index(most_common[0])
        print("INDEX (guessed)", key, "OR", key - len(self.alphabet))
        return CaesarCypher(key).decrypt(string_input)


if __name__ == '__main__':
    a = CaesarCypher(6)
    # d = "You and I know what infatuation is You know all laws of infatuation and so do I"   # no e in phrase
    d = "Quoique ce detail ne touche en aucune maniere au fond meme de ce que je vais vous parler"
    d = d.lower()
    b = a.crypt(d)
    c = a.decrypt(b)
    e = a.crack(b)
    print(f"IN : {d}\nOUT: {b}\nDEC: {c}\nCRK: {e}")
