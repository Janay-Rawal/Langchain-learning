import hashlib
import itertools
import string

target_suffix = "10f4a7"

for combo in itertools.product(string.ascii_lowercase, repeat=4):
    word = ''.join(combo)
    md5_hex = hashlib.md5(word.encode()).hexdigest()
    if md5_hex.endswith(target_suffix):
        print("Found:", word, md5_hex)
        break