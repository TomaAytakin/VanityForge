import hashlib
from solders.pubkey import Pubkey

SNS_PROGRAM_ID = Pubkey.from_string("namesLPneVptA9Z5rqUDD9tMTWEJwofgaYwp8cawRkX")
ROOT_DOMAIN = Pubkey.from_string("58PwtjSDuFHuUkYjH9BYnnQKHfwo9reZhC2zMJv9JPkx")

def get_hashed_name(name):
    return hashlib.sha256(name.encode("utf-8")).digest()

def derive(name):
    hashed = get_hashed_name(name)

    # Variant 1: [hash, class(zeros), parent]
    # Class is 32 bytes of zeros
    class_elem = bytes([0] * 32)
    seeds1 = [hashed, class_elem, bytes(ROOT_DOMAIN)]
    pda1, _ = Pubkey.find_program_address(seeds1, SNS_PROGRAM_ID)

    # Variant 2: [hash, parent]
    seeds2 = [hashed, bytes(ROOT_DOMAIN)]
    pda2, _ = Pubkey.find_program_address(seeds2, SNS_PROGRAM_ID)

    print(f"Name: {name}")
    print(f"Variant 1 (hash, zeros, parent): {pda1}")
    print(f"Variant 2 (hash, parent): {pda2}")

if __name__ == "__main__":
    derive("bonfida")
