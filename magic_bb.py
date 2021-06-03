from constants import *
from bb_operations import count_bits, print_bb
from attack_tables import mask_bishop_attacks, mask_rook_attacks, bishop_attacks_on_the_fly, rook_attacks_on_the_fly, set_occupancy


def xorshift32():
    number = np.uint(1804289383)
    while True:
        number ^= number << np.uint(13)
        number ^= number >> np.uint(17)
        number ^= number << np.uint(5)
        yield number


xs32 = xorshift32()


def random_u64():
    n1 = np.uint64(next(xs32)) & np.uint64(0xFFFF)
    n2 = np.uint64(next(xs32)) & np.uint64(0xFFFF)
    n3 = np.uint64(next(xs32)) & np.uint64(0xFFFF)
    n4 = np.uint64(next(xs32)) & np.uint64(0xFFFF)

    return n1 | (n2 << np.uint64(16)) | (n3 << np.uint64(32)) | (n4 << np.uint64(48))


def generate_magic_number():
    return random_u64() & random_u64() & random_u64()


def find_magic_number(sq, relevant_bits, bishop):
    """if not bishop then rook"""

    occupancies = np.zeros(4096, dtype=np.uint64)
    attacks = np.zeros(4096, dtype=np.uint64)
    used_attacks = np.zeros(4096, dtype=np.uint64)

    attack_mask = mask_bishop_attacks(sq) if bishop else mask_rook_attacks(sq)

    occupancy_indices = BIT << relevant_bits

    for index in range(occupancy_indices):
        occupancies[index] = set_occupancy(index, relevant_bits, attack_mask)

        attacks[index] = bishop_attacks_on_the_fly(sq, occupancies[index]) if bishop \
            else rook_attacks_on_the_fly(sq, occupancies[index])

    for count in range(10 ** 8):
        magic_number = generate_magic_number()

        # skip inappropriate mn
        if count_bits((attack_mask * magic_number) & RANKS[7]) < 6:
            continue

        used_attacks = np.zeros(4096, dtype=np.uint64)

        fail = False

        for index in range(occupancy_indices):
            if fail:
                break
            magic_index = np.int64((occupancies[index] * magic_number) >> (np.uint64(64) - relevant_bits))

            if not used_attacks[magic_index]:
                used_attacks[magic_index] = attacks[magic_index]
            elif used_attacks[magic_index] != attacks[magic_index]:
                fail = True

        if not fail:  # magic number works
            return magic_number
        else:
            print("fail!")
            return EMPTY

