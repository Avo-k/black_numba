from numba import njit
import numba as nb
import numpy as np

"""
This file contains a variety of functions for manipulating bitboards

big-endian rank-file mapping:

 0  1  2  3  4  5  6  7         A8 B8 C8 D8 E8 F8 G8 H8
 8  9 10 11 12 13 14 15         A7 B7 C7 D7 E7 F7 G7 H7
16 17 18 19 20 21 22 23         A6 B6 C6 D6 E6 F6 G6 H6
24 25 26 27 28 29 30 31    =    A5 B5 C5 D5 E5 F5 G5 H5
32 33 34 35 36 37 38 39         A4 B4 C4 D4 E4 F4 G4 H4
40 41 42 43 44 45 46 47         A3 B3 C3 D3 E3 F3 G3 H3
48 49 50 51 52 53 54 55         A2 B2 C2 D2 E2 F2 G2 H2
56 57 58 59 60 61 62 63         A1 B1 C1 D1 E1 F1 G1 H1

"""


@njit(nb.b1(nb.uint64, nb.uint8), cache=True)
def get_bit(bb, sq):
    return bb & (1 << sq)


@njit(nb.uint64(nb.uint64, nb.uint8), cache=True)
def set_bit(bb, sq):
    return bb | (1 << sq)


@njit(nb.uint64(nb.uint64, nb.uint8), cache=True)
def pop_bit(bb, sq):
    return bb & ~(1 << sq)


@njit(nb.uint8(nb.uint64), cache=True)
def count_bits(bb) -> int:
    c = 0
    while bb:
        c += 1
        bb &= bb - np.uint64(1)
    return c


@njit(nb.uint8(nb.uint64), cache=True)
def get_ls1b_index(bb) -> int:
    return count_bits((bb & -bb) - 1)


def print_bb(bb):
    print("\n")
    for rank in range(8):
        r = ""
        for file in range(8):
            sq = rank * 8 + file
            r += f" {'1' if get_bit(bb, sq) else '·'} "
        print(8 - rank, r)
    print("   A  B  C  D  E  F  G  H")

    print("Bitboard:", bb)


def bb_print(bb):
    bb = bin(bb)[2:]
    bb = ('0' * (64 - len(bb))) + bb
    bb = ''.join(reversed(bb))
    for y in range(8):
        i = (7 - y) * 8
        print(bb[i:i+8])
