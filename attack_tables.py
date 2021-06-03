from constants import *
from bb_operations import *


# PAWN ATTACKS
# @njit(nb.uint64(nb.b1, nb.uint8))
def mask_pawn_attacks(color, sq):
    bb = set_bit(EMPTY, sq)

    if not color:  # WHITE
        s1 = (bb >> np.uint(7)) & ~FILES[File.A]
        s2 = (bb >> np.uint(9)) & ~FILES[File.H]
    else:  # BLACK
        s1 = (bb << np.uint(7)) & ~FILES[File.H]
        s2 = (bb << np.uint(9)) & ~FILES[File.A]

    return s1 | s2


pawn_attacks = np.fromiter(
    (mask_pawn_attacks(color, sq) for color in range(2) for sq in squares), dtype=np.uint64, count=2 * 64)
pawn_attacks.shape = (2, 64)


# KNIGHT
# @njit(nb.uint64(nb.uint8))
def mask_knight_attacks(sq):
    bb = set_bit(EMPTY, sq)

    m1 = ~(FILES[File.A] | FILES[File.B])
    m2 = ~FILES[File.A]
    m3 = ~FILES[File.H]
    m4 = ~(FILES[File.H] | FILES[File.G])

    s1 = (bb & m1) << np.uint8(6)
    s2 = (bb & m2) << np.uint8(15)
    s3 = (bb & m3) << np.uint8(17)
    s4 = (bb & m4) << np.uint8(10)

    s5 = (bb & m4) >> np.uint8(6)
    s6 = (bb & m3) >> np.uint8(15)
    s7 = (bb & m2) >> np.uint8(17)
    s8 = (bb & m1) >> np.uint8(10)

    return s1 | s2 | s3 | s4 | s5 | s6 | s7 | s8


knight_attacks = np.fromiter(
    (mask_knight_attacks(i) for i in squares),
    dtype=np.uint64,
    count=64)


# KING
# @njit(nb.uint64(nb.uint8))
def mask_king_attacks(sq):
    bb = set_bit(EMPTY, sq)

    m1 = ~FILES[File.A]
    m2 = ~FILES[File.H]

    nw = (bb & m1) << np.uint8(7)
    n = bb << np.uint8(8)
    ne = (bb & m2) << np.uint8(9)
    e = (bb & m2) << np.uint8(1)
    se = (bb & m2) >> np.uint8(7)
    s = bb >> np.uint8(8)
    sw = (bb & m1) >> np.uint8(9)
    w = (bb & m1) >> np.uint8(1)

    return nw | n | ne | e | se | s | sw | w


king_attacks = np.fromiter(
    (mask_king_attacks(i) for i in squares),
    dtype=np.uint64,
    count=64)


# @njit(nb.uint64(nb.uint8))
def mask_bishop_attacks(sq):
    attacks = EMPTY
    tr = sq // 8
    tf = sq % 8

    for direction in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for i in range(1, 7):
            r = tr + direction[0] * i
            f = tf + direction[1] * i
            if not 0 < r < 7 or not 0 < f < 7:
                break
            attacks |= (BIT << np.uint64(r * 8 + f))

    return attacks


# @njit(nb.uint64(nb.uint8))
def mask_rook_attacks(sq):
    attacks = EMPTY
    tr = sq // 8
    tf = sq % 8

    for direction in (-1, 1):
        for i in range(1, 7):
            r = tr + direction * i
            if not 0 < r < 7:
                break
            attacks |= BIT << np.uint64(r * 8 + tf)

        for i in range(1, 7):
            f = tf + direction * i
            if not 0 < f < 7:
                break
            attacks |= BIT << np.uint64(tr * 8 + f)

    return attacks


# @njit(nb.uint64(nb.uint8, nb.uint64))
def bishop_attacks_on_the_fly(sq, block) -> np.uint64:
    attacks = EMPTY
    tr = sq // 8
    tf = sq % 8

    for direction in ((1, 1), (-1, 1), (1, -1), (-1, -1)):
        for reach in range(1, 7):
            r = tr + direction[0] * reach
            f = tf + direction[1] * reach
            if not 0 <= r <= 7 or not 0 <= f <= 7:
                break
            attacked_bit = BIT << np.uint64(r * 8 + f)
            attacks |= attacked_bit
            if attacked_bit & block:
                break

    return attacks


# @njit(nb.uint64(nb.uint8, nb.uint64))
def rook_attacks_on_the_fly(sq, block) -> np.uint64:
    attacks = EMPTY
    tr = sq // 8
    tf = sq % 8

    for direction in (1, -1):
        for i in range(1, 7):
            r = tr + direction * i
            if not 0 <= r <= 7:
                break
            attacked_bit = BIT << np.uint8(r * 8 + tf)
            attacks |= attacked_bit
            if attacked_bit & block:
                break

        for i in range(1, 7):
            f = tf + direction * i
            if not 0 <= f <= 7:
                break
            attacked_bit = BIT << np.uint8(tr * 8 + f)
            attacks |= attacked_bit
            if attacked_bit & block:
                break

    return attacks


# @njit(nb.uint64(nb.uint8, nb.uint8, nb.uint64))
def set_occupancy(index, bits_in_mask, attack_mask) -> int:
    occupancy = EMPTY

    for count in range(bits_in_mask):

        square = np.uint8(get_ls1b_index(attack_mask))

        attack_mask = pop_bit(attack_mask, square)

        if index & (1 << count):
            occupancy |= BIT << square

    return int(occupancy)


bishop_relevant_bits = [6, 5, 5, 5, 5, 5, 5, 6,
                        5, 5, 5, 5, 5, 5, 5, 5,
                        5, 5, 7, 7, 7, 7, 5, 5,
                        5, 5, 7, 9, 9, 7, 5, 5,
                        5, 5, 7, 9, 9, 7, 5, 5,
                        5, 5, 7, 7, 7, 7, 5, 5,
                        5, 5, 5, 5, 5, 5, 5, 5,
                        6, 5, 5, 5, 5, 5, 5, 6]

rook_relevant_bits = [12, 11, 11, 11, 11, 11, 11, 12,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      11, 10, 10, 10, 10, 10, 10, 11,
                      12, 11, 11, 11, 11, 11, 11, 12]

# big endian
MAGIC_BISHOP = np.array([0x88840101012000, 0x10808041101000, 0x1090c00110511001, 0x2124000208420208,
                         0x800102118030400, 0x10202120024080, 0x24a4208221410, 0x10a0a1023020080,
                         0x804230108200880, 0x804230108200880, 0x8001010402090010, 0x8000042020080,
                         0x4200012002440000, 0x80084010228880a0, 0x4244049014052040, 0x50100083024000,
                         0x6401884600c280, 0x1204028210809888, 0x8000a01402005002, 0x41d8a021a000400,
                         0x41d8a021a000400, 0x201a102004102, 0x408010842041282, 0x201a102004102,
                         0x1004501c200301, 0xa408025880100100, 0x1042080300060a00, 0x4100a00801110050,
                         0x11240100c40c0040, 0x24a0281141188040, 0x8100c4081030880, 0x20c310201002088,
                         0x4282180040080888, 0x44200002080108, 0x2404c80a04002400, 0x2020808028020002,
                         0x129010050304000, 0x8020108430092, 0x5600450c884800, 0x5600450c884800,
                         0x2410002102020800, 0x10202004098180, 0x1104000808001010, 0x274802008a044000,
                         0x1400884400a00000, 0x82000048260804, 0x4004840500882043, 0x81001040680440,
                         0x400420202041100, 0x400420202041100, 0x1100300082084211, 0x124081000000,
                         0x405040308000411, 0x1000110089c1008, 0x30108805101224, 0x10808041101000,
                         0x10a0a1023020080, 0x50100083024000, 0x8826083200800802, 0x102408100002400,
                         0x414242008000000, 0x414242008000000, 0x804230108200880, 0x88840101012000],
                        dtype=np.uint64)

# big endian
MAGIC_ROOK = np.array([0x300300a043168001, 0x106610218400081, 0x8200c008108022, 0x201041861017001,
                       0x20010200884e2, 0x205000e18440001, 0x202008104a08810c, 0x800a208440230402,
                       0x80044014200240, 0x40012182411500, 0x3102001430100, 0x4c43502042000a00,
                       0x1008000400288080, 0x806003008040200, 0x4200020801304400, 0x8100640912804a00,
                       0x804000308000, 0x4008100020a000, 0x1001208042020012, 0x400220088420010,
                       0x8010510018010004, 0x8009000214010048, 0x6445006200130004, 0xa008402460003,
                       0x400884010800061, 0xc202401000402000, 0x800401301002004, 0x4c43502042000a00,
                       0x4a80082800400, 0xd804040080800200, 0x60200080e002401, 0x203216082000104,
                       0x60400280008024, 0x9810401180200382, 0x200201200420080, 0x280300100210048,
                       0x80080800400, 0x2010200081004, 0x8089000900040200, 0x40008200340047,
                       0x1040208000400091, 0x10004040002008, 0x82020020804011, 0x5420010220208,
                       0x8010510018010004, 0x5050100088a1400, 0x9008080020001, 0x2001060000408c01,
                       0x10208008a8400480, 0x4064402010024000, 0x2181002000c10212, 0x5101000850002100,
                       0x10800400080081, 0x12000200300815, 0x60200080e002401, 0x4282000420944201,
                       0x80004000608010, 0x2240100040012002, 0x8008a000841000, 0x100204900500004,
                       0x20008200200100c, 0x40800c0080020003, 0x80018002000100, 0x4200042040820d04],
                      dtype=np.uint64)


def little_to_big(little):
    """change array from little to big endian"""

    big = [0 for _ in range(64)]
    c = 0

    for rank in range(56, -1, -8):
        for file in range(8):
            big[c] = hex(little[rank + file])
            c += 1
    return big


rook_magic_numbers = [0x8a80104000800020,
                      0x140002000100040,
                      0x2801880a0017001,
                      0x100081001000420,
                      0x200020010080420,
                      0x3001c0002010008,
                      0x8480008002000100,
                      0x2080088004402900,
                      0x800098204000,
                      0x2024401000200040,
                      0x100802000801000,
                      0x120800800801000,
                      0x208808088000400,
                      0x2802200800400,
                      0x2200800100020080,
                      0x801000060821100,
                      0x80044006422000,
                      0x100808020004000,
                      0x12108a0010204200,
                      0x140848010000802,
                      0x481828014002800,
                      0x8094004002004100,
                      0x4010040010010802,
                      0x20008806104,
                      0x100400080208000,
                      0x2040002120081000,
                      0x21200680100081,
                      0x20100080080080,
                      0x2000a00200410,
                      0x20080800400,
                      0x80088400100102,
                      0x80004600042881,
                      0x4040008040800020,
                      0x440003000200801,
                      0x4200011004500,
                      0x188020010100100,
                      0x14800401802800,
                      0x2080040080800200,
                      0x124080204001001,
                      0x200046502000484,
                      0x480400080088020,
                      0x1000422010034000,
                      0x30200100110040,
                      0x100021010009,
                      0x2002080100110004,
                      0x202008004008002,
                      0x20020004010100,
                      0x2048440040820001,
                      0x101002200408200,
                      0x40802000401080,
                      0x4008142004410100,
                      0x2060820c0120200,
                      0x1001004080100,
                      0x20c020080040080,
                      0x2935610830022400,
                      0x44440041009200,
                      0x280001040802101,
                      0x2100190040002085,
                      0x80c0084100102001,
                      0x4024081001000421,
                      0x20030a0244872,
                      0x12001008414402,
                      0x2006104900a0804,
                      0x1004081002402]

bishop_magic_numbers = [0x40040844404084,
                        0x2004208a004208,
                        0x10190041080202,
                        0x108060845042010,
                        0x581104180800210,
                        0x2112080446200010,
                        0x1080820820060210,
                        0x3c0808410220200,
                        0x4050404440404,
                        0x21001420088,
                        0x24d0080801082102,
                        0x1020a0a020400,
                        0x40308200402,
                        0x4011002100800,
                        0x401484104104005,
                        0x801010402020200,
                        0x400210c3880100,
                        0x404022024108200,
                        0x810018200204102,
                        0x4002801a02003,
                        0x85040820080400,
                        0x810102c808880400,
                        0xe900410884800,
                        0x8002020480840102,
                        0x220200865090201,
                        0x2010100a02021202,
                        0x152048408022401,
                        0x20080002081110,
                        0x4001001021004000,
                        0x800040400a011002,
                        0xe4004081011002,
                        0x1c004001012080,
                        0x8004200962a00220,
                        0x8422100208500202,
                        0x2000402200300c08,
                        0x8646020080080080,
                        0x80020a0200100808,
                        0x2010004880111000,
                        0x623000a080011400,
                        0x42008c0340209202,
                        0x209188240001000,
                        0x400408a884001800,
                        0x110400a6080400,
                        0x1840060a44020800,
                        0x90080104000041,
                        0x201011000808101,
                        0x1a2208080504f080,
                        0x8012020600211212,
                        0x500861011240000,
                        0x180806108200800,
                        0x4000020e01040044,
                        0x300000261044000a,
                        0x802241102020002,
                        0x20906061210001,
                        0x5a84841004010310,
                        0x4010801011c04,
                        0xa010109502200,
                        0x4a02012000,
                        0x500201010098b028,
                        0x8040002811040900,
                        0x28000010020204,
                        0x6000020202d0240,
                        0x8918844842082200,
                        0x401001102902002]


rook_masks = np.fromiter(
    (mask_rook_attacks(sq) for sq in squares),
    dtype=np.uint64)

bishop_masks = np.fromiter(
    (mask_bishop_attacks(sq) for sq in squares),
    dtype=np.uint64)


# @njit(nb.uint64[:,:](nb.uint64[:,:], nb.b1))
def init_sliders(attacks, bish):
    for sq in squares:

        attack_mask = bishop_masks[sq] if bish else rook_masks[sq]

        relevant_bits_count = count_bits(attack_mask)
        occupancy_indices = 1 << relevant_bits_count

        for index in range(occupancy_indices):
            if bish:
                occupancy = set_occupancy(index, relevant_bits_count, attack_mask)
                magic_index = (occupancy * bishop_magic_numbers[sq]) >> (64 - bishop_relevant_bits[sq])
                attacks[sq][magic_index] = bishop_attacks_on_the_fly(sq, np.uint64(occupancy))

            else:   # rook
                occupancy = set_occupancy(index, relevant_bits_count, attack_mask)
                magic_index = (occupancy * rook_magic_numbers[sq]) >> (64 - rook_relevant_bits[sq])
                attacks[sq][magic_index] = rook_attacks_on_the_fly(sq, np.uint64(occupancy))

    return attacks


bishop_attacks = init_sliders(np.zeros((64, 512), dtype=np.uint64), bish=True)
rook_attacks = init_sliders(np.zeros((64, 4096), dtype=np.uint64), bish=False)


# @njit(nb.uint64(nb.uint8, nb.uint64))
def get_bishop_attacks(sq, occ):
    occ &= bishop_masks[sq]
    # occ *= MAGIC_BISHOP[sq]
    occ *= bishop_magic_numbers[sq]
    occ >>= np.uint64(64) - bishop_relevant_bits[sq]

    return bishop_attacks[sq][occ]


# @njit(nb.uint64(nb.uint8, nb.uint64))
def get_rook_attacks(sq, occ):
    occ &= rook_masks[sq]
    # occ *= MAGIC_ROOK[sq]
    occ *= rook_magic_numbers[sq]
    occ >>= np.uint64(64) - rook_relevant_bits[sq]

    return rook_attacks[sq][occ]


# @njit(nb.uint64(nb.uint8, nb.uint64))
def get_queen_attacks(sq, occ):
    return get_rook_attacks(sq, occ) | np.uint64(get_bishop_attacks(sq, occ))
