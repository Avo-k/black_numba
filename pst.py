from constants import np, njit, \
    opening, end_game, pawn, knight, bishop, rook, queen, king, a8, h8, d5, e5, d4, e4, d3, e3, a1, h1

PawnFileOpening = 5
KnightCentreOpening = 5
KnightCentreEndgame = 5
KnightRankOpening = 5
KnightBackRankOpening = 0
KnightTrapped = 100
BishopCentreOpening = 2
BishopCentreEndgame = 3
BishopBackRankOpening = 10
BishopDiagonalOpening = 4
RookFileOpening = 3
QueenCentreOpening = 0
QueenCentreEndgame = 4
QueenBackRankOpening = 5
KingCentreEndgame = 12
KingFileOpening = 10
KingRankOpening = 10


PawnFile = (-3, -1, +0, +1, +1, +0, -1, -3)
KnightLine = (-4, -2, +0, +1, +1, +0, -2, -4)
KnightRank = (+1, +2, +3, +2, +1, +0, -1, -2)
BishopLine = (-3, -1, +0, +1, +1, +0, -1, -3)
RookFile = (-2, -1, +0, +1, +1, +0, -1, -2)
QueenLine = (-3, -1, +0, +1, +1, +0, -1, -3)
KingLine = (-3, -1, +0, +1, +1, +0, -1, -3)
KingFile = (+3, +4, +2, +0, +0, +2, +4, +3)
KingRank = (-7, -6, -5, -4, -3, -2, +0, +1)


@njit(cache=True)
def init_pst():
    pst = np.zeros((2, 6, 64), dtype=np.int16)

    for sq in range(64):
        # PAWN
        pst[opening][pawn][sq] += PawnFile[sq % 8] * PawnFileOpening

        # KNIGHT
        pst[opening][knight][sq] += KnightLine[sq % 8] * KnightCentreOpening
        pst[opening][knight][sq] += KnightLine[sq // 8] * KnightCentreOpening
        pst[end_game][knight][sq] += KnightLine[sq % 8] * KnightCentreEndgame
        pst[end_game][knight][sq] += KnightLine[sq // 8] * KnightCentreEndgame
        # rank
        pst[opening][knight][sq] += KnightRank[sq // 8] * KnightRankOpening

        # BISHOP
        pst[opening][bishop][sq] += BishopLine[sq % 8] * BishopCentreOpening
        pst[opening][bishop][sq] += BishopLine[sq // 8] * BishopCentreOpening
        pst[end_game][bishop][sq] += BishopLine[sq % 8] * BishopCentreEndgame
        pst[end_game][bishop][sq] += BishopLine[sq // 8] * BishopCentreEndgame

        # ROOK
        pst[opening][rook][sq] += RookFile[sq % 8] * RookFileOpening

        # QUEEN
        pst[opening][queen][sq] += QueenLine[sq % 8] * QueenCentreOpening
        pst[opening][queen][sq] += QueenLine[sq // 8] * QueenCentreOpening
        pst[end_game][queen][sq] += QueenLine[sq % 8] * QueenCentreEndgame
        pst[end_game][queen][sq] += QueenLine[sq // 8] * QueenCentreEndgame

        # KING
        pst[end_game][king][sq] += KingLine[sq % 8] * KingCentreEndgame
        pst[end_game][king][sq] += KingLine[sq // 8] * KingCentreEndgame

        pst[opening][king][sq] += KingFile[sq % 8] * KingFileOpening
        pst[opening][king][sq] += KingRank[sq // 8] * KingRankOpening

    # pawn center control
    for sq in (d3, e3, d5, e5, d4, d4, e4, e4):
        pst[opening][pawn][sq] += 10

    # back rank
    for sq in range(a1, h1 + 1):
        pst[opening][knight][sq] -= KnightBackRankOpening
        pst[opening][bishop][sq] -= BishopBackRankOpening
        pst[opening][queen][sq] -= QueenBackRankOpening

    # trapped knight
    for sq in (a8, h8):
        pst[opening][knight][sq] -= KnightTrapped

    # bishop diagonals
    for coord in range(8):
        sq1 = (coord + 1) * 7
        sq2 = coord * 9
        pst[opening][bishop][sq1] += BishopDiagonalOpening
        pst[opening][bishop][sq2] += BishopDiagonalOpening

    return pst


PST = init_pst()
