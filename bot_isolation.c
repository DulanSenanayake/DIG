/* Isolation bot (C11).
 *
 * Strategy: iterative-deepening negamax with alpha-beta pruning, a transposition table,
 * and a Voronoi-territory + mobility evaluation (queen-move distances; the side to move wins ties).
 * When the board is small enough the search reaches the end of the game and plays perfectly.
 *
 * stdout = moves only, debug goes to stderr.
 */
#include <stdio.h>
#include <string.h>
#include <stdint.h>
#include <time.h>

#define S 11 /* padded stride: 9 + 2 border */
#define CELLS (S * S)
#define INF 1000000
#define TT_BITS 18
#define TT_SIZE (1u << TT_BITS)

static const int DD[8] = {-S, -S + 1, 1, S + 1, S, S - 1, -1, -S - 1};

static unsigned char occ[CELLS]; /* 1 = wall / burned / piece / outside */
static int W, H;

static double now_ms(void) { return (double)clock() * 1000.0 / CLOCKS_PER_SEC; }
static double deadline;
static int aborted;
static long nodes;

/* ---------- Zobrist hashing + transposition table ---------- */
static uint64_t zocc[CELLS], zpos[2][CELLS];
typedef struct {
    uint64_t key;
    int score;
    short depth;
    signed char flag; /* 0 exact, 1 lower, 2 upper */
    short best;       /* best destination cell */
} TT;
static TT tt[TT_SIZE];

static uint64_t rng_state = 88172645463325252ULL;
static uint64_t rnd(void) {
    rng_state ^= rng_state << 13;
    rng_state ^= rng_state >> 7;
    rng_state ^= rng_state << 17;
    return rng_state;
}

/* ---------- move generation ---------- */
static int genMoves(int pos, int *out) {
    int n = 0;
    for (int d = 0; d < 8; d++) {
        int c = pos + DD[d];
        while (!occ[c]) {
            out[n++] = c;
            c += DD[d];
        }
    }
    return n;
}

static int countMoves(int pos) {
    int n = 0;
    for (int d = 0; d < 8; d++) {
        int c = pos + DD[d];
        while (!occ[c]) {
            n++;
            c += DD[d];
        }
    }
    return n;
}

/* ---------- evaluation ---------- */
static signed char dA[CELLS], dB[CELLS];
static int qA[CELLS], qB[CELLS];

static int bfs(int start, signed char *dist, int *q) {
    int head = 0, tail = 0;
    dist[start] = 0;
    q[tail++] = start;
    while (head < tail) {
        int c = q[head++];
        int nd = dist[c] + 1;
        for (int d = 0; d < 8; d++) {
            int n = c + DD[d];
            while (!occ[n]) {
                if (dist[n] < 0) {
                    dist[n] = (signed char)nd;
                    q[tail++] = n;
                }
                n += DD[d];
            }
        }
    }
    return tail;
}

/* Score from the point of view of the side to move (pos = mover, op = other). */
static int evaluate(int pos, int op) {
    memset(dA, 0xFF, sizeof dA);
    memset(dB, 0xFF, sizeof dB);
    int na = bfs(pos, dA, qA);
    int nb = bfs(op, dB, qB);
    int ta = 0, tb = 0, contact = 0;
    for (int i = 1; i < na; i++) {
        int c = qA[i];
        if (dB[c] < 0) ta++;
        else {
            contact = 1;
            if (dA[c] <= dB[c]) ta++;
        }
    }
    for (int i = 1; i < nb; i++) {
        int c = qB[i];
        if (dA[c] < 0) tb++;
        else if (dB[c] < dA[c]) tb++;
    }
    int ma = countMoves(pos), mb = countMoves(op);
    if (!contact) return 40 * (ta - tb) + (ma - mb);
    return 10 * (ta - tb) + 2 * (ma - mb);
}

/* ---------- search ---------- */
/* Full hash recomputation: cheap on a <= 9x9 board. */
static uint64_t fullHash(int pos, int op) {
    uint64_t h = zpos[0][pos] ^ zpos[1][op];
    for (int i = 0; i < CELLS; i++)
        if (occ[i]) h ^= zocc[i];
    return h;
}

static int negamax(int pos, int op, int depth, int alpha, int beta, int ply, int *bestOut) {
    if ((++nodes & 31) == 0 && now_ms() > deadline) aborted = 1;
    if (aborted) return 0;

    int moves[64];
    int n = genMoves(pos, moves);
    if (n == 0) return -INF + ply;
    if (depth == 0) return evaluate(pos, op);

    uint64_t h = fullHash(pos, op);
    int alpha0 = alpha, ttBest = -1;
    TT *e = &tt[h & (TT_SIZE - 1)];
    if (e->key == h) {
        ttBest = e->best;
        if (e->depth >= depth && !bestOut) {
            if (e->flag == 0) return e->score;
            if (e->flag == 1 && e->score > alpha) alpha = e->score;
            else if (e->flag == 2 && e->score < beta) beta = e->score;
            if (alpha >= beta) return e->score;
        }
    }

    int key[64];
    for (int i = 0; i < n; i++) key[i] = (moves[i] == ttBest) ? 100000 : countMoves(moves[i]);
    for (int i = 1; i < n; i++) {
        int mv = moves[i], k = key[i], j = i - 1;
        while (j >= 0 && key[j] < k) {
            moves[j + 1] = moves[j];
            key[j + 1] = key[j];
            j--;
        }
        moves[j + 1] = mv;
        key[j + 1] = k;
    }

    int best = -INF - 1, bestMove = moves[0];
    for (int i = 0; i < n; i++) {
        int c = moves[i];
        occ[c] = 1;
        int v = -negamax(op, c, depth - 1, -beta, -alpha, ply + 1, NULL);
        occ[c] = 0;
        if (aborted) return 0;
        if (v > best) {
            best = v;
            bestMove = c;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    if (bestOut) *bestOut = bestMove;

    e->key = h;
    e->score = best;
    e->depth = (short)depth;
    e->best = (short)bestMove;
    e->flag = best <= alpha0 ? 2 : (best >= beta ? 1 : 0);
    return best;
}

int main(void) {
    static char row[64];
    int turn, me;
    if (scanf("%d %d %d", &W, &H, &me) != 3) return 0;
    for (int i = 0; i < CELLS; i++) {
        zocc[i] = rnd();
        zpos[0][i] = rnd();
        zpos[1][i] = rnd();
    }
    while (scanf(" turn %d", &turn) == 1) {
        int px[2], py[2];
        for (int p = 0; p < 2; p++)
            if (scanf("%d %d", &px[p], &py[p]) != 2) return 0;
        memset(occ, 1, sizeof occ);
        for (int y = 0; y < H; y++) {
            if (scanf("%63s", row) != 1) return 0;
            for (int x = 0; x < W; x++) occ[(y + 1) * S + x + 1] = row[x] != '.';
        }
        int pos = (py[me] + 1) * S + px[me] + 1;
        int op = (py[1 - me] + 1) * S + px[1 - me] + 1;

        double t0 = now_ms();
        deadline = t0 + 30.0;
        int moves[64];
        int n = genMoves(pos, moves);
        int bestMove = n > 0 ? moves[0] : pos;
        int lastScore = 0, depthDone = 0;
        if (n > 1) {
            for (int depth = 1; depth <= 90; depth++) {
                aborted = 0;
                int mv = -1;
                int sc = negamax(pos, op, depth, -INF - 1, INF + 1, 0, &mv);
                if (aborted || mv < 0) break;
                bestMove = mv;
                lastScore = sc;
                depthDone = depth;
                if (sc >= INF - 200 || sc <= -INF + 200) break; /* game decided */
            }
        }
        int x = bestMove % S - 1, y = bestMove / S - 1;
        fprintf(stderr, "turn %d: depth %d score %d -> %d %d\n", turn, depthDone, lastScore, x, y);
        printf("%d %d\n", x, y);
        fflush(stdout);
    }
    return 0;
}
