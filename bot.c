/* Light Cycles bot (C11).
 *
 * Strategy:
 *  1. Simultaneous-move minimax (alpha-beta, iterative deepening, ~35 ms budget).
 *     Each ply = both bikes move at once; we maximise over our moves and assume the
 *     opponent picks the reply that is worst for us.
 *  2. Leaf evaluation = Voronoi territory (cells we reach strictly before the opponent
 *     minus cells the opponent reaches strictly before us).
 *  3. Once the two bikes are separated (no shared reachable cell) the game is just
 *     "fill your own room": we search for the longest path with a DFS (Warnsdorff order).
 *
 * stdout = moves only, debug goes to stderr.
 */
#include <stdio.h>
#include <string.h>
#include <time.h>

#define S 34 /* padded stride; the grid is surrounded by a ring of occupied cells */
#define CELLS (S * S)
#define INF 100000

static const int DD[4] = {-S, S, -1, 1}; /* UP, DOWN, LEFT, RIGHT */
static const char *NAMES[4] = {"UP", "DOWN", "LEFT", "RIGHT"};

static unsigned char occ[CELLS]; /* 1 = wall / trail / head / outside */
static int W, H, me;

/* ---------- timing ---------- */
static double now_ms(void) {
    return (double)clock() * 1000.0 / CLOCKS_PER_SEC;
}
static double deadline;
static int aborted;
static long nodes;

/* ---------- BFS / Voronoi ---------- */
static int qa[CELLS], qb[CELLS];
static short da[CELLS], db[CELLS];

static int bfs(int start, short *dist, int *q) {
    int head = 0, tail = 0;
    dist[start] = 0;
    q[tail++] = start;
    while (head < tail) {
        int c = q[head++];
        for (int d = 0; d < 4; d++) {
            int n = c + DD[d];
            if (!occ[n] && dist[n] < 0) {
                dist[n] = dist[c] + 1;
                q[tail++] = n;
            }
        }
    }
    return tail;
}

/* Returns territory difference (mine - theirs); *contact = 1 if some cell is reachable by both. */
static int voronoi(int mp, int op, int *contact) {
    memset(da, 0xFF, sizeof da);
    memset(db, 0xFF, sizeof db);
    int na = bfs(mp, da, qa);
    int nb = bfs(op, db, qb);
    int mine = 0, theirs = 0, c = 0;
    for (int i = 1; i < na; i++) {
        int x = qa[i];
        if (db[x] < 0) mine++;
        else {
            c = 1;
            if (da[x] < db[x]) mine++;
        }
    }
    for (int i = 1; i < nb; i++) {
        int x = qb[i];
        if (da[x] < 0) theirs++;
        else if (db[x] < da[x]) theirs++;
    }
    if (contact) *contact = c;
    return mine - theirs;
}

/* ---------- minimax over simultaneous moves ---------- */
static int search(int depth, int mp, int op, int alpha, int beta, int ply, int *bestMove, const int *order) {
    if (depth == 0) return voronoi(mp, op, NULL);
    if ((++nodes & 15) == 0 && now_ms() > deadline) aborted = 1;
    if (aborted) return 0;

    int best = -INF - 1;
    for (int oi = 0; oi < 4; oi++) {
        int m = order ? order[oi] : oi;
        int mn = mp + DD[m];
        int mlegal = !occ[mn];
        int v = INF + 1;
        for (int o = 0; o < 4; o++) {
            int on = op + DD[o];
            int olegal = !occ[on];
            int val;
            if (!mlegal && !olegal) val = 0;
            else if (!mlegal) val = -INF + ply;
            else if (!olegal) val = INF - ply;
            else if (mn == on) val = 0;
            else {
                occ[mn] = 1;
                occ[on] = 1;
                int b = v < beta ? v : beta;
                val = search(depth - 1, mn, on, alpha, b, ply + 1, NULL, NULL);
                occ[mn] = 0;
                occ[on] = 0;
                if (aborted) return 0;
            }
            if (val < v) v = val;
            if (v <= alpha) break;
        }
        if (v > best) {
            best = v;
            if (bestMove) *bestMove = m;
        }
        if (best > alpha) alpha = best;
        if (alpha >= beta) break;
    }
    return best;
}

/* ---------- longest path in our own room ---------- */
static long dfsCount;
static double dfsDeadline;
static int dfsAbort;

static int freeDeg(int p) {
    int k = 0;
    for (int d = 0; d < 4; d++) k += !occ[p + DD[d]];
    return k;
}

static int dfs(int p) {
    if ((++dfsCount & 255) == 0 && now_ms() > dfsDeadline) dfsAbort = 1;
    if (dfsAbort) return 0;
    int nb[4], deg[4], k = 0;
    for (int d = 0; d < 4; d++) {
        int n = p + DD[d];
        if (!occ[n]) {
            int g = freeDeg(n);
            int j = k++;
            while (j > 0 && deg[j - 1] > g) {
                nb[j] = nb[j - 1];
                deg[j] = deg[j - 1];
                j--;
            }
            nb[j] = n;
            deg[j] = g;
        }
    }
    int best = 0;
    for (int i = 0; i < k; i++) {
        occ[nb[i]] = 1;
        int v = 1 + dfs(nb[i]);
        occ[nb[i]] = 0;
        if (v > best) best = v;
        if (dfsAbort) break;
    }
    return best;
}

static int fillMove(int mp, double budgetEnd) {
    int legal[4], nl = 0;
    for (int d = 0; d < 4; d++)
        if (!occ[mp + DD[d]]) legal[nl++] = d;
    if (nl == 0) return 0;
    if (nl == 1) return legal[0];
    int bestM = legal[0], bestV = -1;
    double start = now_ms();
    double slice = (budgetEnd - start) / nl;
    for (int i = 0; i < nl; i++) {
        int m = legal[i];
        int n = mp + DD[m];
        occ[n] = 1;
        dfsCount = 0;
        dfsAbort = 0;
        dfsDeadline = start + slice * (i + 1);
        int v = 1 + dfs(n);
        occ[n] = 0;
        if (v > bestV) {
            bestV = v;
            bestM = m;
        }
    }
    return bestM;
}

/* ---------- decision ---------- */
static const char *decide(int mp, int op, int emptyCells) {
    double t0 = now_ms();

    /* Fallback: first legal move that is not a head-on collision cell. */
    int fallback = -1;
    for (int d = 0; d < 4; d++)
        if (!occ[mp + DD[d]]) { fallback = d; break; }
    if (fallback < 0) return "UP";

    int contact = 1;
    voronoi(mp, op, &contact);
    if (!contact) return NAMES[fillMove(mp, t0 + 30.0)];

    deadline = t0 + 30.0;
    int bestMove = fallback;
    int order[4] = {0, 1, 2, 3};
    for (int depth = 1; depth <= emptyCells && depth <= 60; depth++) {
        aborted = 0;
        int mv = -1;
        int sc = search(depth, mp, op, -INF - 1, INF + 1, 0, &mv, order);
        if (aborted || mv < 0) break;
        bestMove = mv;
        /* put the best move first for the next iteration */
        int k = 0, tmp[4];
        tmp[k++] = mv;
        for (int d = 0; d < 4; d++)
            if (d != mv) tmp[k++] = d;
        memcpy(order, tmp, sizeof order);
        if (sc >= INF - 100 || sc <= -INF + 100) break; /* forced result found */
    }
    if (occ[mp + DD[bestMove]]) bestMove = fallback;
    return NAMES[bestMove];
}

int main(void) {
    static char row[64];
    int turn;
    if (scanf("%d %d %d", &W, &H, &me) != 3) return 0;
    while (scanf(" turn %d", &turn) == 1) {
        int hx[2], hy[2];
        for (int p = 0; p < 2; p++)
            if (scanf("%d %d", &hx[p], &hy[p]) != 2) return 0;
        memset(occ, 1, sizeof occ);
        int empty = 0;
        for (int y = 0; y < H; y++) {
            if (scanf("%63s", row) != 1) return 0;
            for (int x = 0; x < W; x++) {
                int free_ = row[x] == '.';
                occ[(y + 1) * S + x + 1] = !free_;
                empty += free_;
            }
        }
        int mp = (hy[me] + 1) * S + hx[me] + 1;
        int op = (hy[1 - me] + 1) * S + hx[1 - me] + 1;
        const char *mv = decide(mp, op, empty);
        printf("%s\n", mv);
        fflush(stdout);
    }
    return 0;
}
