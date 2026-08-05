# THEORY.md -- The Mathematics of a Sparse, Trainable, Rewireable Weight Store

These are lecture notes, not a reference card. They assume you are a
strong graduate student who knows linear algebra, basic probability, and
what a neural network is, but who has never touched sparse matrix
formats, coding theory, or quantized training. Every claim used by the
implementation is either proved here, derived here, or explicitly
labeled as measured or heuristic. Formulas are ASCII; read `^` as power,
`<=` as at-most, `sum_j` as summation over j.

Notation used throughout:

    W          an M x N weight matrix, M outputs (rows), N inputs (cols)
    nnz        number of stored (potential) synapses
    R          synapses stored per row      (capacity, not live count)
    C          synapses stored per column
    S          = M/C = N/R when both are integers (proved equal, L4)
    LOG_S, R_LOG, C_LOG   log2 of S, R, C (all powers of two here)
    KC = LOG_S + R_LOG    bits in a column key
    KR = LOG_S + C_LOG    bits in a row key
    w4         a 4-bit weight code, decoded through a 16-entry table
    imp        a 4-bit importance nibble
    live       a slot whose byte (imp<<4 | w4) is nonzero

---------------------------------------------------------------------------
## Lecture 0. The design problem

We want to store the weights of a sparse layer that is TRAINED ONLINE --
weights change every step, and the graph itself changes on a slower
clock (synaptogenesis and pruning). Four requirements, each of which
will get a precise mechanism:

1. Fast: every training pass should touch each synapse O(1) times with
   no searching, no sorting, and ideally no atomic memory operations.
2. Memory efficient: bits per synapse is the budget. A naive trainable
   sparse layer costs 192 bits per synapse (Lecture 1). We will reach 32.
3. Sparse in operation, not just in storage: when only a fraction of
   inputs or output-gradients are nonzero, work should scale with that
   fraction where the memory system allows it.
4. Capable of reaching a good graph: the structural moves (add, remove,
   relocate a synapse; expand or contract total capacity) must be cheap,
   exact, and expressive enough that a learning rule can steer the graph
   toward wherever it wants to go.

The tension is that 1 and 2 pull toward rigid, implicit structure while
4 pulls toward arbitrary graphs. The resolution of these notes is a
structured family of graphs (Lecture 4) rich enough for 4, rigid enough
for 1 and 2, paired with a second, fully general codec (Lecture 5) and a
controller that chooses per layer (Lecture 7 and 9).

A remark on cost models before we start. "Fast" on modern hardware is
mostly a statement about memory behavior: how many bytes move, whether
accesses are contiguous (coalesced), whether lanes of a SIMD unit or GPU
warp follow the same control flow, and whether two lanes ever contend
for the same address (atomics). Operation counts matter less. Keep that
lens on for everything below.

---------------------------------------------------------------------------
## Lecture 1. Storing a sparse matrix twice, and why that is expensive

A sparse matrix is a set of triples (i, j, w). The three classical
layouts:

    COO   three parallel arrays row[], col[], val[] of length nnz.
    CSR   sort by row; store col[] and val[] plus row_ptr[M+1], where
          row i's entries occupy [row_ptr[i], row_ptr[i+1]).
    CSC   the same with rows and columns exchanged.

Training needs BOTH orientations. The forward pass y = W x wants, for
each output i, the list of its inputs: that is a row traversal (CSR).
The backward input-gradient dx = W^T dy wants, for each input j, the
list of outputs it feeds: a column traversal (CSC). You can compute
either pass from either layout, but the "wrong" layout forces a scatter:
many lanes adding into the same output cell, which on parallel hardware
means atomic adds. Gather form -- one owner per output cell, lanes read
many places and reduce locally -- needs no atomics. So the comfortable
design is: forward gathers through the row view, backward-dx gathers
through the column view, and each pass has exactly one writer per
output. That is the shape of every kernel in this codebase, and it is
why we insist on two views.

Two views cost memory. The baseline trainable layer (see
fixed_margin_ell.cuh, the first iteration in this lineage) stores:

    2 x fp32 weight copies                      64 bits / synapse
    2 x int32 index arrays (col ids, row ids)   64
    2 x int32 permutations linking the copies   64
                                               ---
                                               192 bits / synapse

The permutations (r2c, c2r) exist because a synapse lives at one
position in the row view and another in the column view, and updates
applied in one view must find the mirror slot in the other. The rest of
these notes is a systematic demolition of each line of that table.

Exercise 1.1. Convince yourself that with both views present, the
forward gather and the backward-dx gather are the SAME kernel with the
two views' arrays swapped (rows <-> columns, R <-> C). This symmetry is
implemented literally: one gather kernel serves both directions.

---------------------------------------------------------------------------
## Lecture 2. Degree-regular bipartite graphs

Model the sparsity pattern as a bipartite graph: left vertices are rows,
right vertices are columns, edges are synapses. The DEGREE of a vertex
is its edge count; row degrees and column degrees are also called the
matrix's MARGINS.

We now impose the central structural axiom: every row has exactly R
edges and every column exactly C ("doubly regular", "biregular",
"constant margins"). Three immediate consequences.

(2.1) The handshake identity. Count edges by rows and by columns:

    M * R = nnz = N * C.

Dividing by M*N gives r/N = C/M: the DENSITY along a row equals the
density along a column, always, even when the raw counts R and C are
wildly different (they differ by exactly the aspect ratio, C/R = M/N).
This is two lines of double counting, but it silently governs
everything: it is why the two views have the same total size, why the
blocks of Lecture 4 are square, and why capacity parameters cannot be
chosen independently.

(2.2) Feasibility. Which (R, C) are possible for given (M, N)? Let
g = gcd(M, N), M = g*m', N = g*n' with gcd(m', n') = 1. Then M*R = N*C
forces n' | R, so all solutions are

    R = t * n',   C = t * m',   t = 1, 2, 3, ...

Feasible densities are quantized in units of 1/g. (For irregular margin
sequences the existence question is answered by the Gale-Ryser theorem;
we only need the regular case.)

(2.3) The ELL collapse. ELLPACK ("ELL") is the format that stores a
FIXED number of entries per row as two dense [rows x width] arrays
(indices and values), so CSR's row_ptr becomes the closed form i*R and
vanishes. Normally ELL is a compromise -- ragged rows are padded to the
longest and the padding is waste. Constant margins are the one case
where ELL is exact: zero padding, perfect load balance (every row is
identical work, so a static schedule is already optimal), and row blocks
that are exact multiples of the cache line when R is a multiple of the
SIMD/warp width. The same collapse, transposed, gives the column view as
a dense [N x C] array.

(2.4) A constructive existence proof: the wrapped diagonal. Enumerate
edge slots k = 0 .. M*R - 1 and place edge k at

    (row, col) = (k div R, k mod N).

Row i receives k in [iR, iR+R): R consecutive residues mod N, distinct
because R <= N. Column j receives every k with k = j (mod N); since the
range has length M*R = N*C, that is exactly C hits. Both margins hold.
Pictorially the pattern is a thick diagonal stripe that never truncates
at the boundary -- it wraps the torus:

        j 0 1 2 3 4 5 6 7 8
      i +------------------
      0 | X X X . . . . . .
      1 | . . . X X X . . .        M=6 N=9 R=3 C=2
      2 | . . . . . . X X X
      3 | X X X . . . . . .
      4 | . . . X X X . . .
      5 | . . . . . . X X X

Exercise 2.1. Show that restricting the wrapped diagonal to a subset of
slot indices s in [0, L), L < R, does NOT generally give equal column
counts. (This is why "sheets" in the codebase are built as independent
offset diagonals rather than prefixes of one diagonal.)

(2.5) Moving through the space of patterns. Two margin-preserving moves:

  - Interchange (Ryser 1957): if (i, j) and (i', j') are edges while
    (i, j') and (i', j) are not, replace the former pair by the latter.
    Ryser's theorem: ANY two 0-1 matrices with the same margins are
    connected by a sequence of interchanges. So the doubly-regular
    invariant costs no reachability at all -- every biregular pattern is
    reachable from every other without ever breaking the margins, even
    transiently.
  - Curveball trade (Strona et al. 2014): pick two rows, compute the
    intersection and symmetric difference of their column sets, and
    redistribute the symmetric difference between them (keeping the
    split sizes). Each trade equals a batch of interchanges; the
    resulting Markov chain mixes much faster, and the primitive it needs
    -- set operations on two sorted lists -- is exactly a sorted-merge
    kernel. Drawing partners from the symmetric difference is also what
    guarantees a trade never creates a duplicate edge.

This is the state of the art BEFORE hashing: exact formats, zero
padding, margin-preserving rewiring. What it does not yet give us is
cheap cross-view addressing (the 64 bits of permutations) or small
indices. That requires structure inside the ids themselves.

---------------------------------------------------------------------------
## Lecture 3. Hashing the identifiers

We will split every id into a small "bank" part and a large "quotient"
part, and impose regularity per bank. For that to be safe, bank
membership must not correlate with the data's structure. Enter
multiplicative hashing.

(3.1) Odd numbers are invertible mod 2^k. The units of the ring
Z / 2^k Z are exactly the odd residues. So j -> (a * j) mod 2^k is a
bijection for any odd a, with an inverse computable by Newton's
iteration: if a*x = 1 (mod 2^m), set x' = x * (2 - a*x); then writing
a*x = 1 + t*2^m,

    a*x' = (1 + t*2^m)(1 - t*2^m) = 1 - t^2 * 2^(2m) = 1  (mod 2^(2m)).

Correct bits DOUBLE per iteration; starting from x = 1 (correct mod 2
since a is odd), five iterations reach 2^32. This is `modinv_pow2`.

(3.2) High bits, not low bits. Suppose banks were the LOW bits:
bank(j) = (a*j) mod R. For any stride D divisible by R (the Atari raster
has vertical-neighbor stride D = 480 = 15 * 32),

    a*(j + D) = a*j + a*D = a*j  (mod R),

because R | D implies R | a*D for every a. Low-bit banks can NEVER
separate such a pair; multiplication cannot repair divisibility.
Taking banks from the HIGH bits of the product,

    bank(j) = floor( ((a*j) mod 2^KC) / 2^LOG_S ),

fixes this: a*D mod 2^KC is a fixed nonzero offset whose high bits
perturb the bank for almost all j (Knuth's multiplicative hashing).
This distinction was gotten wrong once during development -- the low-bit
version shipped for a turn -- and the raster counterexample above is the
concrete thing it broke. Remember it.

(3.3) The multiplier is a PARAMETER, and structured data has good ones.
Blind hashing gives you average-case bank balance. But if every row's
neighborhood shares an offset family (receptive fields: every row is
{base + o : o in Offsets}), then a single multiplier that spreads the
set {a*o mod 2^KC} across distinct high-bit blocks works for EVERY row
simultaneously. Searching a few dozen candidate multipliers ("audition",
Lecture 7) is therefore not a tweak; on structured graphs it is the
difference between 86.3% and 0.0% conflict (measured, Lecture 9).

From here on: stored keys are

    jp = (aC * j) mod 2^KC,   split as  jp = [ s : R_LOG | v : LOG_S ]
    ip = (aR * i) mod 2^KR,   split as  ip = [ p : C_LOG | u : LOG_S ]

with s the column-bank, p the row-bank, and v, u the quotients. Real ids
are recovered with the inverse multipliers. A codec that wants no
relabeling at all (the packed codec) simply sets all multipliers to 1;
the kernels do not branch on this.

---------------------------------------------------------------------------
## Lecture 4. The permutation-plane theorem

Impose, on top of double regularity, the BANKED constraint:

    (B1) each row has exactly one synapse in each column-bank s;
    (B2) each column has exactly one synapse in each row-bank p.

Since there are R column-banks and C row-banks, (B1) and (B2) imply the
margins R and C automatically. The payoff is the following.

THEOREM (plane decomposition). Let S := M / C and S' := N / R. Then
(a) S = S' (both equal 2^LOG_S here), and (b) a graph satisfies (B1) and
(B2) iff for every pair (p, s) the edges between row-bank p and
column-bank s form a PERFECT MATCHING between the S row-quotients u and
the S column-quotients v. Consequently the entire graph is a choice of
R*C independent permutations pi_{p,s} of the set {0, ..., S-1}.

Proof. (a) M*R = N*C (handshake) is equivalent to M/C = N/R. (b) Fix
(p, s). By (B1), each of the S rows in bank p contributes exactly one
edge into column-bank s; by (B2), each of the S columns in bank s
receives exactly one edge from row-bank p. A bipartite graph on S + S
vertices in which every vertex has degree exactly 1 is a perfect
matching, i.e. a permutation u -> v. Conversely, any choice of R*C
permutations gives each row one edge per column-bank (R total) and each
column one per row-bank (C total). QED.

Corollaries, which are the entire engineering payoff:

(4.1) Closed-form addresses. Store pi as an array indexed by (ip, s) and
its inverse by (jp, p):

    row slot of (i, j):     k = ip * R + s        (s = jp >> LOG_S)
    column slot of (i, j):  q = jp * C + p        (p = ip >> LOG_S)

Both are arithmetic. The 64 bits of r2c / c2r permutation arrays from
Lecture 1 become zero bits: cross-view addressing is a shift, an AND,
and a multiply. There is nothing to search and nothing to synchronize.

(4.2) Index size. A slot stores only its quotient: LOG_S bits in each
view, 2 * LOG_S total (e.g. 24 bits at S = 4096), versus 2 x 32 before.

(4.3) Rewiring is a transposition. Swapping the images of u1, u2 inside
one plane pi_{p,s} changes which columns rows i1, i2 own, patches four
stored fields and (if live) two weight bytes, and preserves (B1)/(B2)
EXACTLY, even mid-operation. There is no transient illegal state.

(4.4) Reachability, honestly stated. Transpositions generate the
symmetric group, so trades reach the entire family: (S!)^(R*C) graphs.
That family is a strict SUBSET of all biregular graphs -- the price of
(B1) is that a row cannot hold two synapses whose columns share a bank
(equivalently, whose hashed keys agree in the top R_LOG bits). With
interleaved/hashed banks this forbids only pseudorandom pairs, and the
audition (3.3) can often empty the constraint of practical content, but
it is a real restriction and you should know it is there. The fully
general biregular space remains reachable via the packed codec plus
curveball trades (Lecture 2), which is one of the two reasons the system
keeps two codecs.

(4.5) Pedigree. This is the structure of quasi-cyclic LDPC codes:
protograph blocks realized as permutation (often circulant) matrices.
Twenty years of coding theory says the family is expressive (good
expansion, controllable girth) and hardware-loving (banked memories,
routing-network implementations). We inherit both properties and the
literature (Lecture 10).

Exercise 4.1. Show that a trade's two partner rows must lie in the same
row-bank p, and its two traded columns in the same column-bank s.
(Hint: a transposition lives inside one plane.)

Exercise 4.2. Derive the "one synapse per bank" constraint directly from
(B1), and show that with bank = high bits of a hashed key, any R
CONSECUTIVE real column ids can still be fully connected by one row for
typical multipliers. (This is why receptive fields survive.)

---------------------------------------------------------------------------
## Lecture 5. Bit budgets: the ladder from 192 to 32, and the rival codec

The banked format's per-slot cost:

    quotient in row view (pi entry)      LOG_S      = 12
    quotient in column view (inverse)    LOG_S      = 12
    weight code (fp4)                    4
    importance                           4
    permutations / sync machinery        0   (closed form, single store)
                                        ----
                                        32 bits per stored slot

The single weight store deserves a sentence: because both views can
compute the OTHER view's slot in O(1) (4.1), weights are stored once (we
keep them row-major, since forward and the update pass traverse rows)
and the second copy plus its incremental synchronization cursor simply
cease to exist. The backward-dx pass pays one scattered READ per synapse
through the closed form; that is the only scattered access left in the
system, and it is read-only.

(5.1) The packed rival: anchor + fixed-width deltas. Sort each row's
column ids; split into groups of gsz (the SIMD/warp width); per group
store an anchor (the first id) and encode every member as a fixed-width
delta from the anchor:

    width w = ceil(log2(span + 1)),  span = max(id) - anchor,
    group cost = header (anchor, w) + offset + gsz * w bits.

Anchoring to the GROUP START rather than the previous element removes
the serial dependency between consecutive deltas, so all gsz lanes
decode independently -- shift, mask, add. (This mirrors the SIMD uleb128
scheme already in sili; the fixed width additionally buys O(1) random
access WITHIN the group and removes the varint continuation-bit tax and
its byte-granular state machine, which matters for GPU lockstep and for
FPGA synthesis. The lineage is the database literature's
frame-of-reference / PForDelta / SIMD-BP128 family.)

Costs, per index: about log2(expected span). For a row whose live ids
cluster within a window D, span ~ D and the cost is log2(D) -- 8 bits
for D = 256. For ids spread uniformly over N = 131072 with gsz = 32,
the expected span is nearly the whole range and the cost is ~17 bits.
Compare uleb128 at the same spreads: a 12-bit delta costs 2 bytes = 16
bits (7 payload bits per byte), so fixed-width matches or beats varints
at every spread while decoding branchlessly.

(5.2) The backpointer trick. The packed codec has no closed-form
cross-view addressing, but note what the missing information actually
is: the global mirror position decomposes as (row id, slot-within-row),
and the column view already stores the row id. Only the slot index is
missing -- log2(R) bits, not 32. Storing that "slotback" field per
column slot restores O(1) backward weight addressing for R_LOG extra
bits per synapse (5 bits at R = 32). Alternatively spend nothing and
binary-search the row group in O(log gsz).

(5.3) Accounting discipline: compare PER STORED SLOT, never per live
synapse, when choosing between formats. Both formats pay for capacity;
bits-per-live = bits-per-slot * capacity/live scales identically in the
fill factor, so per-live comparisons just multiply both sides by the
same number and per-slot comparisons decide identically -- but mixing
the two conventions (as an early version of the controller did) makes an
underfilled layer look arbitrarily bad in whichever format you divided
by live. Measured on a uniformly spread graph: packed 50.9 vs banked
32.0 bits per slot. On clustered sensory layers the inequality flips.

(5.4) Which codec when. Clustered spans (sensory projections, learned
topography): packed wins on bits and imposes no bank constraint. Spread
spans (a global recurrent fabric under exploratory rewiring): banked
wins on bits AND on structural-op cost (O(1) trades vs group
re-encodes) AND on addressing. This is a per-layer, per-epoch empirical
question, which is why it is answered by a measuring controller with
hysteresis rather than by a design-time decision (Lecture 7 and 9).

(5.5) Below 32: selling back the redundant half. The 2*LOG_S index bits
store a permutation AND its inverse, but the family's entropy is only
log2(S!)/S ~ LOG_S - 1.44 bits per slot (10.56 at S = 4096): nearly half
the index is redundancy purchased for O(1) BIDIRECTIONAL addressing.
The succinct-permutation construction (Munro, Raman, Raman, Rao 2003)
sells it back at a latency price: keep pi explicit, drop the inverse
array entirely, and answer pi^{-1}(v) by walking v's cycle with shortcut
pointers placed every t steps. Marks go at cycle positions 0, t, 2t,
..., so no inter-mark gap exceeds t and the back jump always lands
strictly BEHIND the target; mark at t-1, 2t-1, ... instead and the wrap
gap reaches 2t-1, the jump can land past the target, and the O(t) bound
fails -- a bug this codebase shipped for exactly one test run.

Per-slot cost replacing the LOG_S-bit inverse: 1 bit of marker bitmap,
0.25 bits of rank counters, LOG_S/t bits of shortcuts. The O(t)
plane-local pi fetches are paid only by backward-dx and the growth
probe; forward, update, and trades are untouched. Measured (CPU, S=4096,
average hops = t + 1 almost exactly):

    layout          bits/slot   dx throughput vs full
    full inverse       32.0          1.0x
    succinct t=4       24.3          8.7x slower
    succinct t=6       23.3         13x
    succinct t=8       22.8         18x

The wall-clock multiplier exceeds t because the walk also forfeits the
SIMD vectorization the dense-inverse kernel enjoys; on a GPU, where
thread-parallelism across queries survives (each thread walks its own
cycle, bounded by ~t steps), the expected penalty is closer to t itself.
Trades dirty a plane's shortcuts; lookups on dirty planes stay CORRECT
via an O(cycle) fallback (shortcuts are accelerators, never truth), and
finalize rebuilds dirty planes in O(S) each, amortized over a batch.

Further cuts on the menu: importance to 2 bits or externalized into
trace machinery (-2 to -4); 3-bit weight codes with per-row scale bytes
(-2, run the learning experiment first); a Benes-network representation
(~LOG_S - 0.5 bits with BOTH directions derivable, but a 2*log2(S)
serial-dependent switch chain per lookup and O(S) rerouting per trade --
the FPGA-native answer, wrong for GPU hot paths); and circulant planes
at ~0 index bits with no per-synapse rewiring at all (static
quasi-cyclic LDPC, the far endpoint of the spectrum). One tempting
non-lever: shrinking LOG_S by raising R and C (2 bits per doubling)
multiplies the slot count, so at a fixed live budget the per-live bits
get WORSE unless fill is maintained -- the right memory lever there is
tight capacity plus the demotion-free expansion of Lecture 7, not
bigger banks. The floor for this family with fp4 weights and any access
cost is ~ (LOG_S - 1.44) + 4 ~ 14.6 bits per slot.

---------------------------------------------------------------------------
## Lecture 6. Learning at four bits per weight

(6.1) The codebook. Weights are codes into a 16-entry table (an
E2M1-flavored set {0, +-0.5, 1, 1.5, 2, 3, 4, 6} * scale here; the table
is a free parameter). Code 0 decodes to exactly 0.0 and doubles as the
SILENCE mark: a silent slot multiplies everything by zero, so the
forward and backward kernels need no branch to skip it, and pruning is
literally one byte write. Silence being in-band is what makes structural
sparsity free at run time.

(6.2) Deterministic rounding kills online learning at this precision.
The smallest code gap is g = 0.5 * scale. Any gradient step smaller than
g/2 rounds back to the same code: a dead zone. Online RL gradients are
almost always smaller than g/2, so a deterministically rounded fp4 layer
simply stops learning.

(6.3) Stochastic rounding (SR). Round a target t lying between adjacent
codes lo <= t <= hi to hi with probability (t - lo)/(hi - lo), else lo.
Then

    E[Q(t)] = lo + (t - lo) = t          (unbiased),
    Var[Q(t)] = (t - lo)(hi - t) <= (g/2)^2.

Unbiasedness is the whole point: a stream of sub-gap gradient steps
moves the EXPECTED weight exactly as fp32 would; the weight performs a
biased random walk whose drift is the true gradient. The variance term
is real injected noise -- read it as temperature. In a system that
regulates its own dynamics (a branching-ratio homeostat), that
temperature is visible to the regulator; widen the regulator's tolerance
for fp4 layers rather than letting density control chase quantization
noise.

(6.4) The parked-weight theorem (why the zero-update guard exists).
Suppose a weight receives NO gradient (masked input, zero error) but we
re-quantize it anyway. Under unbiased SR the weight value is a
martingale -- zero drift -- but its variance grows with every
re-rounding: floating-point epsilon puts the recomputed target slightly
off-grid, giving a small hop probability each step, and a driftless
walker diffuses across the codebook. E stays put; the weight does not.
Hence the rule, present in every update kernel: if the computed step is
exactly zero, pass the byte through untouched. This bug class was found
by inspection here and would have been a slow, mysterious quality leak.

(6.5) Reproducibility. The SR randomness comes from a counter-free
integer hash (Wang) of (slot id XOR step seed): stateless, parallel,
and bit-reproducible. Corollary discovered by a failing test: because
the two codecs assign different slot ids to the same synapse, identical
training histories produce DIVERGENT weight trajectories across codecs
-- determinism holds within a representation, not across a format
switch. If cross-representation determinism matters, key the hash on the
codec-independent pair (i, j) instead; both update kernels already have
j in hand at the hash site.

(6.6) The importance nibble as the entire optimizer state. The fused
update IS the optimizer -- there are no master weights, no moment
arrays, no gradient buffers scaling with nnz -- so per-parameter memory
is exactly the stored slot, and the only remaining question is whether
importance can live in its 4 bits under the same discipline as the
weights. The rule: importance accrues the synapse's contribution and is
countered by its gradient pressure,

    d = imp_lr * |x_j| * ( |w_ij| - |dy_i| )    per touched synapse,

applied as ONE signed dithered step (integer part plus Bernoulli on the
fraction, drawn from an independently salted hash so weight hops and
importance bumps do not correlate). Although the two terms are stated as
"contribution at forward, gradient at backward", the per-step loop sees
the same x in both passes, so folding both into the update kernel is
exactly equivalent in expectation and keeps the forward pass pure read.
E[imp] tracks the clamped integral of d with sqrt(T) signal-to-noise --
the same theorem as fp4 SGD, applied one nibble to the left.

Two emergent properties. First, a weight parked at zero has zero
contribution but nonzero gradient pressure, so its importance drains to
the floor and the byte reaches 0x00: ZOMBIES SILENCE THEMSELVES, and
the explicit zombie-cleanup pass of Lecture 7.6 becomes a special case
of the importance dynamics (the lower clamp is 1 while the weight code
is nonzero, 0 once it is not). Second, saturation at 15 flattens ranking
among the strongest synapses, so pruning comparators should break
importance ties by weight magnitude.

Measured (kernel-level, not simulation): under constant drift
+-0.040/step for 120 steps, mean importance across 1024 synapses hit
12.54 and 3.46 against clamp-free predictions of 12.80 and 3.20 -- the
deviations point exactly where clipping at 15 and at the floor pulls
them. After 800 training steps on the teacher-student task, importance
separated teacher-position synapses (mean 9.12) from junk (4.50), and
pruning 40% of live synapses by importance cost +0.00054 MSE against
+0.01302 for random and +0.00018 for magnitude: 24x less damage than
random, at parity with magnitude, from 4 stochastic bits.

The ledger, asserted rather than asserted-in-prose: persistent storage
is 32.000 bits per parameter nominal AND allocated (the slack words
vanish at scale); per-step transients (x, y, dy, dx, bitmaps, active
list) total O(M + N) -- 2.5 MB against 16 MB of weights at the reference
sizes. The one way this budget breaks is per-synapse eligibility traces
for multi-step credit assignment: that is per-parameter state, and it
must receive the same in-slot stochastic treatment at whatever bit width
is affordable, or it doubles the layer.

---------------------------------------------------------------------------
## Lecture 7. Structural plasticity: growth, pruning, capacity

Separate two things ruthlessly: POLICY (which synapse deserves to exist;
importance scores, outer-product candidates) and MECHANISM (how a
decision becomes bytes). This lecture is mechanism; Lecture 8 analyzes
the policy used in the tests.

(7.1) Prune = one byte write (code 0, imp 0). In the banked codec the
wiring (the permutation entry) is deliberately left in place: if the
same pair is re-grown later, its slot is already pointing at the right
column and the commit is a byte write with no trade at all.

(7.2) Grow, banked. A candidate (i, j) has a PREDETERMINED slot, by
(4.1): k = ip*R + s. The probe is two byte reads at closed-form
addresses:

    row-bank check:    is syn[k] == 0?   (row i's bank s free?)
    column-bank check: let u2 = inv[jp*C + p], k2 = ((p<<LOG_S)|u2)*R + s;
                       is syn[k2] == 0?  (column j's residue-p slot free?)

If either is occupied, the probe DECODES THE OCCUPANT -- the exact live
synapse (i, j_other) or (i_other, j) standing in the way -- and returns
it as an explicit, adjudicable block reason. Eviction becomes a policy
call ("is the newcomer more important than this incumbent?") instead of
an invisible failure. If both are free, commit is:

    - if pi[k] != v_target: one transposition inside plane (p, s)
      between u and u2. Both affected slots are SILENT, so the trade has
      ZERO semantic effect -- it is pure bookkeeping that rewires empty
      capacity -- and by (4.3) the invariants hold even mid-operation.
    - write the byte at k.

O(1), exact, and on the GPU it is a TRADE-stage record plus a byte
scatter; the probe and dedup stay host-side where candidate selection
already lives.

(7.3) Capacity expansion is provably lossless. Banks are the high bits
of the hashed key. Raise R_LOG by 1 (with the SAME multiplier): the new
bank index is the old one with one additional, higher bit appended --
the product a*j mod 2^(KC+1) agrees with a*j mod 2^KC on all low bits.
Therefore

    bank'(j1) = bank'(j2)  ==>  bank(j1) = bank(j2):

every old bank SPLITS in two, collisions-after are a subset of
collisions-before, and rebuilding at the larger R_LOG can never demote a
synapse. The implementation refuses to merely trust this: `assemble`
hard-asserts zero collisions on every expansion, re-proving the theorem
at run time. Two testable predictions follow and were both confirmed:

    - Pigeonhole starvation bound: a layer needing d live synapses per
      row against B < d banks must have at least d - B blocked positions
      per row. (Teacher d = 12, B = 8: >= 4000 blocked over 1000 rows;
      measured 4691.)
    - Unblocking rate: a previously colliding pair separates iff the two
      keys differ in the appended bit, which under the hash is a fair
      coin: expect one half. (Measured on a frozen list across the
      expansion: 50%.)

Contraction is the mirror without the free lunch: lowering R_LOG MERGES
banks pairwise, new collisions can appear, and the rebuild runs the
importance CONTESTS (keep the argmax-importance synapse per merged bank;
losers exit as a regrow queue rather than being deleted -- the growth
machinery, which exists precisely to place synapses well, re-places
them). Expansion is safe by theorem; contraction is safe by policy.
Design your controller asymmetrically to match.

(7.4) Conversion between codecs uses the same contests. packed ->
banked must fit an arbitrary graph into the banked family; synapses
sharing a (row, column-bank) or (column, row-bank) fight, argmax
importance wins, losers regrow. How bad can it be? Blind-hash worst case
is the balls-in-bins occupancy formula: r live synapses into B banks,

    E[demoted] = r - B * (1 - (1 - 1/B)^r),

which at r = 18, B = 32 predicts 22.6% on the row side alone; the
measured total with the column contest stacked on top was 28.7% on a
uniform-random graph. That is why the controller GATES this direction on
the audited demotion fraction and why the audition (3.3) matters: on the
structured receptive-field graph the searched multiplier took demotion
from 86.3% (identity banking) to 0.0%. Random graphs are simultaneously
banked's best case for memory and worst case for conversion; real
learned graphs, which have structure, sit far from that worst case --
and importance weighting biases the losses toward synapses that mattered
least anyway.

(7.5) The counters are a sensor. Grow attempts, block reasons, and
zombie prunes (7.6) are maintained by the mechanism for free, and they
constitute a memory-pressure signal: sustained high block rate under
high grow demand says EXPAND; high zombie rate with low block rate says
CONTRACT. This is the capacity analog of the bits-per-slot statistic
that drives codec choice: in both cases the structure itself reports
which way to move, and the controller only supplies hysteresis and a
dwell time so noise cannot flap it.

(7.6) Zombie pruning. A synapse whose weight has learned to (exactly
representable) zero occupies a bank, blocks candidates, and contributes
nothing. Culling zeros continuously is a degenerate, importance-free
special case of bottom-k pruning and empirically it is load-bearing: in
the churn experiment it freed the banks that let real candidates land.
Because pruning leaves wiring intact (7.1), a culled synapse that was
about to matter is re-grown for one byte.

(7.7) Format portability, demonstrated. The lifecycle above -- scored
candidates, capacity blocking, eviction adjudication, zombie cleanup,
contraction contests, expansion -- is defined at the interchange level,
and the SAME policy code runs on both codecs; what changes underneath is
the mechanism's cost profile. Banked: O(1) probe/commit, bank-shaped
blocking, theorem-guarded expansion. Packed (standing in for the
slack-space uleb encoder): in-place pruning, growth by batched group
re-encode, blocking on raw row/column capacity only, and rebuild-based
capacity change that is lossless in BOTH directions because there are no
banks to collide and no contests to run. Running the identical
teacher-student churn experiment on packed reproduces the same arc with
codec-correct constants: the pre-expansion wall is HARD (only saturated
rows block: 1005 blocked teacher slots vs banked's 4691, because the
collision wall bites at every occupancy while the capacity wall bites
only at the rim), expansion frees 100% of row blocks (vs banked's
theoretically exact 50%), and the floor is lower.

That last contrast deserves its formula. Against a RANDOM target graph
of degree d per row, a banked row with B banks can ever hold at most

    B * (1 - (1 - 1/B)^d)

of the d target positions (occupancy bound): 8.6 of 12 at B = 16, which
is why banked's hold MSE (0.0204) sat far above packed's (0.0069) at
equal R in this worst-case-for-banks experiment. Structured targets plus
the audition escape the bound (Lecture 3.3, Lecture 9); random targets
do not, and banked buys coverage there only by raising B (85% at B = 32,
94% at B = 64) -- cheap in ops via demotion-free expansion, but paid in
slots. Controller consequence: sustained banked block-rate under growth
demand is a THREE-way signal -- expand banks, or convert the layer to
packed, decided by the measured bits-per-slot of each option. Plasticity
pressure joins memory pressure as an input to codec choice.

---------------------------------------------------------------------------
## Lecture 8. Growth policy: what to grow, and a failure worth learning

The mechanism accepts any candidate stream. The tests use the classic
gradient-correlation policy, and its analysis explains a real bug.

Model: teacher y* = W* x, student yhat = W x, error e = yhat - y*, with
independent zero-mean inputs, E[x_j^2] = m2 (activity probability folded
in). For any pair (i, j),

    E[ e_i * x_j ] = sum_j' (W - W*)_{i j'} E[x_j' x_j]
                   = (W_{ij} - W*_{ij}) * m2,

by independence (all cross terms vanish). At an ABSENT slot (W_ij = 0)
this is -W*_{ij} * m2: the time-averaged outer product of error and
input, evaluated at empty positions, IS (up to scale and sign) the map
of missing weights -- equivalently, the loss gradient with respect to
weights that do not exist yet. Grow where its magnitude is largest, with
initial sign opposite to the score. This is precisely the "grow by
gradient magnitude" rule of the modern sparse-training literature
(RigL), and the traces already present in an eligibility-trace system
are its natural streaming estimator.

The failure mode: a SINGLE sample of e * x is a rank-one matrix -- pure
coincidence of "which input was loud when this row's error was loud."
Grown from single samples, the test's layer added almost entirely
wrong-position synapses, which then dutifully learned toward zero
(becoming the zombies of 7.6), and MSE stayed bit-flat at the teacher's
output variance. The signal-to-noise of the accumulated score grows as
sqrt(T): per-sample noise is rms(e) * rms(x) while the signal is
|W*| * m2 * T after T samples, so a window of T ~ 32 samples flipped the
candidate list from junk to teacher positions and the learning curve
from flat to textbook. Moral, stated as a rule: NEVER grow from
instantaneous coincidence; grow from accumulated correlation. If your
outer-product synaptogenesis runs on a single step's activity, this trap
is armed.

What the full churn experiment then shows (numbers in the ledger,
Appendix B): quality improves while capacity-starved, stalls against a
provable wall, resumes after a provably lossless expansion, holds, and
under magnitude-based contraction memory falls 29% while quality
IMPROVES (pruning eats junk before signal) before degrading smoothly
past a visible knee. "Expand or contract memory while maintaining or
improving quality" is not a slogan; it is a curve with assertions on it.

---------------------------------------------------------------------------
## Lecture 9. Verification as part of the mathematics

A structure this rigid is a gift to testing: the invariants are crisp
enough to audit exhaustively, and the theorems make quantitative
predictions that a harness can check against reality. The shipped tests
are organized around that idea, and a student extending the system
should preserve the style.

  - Invariant audits. (B1)/(B2) plus the plane theorem reduce to one
    checkable property: pi and inv are mutual inverses at EVERY slot.
    The audit runs after conversion, after updates, after trades, after
    expansion. Any structural bug anywhere trips it.
  - Independent references. Kernels are checked against scalar
    references that recompute slot addresses from scratch; update
    checks are BYTE-exact because reference and kernel share the
    stochastic quantizer but not the addressing.
  - Prediction checks. The 50% unblock rate, the pigeonhole bound, the
    balls-in-bins demotion estimate, and the analytic starting MSE
    (teacher output variance: 12 * E[w^2] * p_active * E[x^2], which
    matched to three figures) are asserted, not just admired.
  - Compiler as witness. The claim "warp lanes translate to SIMD lanes"
    is checked by reading gcc's vectorization report for the four hot
    loops, not by assuming it.
  - What is NOT yet verified: the GPU shaders on hardware (the CPU
    mirror shares their math line for line, so upload the small-config
    fixtures and diff each stage's outputs), and performance on the
    actual target memory systems.

---------------------------------------------------------------------------
## Lecture 10. Annotated reading list

Degree-constrained matrices and rewiring:
  - Gale (1957); Ryser (1957). Existence (Gale-Ryser) and the
    interchange theorem: same-margin 0-1 matrices are connected by 2x2
    swaps. The reachability backbone of Lecture 2.
  - Brualdi, "Combinatorial Matrix Classes" (2006). The reference text
    for matrices with fixed margins.
  - Strona et al. (2014), the curveball algorithm; Carstens (2015) for
    its connectivity/uniformity proof. Fast margin-preserving mixing.

Coding theory (the structural cousins):
  - Gallager (1962), LDPC codes; Tanner (1981), graph representations.
  - Fossorier (2004) and the quasi-cyclic LDPC literature: permutation /
    circulant block structure, girth and expansion control, hardware
    mappings. Lecture 4's family, two decades early.

Hashing and integer compression:
  - Knuth, TAOCP vol. 3, section 6.4: multiplicative hashing and why the
    high bits of a*j are the good bits.
  - Zukowski et al. (2006), PFOR; Lemire and Boytsov (2015), SIMD-BP128
    and friends: the anchor + fixed-width bit-packing lineage of
    Lecture 5, from the inverted-index world.

Low-precision training:
  - Gupta et al. (2015), "Deep Learning with Limited Numerical
    Precision": stochastic rounding makes low-bit training work.
  - Croci et al. (2022), stochastic rounding survey: the martingale /
    variance viewpoint of Lecture 6.

Sparse training and structure search (where "reach a good graph" lives):
  - Han et al. (2015): magnitude pruning.
  - Mocanu et al. (2018), SET: prune-and-regrow sparse evolutionary
    training.
  - Bellec et al. (2018), Deep Rewiring: rewiring as sampling.
  - Evci et al. (2020), RigL: grow by gradient magnitude, prune by
    weight magnitude -- the policy analyzed in Lecture 8.
  - Frankle and Carbin (2019), lottery tickets: why good sparse
    structures exist to be found at all.

---------------------------------------------------------------------------
## Appendix A. Formula sheet

    handshake            M*R = N*C        density  R/N = C/M
    feasibility          R = t*n', C = t*m',  M = g*m', N = g*n', g = gcd
    wrapped diagonal     edge k -> (k div R, k mod N)
    keys                 jp = (aC*j) mod 2^KC = [s|v],  ip = (aR*i) mod 2^KR = [p|u]
    slots                k = ip*R + s,   q = jp*C + p
    inverse (Newton)     x' = x*(2 - a*x) doubles correct bits mod 2^k
    banked bits/slot     2*LOG_S + 8
    packed bits/slot     ~ log2(span_row) + log2(span_col) + R_LOG + 8 + overhead
    SR                   E[Q(t)] = t,  Var <= (gap/2)^2
    growth score         E[e_i x_j] = (W_ij - W*_ij) * E[x^2],  SNR ~ sqrt(T)
    occupancy demotion   r - B*(1 - (1 - 1/B)^r)
    bank split           bank'(j1)=bank'(j2) => bank(j1)=bank(j2)
    succinct inverse     LOG_S + 1 + 0.25 + LOG_S/t bits/slot, query O(t)
    family entropy       log2(S!)/S ~ LOG_S - 1.44 bits/slot (the floor)

## Appendix B. Measured-results ledger (single-core sandbox, gcc 13, AVX2)

    vectorization        4/4 hot inner loops auto-vectorized (SSE + AVX2)
    stride audition      identity 86.3% demoted -> searched multiplier 0.0%
    blind-hash demotion  28.7% measured vs 22.6% analytic row-side floor
    per-slot bits        packed 50.9 vs banked 32.0 on a spread graph
    starvation           4691 teacher slots blocked (bound: >= 4000)
    bank split           50% of blocked slots freed (prediction: 1/2)
    churn curve          MSE 0.0531 -> 0.0275 starved -> 0.0209 expanded;
                         contraction: -29% memory with MSE improving to
                         0.0200, knee ~3600 live, smooth to 0.0310 at -65%
    CPU throughput       fwd 334 (packed) / 94 (banked) Msyn/s, 1 core --
                         logic-validation numbers, not the target matchup
    packed churn         same policy code, same arc: hard capacity wall
                         (1005 blocked, saturated rows only), expansion
                         freed 100% of row blocks, hold MSE 0.0069 vs
                         banked 0.0204 at equal R on a random teacher
                         (the occupancy coverage bound, 8.6/12 at B=16)
    stochastic imp       counter means 12.54 / 3.46 vs clamp-free
                         predictions 12.80 / 3.20; teacher 9.12 vs junk
                         4.50 after training; 40%-prune damage +0.00054
                         (importance) vs +0.01302 (random), +0.00018
                         (magnitude); ledger 32.000 bits/param exact,
                         transients 2.53 MB vs 16 MB weights
    succinct inverse     avg hops t+1 (5.00 / 7.00 / 8.99 / 12.98 at
                         t = 4/6/8/12); 24.3 / 23.3 / 22.8 / 22.3
                         bits/slot; dx-only slowdown 8.7x at t=4;
                         dirty-plane fallback verified correct

Everything above is reproducible from test_dual_codec.cpp and
test_synapto.cpp; the ledger is what those printed on the run that
shipped with these notes.

    Stars are sparse. Donuts are uniform. Choose the geometry
    that lets you address both. -- end of notes
