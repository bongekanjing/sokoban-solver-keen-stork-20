// A room-decomposed, push-optimal Sokoban solver.
//
// Every level drawn here is one hallway with up to four rooms hanging off it, and a box
// pushed out of a door can never be pushed back, so no box ever crosses a door. The level
// therefore splits into one independent room per door: its push-optimum is the sum of the
// rooms' optima, and an exact push-optimal search inside a 6x6 room is small enough to
// finish. Walking between rooms costs no pushes, so the answer is the walk to each room in
// turn followed by that room's push-optimal solution.
//
// Search inside a room is A* over pushes alone: a node is the set of box squares plus the
// region the player can walk, since where inside that region the player stands never
// changes what can be pushed next. The estimate is the cheapest assignment of boxes to
// goals under walking distance, which no sequence of pushes can beat, so the first goal
// node popped is optimal.

import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const IN = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const OUT = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// up, down, left, right: the four steps, in the order the letters are indexed.
const DX = [0, 0, -1, 1];
const DY = [-1, 1, 0, 0];
const STEP = ['u', 'd', 'l', 'r'];

type Level = {
  w: number;
  h: number;
  open: Uint8Array; // 1 where the player may stand and a box may rest
  goal: Uint8Array; // 1 on a goal square
  boxes: number[];
  player: number;
};

function parse(text: string): Level {
  const rows = text.split('\n');
  while (rows.length > 0 && rows[rows.length - 1].trim() === '') rows.pop();
  const h = rows.length;
  let w = 0;
  for (const r of rows) if (r.length > w) w = r.length;
  const open = new Uint8Array(w * h);
  const goal = new Uint8Array(w * h);
  const boxes: number[] = [];
  let player = 0;
  for (let y = 0; y < h; y++) {
    const row = rows[y];
    for (let x = 0; x < w; x++) {
      const ch = x < row.length ? row[x] : '#';
      const at = y * w + x;
      if (ch === '#') continue;
      open[at] = 1;
      if (ch === '.' || ch === '*' || ch === '+') goal[at] = 1;
      if (ch === '$' || ch === '*') boxes.push(at);
      if (ch === '@' || ch === '+') player = at;
    }
  }
  boxes.sort((a, b) => a - b);
  return { w, h, open, goal, boxes, player };
}

/** The squares reachable from `from` by walking, kept off `solid` (walls and boxes). */
function walkFrom(w: number, solid: Uint8Array, from: number, allowed?: Uint8Array): number[] {
  const seen = new Uint8Array(solid.length);
  const stack = [from];
  seen[from] = 1;
  const out: number[] = [];
  while (stack.length > 0) {
    const at = stack.pop() as number;
    out.push(at);
    for (let d = 0; d < 4; d++) {
      const nx = (at % w) + DX[d];
      const ny = Math.floor(at / w) + DY[d];
      if (nx < 0 || nx >= w) continue;
      const next = ny * w + nx;
      if (next < 0 || next >= solid.length) continue;
      if (solid[next] || seen[next]) continue;
      if (allowed && !allowed[next]) continue;
      seen[next] = 1;
      stack.push(next);
    }
  }
  return out.sort((a, b) => a - b);
}

/** One step from `at` towards direction `d`, or -1 when the step leaves the board. */
function shift(w: number, at: number, d: number): number {
  const nx = (at % w) + DX[d];
  if (nx < 0 || nx >= w) return -1;
  return at + DY[d] * w + DX[d];
}

/** The direction facing the other way: up and down are a pair, left and right another. */
const opposite = (d: number): number => d ^ 1;

/**
 * The squares a box can still be pushed onto a goal from, found backwards: a box standing
 * on `from` got there by a push from `from - d`, which needed the player on `from - 2d`.
 * A box on a square this never reaches can never score, so a state holding one is dead.
 */
function reachable(w: number, open: Uint8Array, goal: Uint8Array): Uint8Array {
  const good = new Uint8Array(open.length);
  const queue: number[] = [];
  for (let i = 0; i < open.length; i++) {
    if (open[i] && goal[i]) {
      good[i] = 1;
      queue.push(i);
    }
  }
  for (let q = 0; q < queue.length; q++) {
    const here = queue[q];
    for (let d = 0; d < 4; d++) {
      // A box now on `here` was pushed here in direction d from `back`, and that push
      // needed the player standing one square further back, at `stand`.
      const back = shift(w, here, opposite(d));
      if (back < 0) continue;
      const stand = shift(w, back, opposite(d));
      if (stand < 0) continue;
      if (!open[back] || !open[stand]) continue;
      if (good[back]) continue;
      good[back] = 1;
      queue.push(back);
    }
  }
  return good;
}

/** Walking distance from every square to the nearest square in `from`, -1 where unreachable. */
function distances(w: number, open: Uint8Array, from: number[]): Int32Array {
  const dist = new Int32Array(open.length).fill(-1);
  const queue: number[] = [];
  for (const f of from) {
    dist[f] = 0;
    queue.push(f);
  }
  for (let q = 0; q < queue.length; q++) {
    const here = queue[q];
    for (let d = 0; d < 4; d++) {
      const next = shift(w, here, d);
      if (next < 0) continue;
      if (!open[next] || dist[next] >= 0) continue;
      dist[next] = dist[here] + 1;
      queue.push(next);
    }
  }
  return dist;
}

/** The cheapest total of box-to-goal walking distances over every assignment of boxes to goals. */
function assign(boxes: number[], goals: number[], dist: Int32Array, memo: Map<number, number>): number {
  const n = boxes.length;
  if (n !== goals.length) {
    let sum = 0;
    for (const b of boxes) sum += dist[b] < 0 ? 1 << 20 : dist[b];
    return sum;
  }
  let best = 1 << 20;
  const used = new Array<boolean>(n).fill(false);
  const pick: number[] = [];
  const go = (i: number, sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let g = 0; g < n; g++) {
      if (used[g]) continue;
      const di = dist[goals[g]];
      if (di < 0) continue;
      used[g] = true;
      pick.push(g);
      // The distance is walked from the box, so read the box's own row out of the matrix below.
      go(i + 1, sum);
      pick.pop();
      used[g] = false;
    }
  };
  go(0, 0);
  return best;
}

/**
 * The cheapest assignment, done properly: `cost[i][j]` is box i to goal j, and n is at most
 * four, so every permutation is cheap enough to enumerate.
 */
function cheapest(cost: number[][]): number {
  const n = cost.length;
  if (n === 0) return 0;
  let best = Number.MAX_SAFE_INTEGER;
  const used = new Array<boolean>(n).fill(false);
  const walk = (i: number, sum: number): void => {
    if (sum >= best) return;
    if (i === n) {
      best = sum;
      return;
    }
    for (let j = 0; j < n; j++) {
      if (used[j]) continue;
      used[j] = true;
      walk(i + 1, sum + cost[i][j]);
      used[j] = false;
    }
  };
  walk(0, 0);
  return best;
}

type Push = { index: number; dir: number; walk: string };

/** The walk (letters only, no pushes) from `from` to `to` across open squares and no box. */
function walkPath(w: number, open: Uint8Array, occupied: Uint8Array, from: number, to: number): string | null {
  if (from === to) return '';
  const prev = new Int32Array(open.length).fill(-2);
  const via = new Int8Array(open.length).fill(-1);
  prev[from] = -1;
  const queue = [from];
  for (let q = 0; q < queue.length; q++) {
    const here = queue[q];
    if (here === to) break;
    for (let d = 0; d < 4; d++) {
      const next = shift(w, here, d);
      if (next < 0) continue;
      if (!open[next] || occupied[next] || prev[next] !== -2) continue;
      prev[next] = here;
      via[next] = d;
      queue.push(next);
    }
  }
  if (prev[to] === -2) return null;
  const letters: string[] = [];
  let at = to;
  while (at !== from) {
    letters.push(STEP[via[at]]);
    at = prev[at];
  }
  return letters.reverse().join('');
}

/**
 * Push-optimal solution of one room. `region` marks the room's own squares plus the one
 * hallway square outside its door, and `entry` is that hallway square: the player starts
 * there and steps in. Everything is restricted to the region, so a box pushed out through
 * the door is simply not a move this search can make.
 */
function solveRoom(
  w: number,
  openFull: Uint8Array,
  goalFull: Uint8Array,
  region: Uint8Array,
  boxes: number[],
  entry: number,
): string | null {
  // The room's own board: its squares and the doorway, nothing else.
  const open = new Uint8Array(openFull.length);
  for (let i = 0; i < openFull.length; i++) if (openFull[i] && region[i]) open[i] = 1;
  open[entry] = 1;
  const goal = new Uint8Array(goalFull.length);
  const goals: number[] = [];
  for (let i = 0; i < open.length; i++) {
    if (open[i] && goalFull[i]) {
      goal[i] = 1;
      goals.push(i);
    }
  }
  if (goals.length !== boxes.length) return null;

  const good = reachable(w, open, goal);
  for (const b of boxes) if (!good[b]) return null; // a box that can never score: the room is lost

  // One distance map per goal, so the estimate is a real box-to-goal assignment.
  const toGoal: Int32Array[] = goals.map((g) => distances(w, open, [g]));

  const isGoal = (bs: number[]): boolean => {
    for (const b of bs) if (!goal[b]) return false;
    return true;
  };

  const estimate = (bs: number[]): number => {
    const rows: number[][] = [];
    for (const b of bs) {
      const row: number[] = [];
      for (const d of toGoal) row.push(d[b] < 0 ? 1 << 20 : d[b]);
      rows.push(row);
    }
    return cheapest(rows);
  };
  for (const b of boxes) {
    let any = false;
    for (const d of toGoal) if (d[b] >= 0) any = true;
    if (!any) return null;
  }

  const keyOf = (bs: number[], at: number): string => `${bs.join(',')}|${at}`;

  const occupiedOf = (bs: number[]): Uint8Array => {
    const occ = new Uint8Array(open.length);
    for (const b of bs) occ[b] = 1;
    return occ;
  };

  // The player's position collapses to the lowest square it can walk to, since where it
  // stands inside that region never changes which pushes are open to it.
  const anchor = (bs: number[], player: number): number => {
    const cells = walkFrom(w, occupiedOf(bs), player, open);
    return cells.length > 0 ? cells[0] : player;
  };

  type Node = { boxes: number[]; player: number; key: string; g: number; f: number; parent: Node | null; push: Push | null };

  let start: Node;
  const born = (b: number[], p: number, g: number, parent: Node | null, push: Push | null): Node => {
    const at = anchor(b, p);
    return { boxes: b, player: p, key: keyOf(b, at), g, f: g + estimate(b), parent, push };
  };
  start = born(boxes.slice(), entry, 0, null, null);

  const bestSeen = new Map<string, number>([[start.key, 0]]);

  // A binary heap on f, ties to the deeper node so a short answer wins its level.
  const heap: Node[] = [start];
  const before = (a: Node, b: Node): boolean => (a.f !== b.f ? a.f < b.f : a.g > b.g);
  const up = (i: number): void => {
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!before(heap[i], heap[p])) break;
      [heap[i], heap[p]] = [heap[p], heap[i]];
      i = p;
    }
  };
  const down = (i0: number): void => {
    let i = i0;
    for (;;) {
      const l = 2 * i + 1;
      const r = l + 1;
      let m = i;
      if (l < heap.length && before(heap[l], heap[m])) m = l;
      if (r < heap.length && before(heap[r], heap[m])) m = r;
      if (m === i) break;
      [heap[i], heap[m]] = [heap[m], heap[i]];
      i = m;
    }
  };

  let expansions = 0;
  while (heap.length > 0 && expansions < 500000) {
    const top = heap[0];
    heap[0] = heap[heap.length - 1];
    heap.pop();
    if (heap.length > 0) down(0);
    const known = bestSeen.get(top.key);
    if (known !== undefined && known < top.g) continue;
    expansions++;

    if (isGoal(top.boxes)) {
      const steps: Push[] = [];
      let at: Node | null = top;
      while (at && at.push) {
        steps.push(at.push);
        at = at.parent;
      }
      steps.reverse();
      let text = '';
      for (const s of steps) text += s.walk + STEP[s.dir];
      return text;
    }

    const occ = occupiedOf(top.boxes);
    const reach = walkFrom(w, occ, top.player, open);
    const inReach = new Uint8Array(open.length);
    for (const c of reach) inReach[c] = 1;

    for (let bi = 0; bi < top.boxes.length; bi++) {
      const box = top.boxes[bi];
      for (let d = 0; d < 4; d++) {
        const standOn = shift(w, box, opposite(d)); // the square the player must stand on
        const lands = shift(w, box, d); // where the box would go
        if (standOn < 0 || lands < 0) continue;
        if (!open[standOn] || !inReach[standOn]) continue;
        if (!open[lands] || occ[lands]) continue;
        if (!good[lands]) continue; // a box there can never be pushed to a goal again
        const walk = walkPath(w, open, occ, top.player, standOn);
        if (walk === null) continue;
        const next = top.boxes.slice();
        next[bi] = lands;
        next.sort((a, b) => a - b);
        const after = box; // the player takes the box's old square
        const at = anchor(next, after);
        const key = keyOf(next, at);
        const g = top.g + 1;
        const prev = bestSeen.get(key);
        if (prev !== undefined && prev <= g) continue;
        bestSeen.set(key, g);
        const node = born(next, after, g, top, { index: bi, dir: d, walk });
        heap.push(node);
        up(heap.length - 1);
      }
    }
  }
  return null;
}

/** The rooms of a level: each connected block of floor cut off from the hallway, with its door and entry. */
function rooms(lv: Level): { region: Uint8Array; entry: number; boxes: number[] }[] {
  const { w, open, boxes } = lv;
  const hallRow = Math.floor(lv.player / w);
  // `blocked` is the obstacle map walkFrom expects: walls, plus every square of the hallway.
  const blocked = new Uint8Array(open.length);
  for (let i = 0; i < open.length; i++) blocked[i] = open[i] ? 0 : 1;
  for (let x = 0; x < w; x++) blocked[hallRow * w + x] = 1;

  const seen = new Uint8Array(open.length);
  const out: { region: Uint8Array; entry: number; boxes: number[] }[] = [];

  for (let s = 0; s < open.length; s++) {
    if (!open[s] || blocked[s] || seen[s]) continue;
    const comp = walkFrom(w, blocked, s); // the room's own cells, hallway cut away
    for (const c of comp) seen[c] = 1;
    const mine = comp.filter((c) => boxes.includes(c));
    if (mine.length === 0) continue; // a pocket with no box needs no work
    // The doorway is the room square that touches the hallway; the entry is the hallway
    // square on its other side, which is where the player walks to before stepping in.
    let doorway = -1;
    let entry = -1;
    for (const c of comp) {
      for (let d = 0; d < 4; d++) {
        const nb = shift(w, c, d);
        if (nb < 0) continue;
        if (Math.floor(nb / w) === hallRow && open[nb]) {
          doorway = c;
          entry = nb;
          break;
        }
      }
      if (doorway >= 0) break;
    }
    if (doorway < 0) continue;
    const region = new Uint8Array(open.length);
    for (const c of comp) region[c] = 1;
    out.push({ region, entry, boxes: mine.slice().sort((a, b) => a - b) });
  }
  // Solve in reading order, so the answer is deterministic.
  out.sort((a, b) => a.entry - b.entry);
  return out;
}

function solve(text: string): string {
  const lv = parse(text);
  const { w, open, goal } = lv;
  const boxesAlive = new Uint8Array(open.length);
  for (const b of lv.boxes) boxesAlive[b] = 1;
  let player = lv.player;
  let moves = '';

  const parts = rooms(lv);
  for (const part of parts) {
    // Walk to the square outside the door, across the squares that are free right now.
    const occ = new Uint8Array(open.length);
    for (let i = 0; i < open.length; i++) if (boxesAlive[i]) occ[i] = 1;
    const toDoor = walkPath(w, open, occ, player, part.entry);
    if (toDoor === null) return ''; // an unreachable door: answer nothing rather than go illegal
    moves += toDoor;
    player = part.entry;
    const inside = solveRoom(w, open, goal, part.region, part.boxes.slice(), part.entry);
    if (inside === null) return '';
    // Replay the room's moves on this board to keep the player's square exact.
    for (const ch of inside) {
      const d = STEP.indexOf(ch);
      const next = shift(w, player, d);
      if (next < 0) return '';
      if (boxesAlive[next]) {
        const beyond = shift(w, next, d);
        if (beyond < 0 || !open[beyond] || boxesAlive[beyond]) return '';
        boxesAlive[next] = 0;
        boxesAlive[beyond] = 1;
      }
      player = next;
    }
    moves += inside;
  }
  return moves;
}

const ids = readdirSync(IN).sort();
for (const id of ids) {
  const level = readFileSync(join(IN, id, 'level.txt'), 'utf8');
  const answer = solve(level);
  mkdirSync(join(OUT, id), { recursive: true });
  writeFileSync(join(OUT, id, 'solution.txt'), `${answer}\n`);
}
