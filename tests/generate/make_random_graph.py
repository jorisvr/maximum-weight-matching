#!/usr/bin/env python3

"""
Generate a random graph in DIMACS format.
"""

from __future__ import annotations

import sys
import argparse
import random
from typing import TextIO


def write_dimacs_graph(
        f: TextIO,
        edges: list[tuple[int, int, int|float]]
        ) -> None:
    """Write a graph in DIMACS edge list format."""

    num_vertex = 1 + max(max(x, y) for (x, y, _w) in edges)
    num_edge = len(edges)

    print(f"p edge {num_vertex} {num_edge}", file=f)

    integer_weights = all(isinstance(w, int) for (_x, _y, w) in edges)

    for (x, y, w) in edges:
        if integer_weights:
            print(f"e {x+1} {y+1} {w}", file=f)
        else:
            print(f"e {x+1} {y+1} {w:.12g}", file=f)


def make_random_graph(
        n: int,
        m: int,
        max_weight: float,
        float_weights: bool,
        rng: random.Random
        ) -> list[tuple[int, int, int|float]]:
    """Generate a random graph with random edge weights."""

    edge_set: set[tuple[int, int]] = set()

    if 3 * m < n * (n - 2) // 2:
        # Simply add random edges until we have enough.
        while len(edge_set) < m:
            x = rng.randint(0, n - 2)
            y = rng.randint(x + 1, n - 1)
            edge_set.add((x, y))

    else:
        # We need a very dense graph.
        # Generate all edge candidates and choose a random subset.
        edge_candidates = [(x, y)
                           for x in range(n - 1)
                           for y in range(x + 1, n)]
        rng.shuffle(edge_candidates)
        edge_set.update(edge_candidates[:m])

    edges: list[tuple[int, int, int|float]] = []
    for (x, y) in sorted(edge_set):
        w: int|float
        if float_weights:
            w = rng.uniform(1.0e-8, max_weight)
        else:
            w = rng.randint(1, int(max_weight))
        edges.append((x, y, w))

    return edges


def main() -> int:
    """Main program."""

    parser = argparse.ArgumentParser()
    parser.description = "Generate a random graph in DIMACS format."

    parser.add_argument("--seed",
                        action="store",
                        type=int,
                        help="random seed")
    parser.add_argument("--maxweight",
                        action="store",
                        type=float,
                        default=1000000,
                        help="maximum edge weight")
    parser.add_argument("--float",
                        action="store_true",
                        help="use floating point edge weights")
    parser.add_argument("n",
                        action="store",
                        type=int,
                        help="number of vertices")
    parser.add_argument("m",
                        action="store",
                        type=int,
                        help="number of edges")

    args = parser.parse_args()

    if args.n < 2:
        print("ERROR: Number of vertices must be >= 2", file=sys.stderr)
        return 1

    if args.m < 1:
        print("ERROR: Number of edges must be >= 1", file=sys.stderr)
        return 1

    if args.m > args.n * (args.n - 1) // 2:
        print("ERROR: Too many edges", file=sys.stderr)
        return 1

    if args.maxweight < 1.0e-6:
        print("ERROR: Invalid maximum edge weight", file=sys.stderr)
        return 1

    if args.seed is None:
        rng = random.Random()
    else:
        rng = random.Random(args.seed)

    edges = make_random_graph(args.n, args.m, args.maxweight, args.float, rng)
    write_dimacs_graph(sys.stdout, edges)

    return 0


if __name__ == "__main__":
    sys.exit(main())
