#!/usr/bin/env python3

"""
Generate a graph that is "difficult" for the O(n**3) matching algorithm.

Output in DIMACS format.
"""

from __future__ import annotations

import sys
import argparse
from typing import TextIO


count_make_blossom = [0]
count_delta_step = [0]


def patch_matching_code() -> None:
    """Patch the matching code to count events."""

    import max_weight_matching

    orig_make_blossom = max_weight_matching._MatchingContext.make_blossom
    orig_substage_calc_dual_delta = (
        max_weight_matching._MatchingContext.substage_calc_dual_delta)

    def stub_make_blossom(*args, **kwargs):
        count_make_blossom[0] += 1
        return orig_make_blossom(*args, **kwargs)

    def stub_substage_calc_dual_delta(*args, **kwargs):
        count_delta_step[0] += 1
        ret = orig_substage_calc_dual_delta(*args, **kwargs)
#        print("DELTA", ret)
        return ret

    max_weight_matching._MatchingContext.make_blossom = stub_make_blossom
    max_weight_matching._MatchingContext.substage_calc_dual_delta = (
        stub_substage_calc_dual_delta)


def run_max_weight_matching(
        edges: list[tuple[int, int, int]]
        ) -> tuple[list[tuple[int, int]], int, int]:
    """Run the matching algorithm and count subroutine calls."""
    import max_weight_matching

    count_make_blossom[0] = 0
    count_delta_step[0] = 0

    pairs = max_weight_matching.maximum_weight_matching(edges)
    return (pairs, count_make_blossom[0], count_delta_step[0])


def write_dimacs_graph(
        f: TextIO,
        edges: list[tuple[int, int, int]]
        ) -> None:
    """Write a graph in DIMACS edge list format."""

    num_vertex = 1 + max(max(x, y) for (x, y, _w) in edges)
    num_edge = len(edges)

    print(f"p edge {num_vertex} {num_edge}", file=f)

    for (x, y, w) in edges:
        print(f"e {x+1} {y+1} {w}", file=f)


def make_dense_slow_graph(n: int) -> list[tuple[int, int, int]]:
    """Generate a dense (not complete) graph with N vertices.

    N must be divisible by 4.

    Number of edges = M = (N**2/16 + N/2).
    Number of delta steps required to solve the matching = (M - 1).
    """

    assert n % 4 == 0

    edges: list[tuple[int, int, int]] = []

    num_peripheral_pairs = n // 4
    num_central_pairs = n // 4
    max_weight = 2 * num_central_pairs * (num_peripheral_pairs + 1)

    # Peripheral pairs will be matched up first.
    for i in range(num_peripheral_pairs):
        x = 2 * i
        y = x + 1
        w = max_weight - i
        edges.append((x, y, w))

    # Then for each central pair:
    for k in range(num_central_pairs):

        x = 2 * num_peripheral_pairs + 2 * k

        # Central pair discovers all peripheral pairs.
        for i in range(num_peripheral_pairs):
            y = 2 * i
            if k % 2 == 0:
                w = max_weight - (1 + k // 2) * (num_peripheral_pairs + 1) + 1
            else:
                w = max_weight - (1 + k // 2) * (num_peripheral_pairs + 1) - i
            edges.append((x, y, w))

        # Then this central pair gets matched.
        y = x + 1
        w = max_weight - (k + 1) * (2 * num_peripheral_pairs + 1)
        edges.append((x, y, w))

    return edges


def make_sparse_slow_graph(n: int) -> list[tuple[int, int, int]]:
    """Generate a sparse graph with N vertices.

    N must be 4 modulo 8.

    Number of edges = M = (5/4 * N - 3).
    Number of delta steps required to solve the matching ~ (N**2 / 16).
    """

    assert n >= 12
    assert n % 8 == 4

    num_p_pairs = (n - 4) // 4
    num_q_pairs = num_p_pairs // 2

    #
    # Graph structure:
    #
    #   O       O       O       O       O
    #   |       |       |       |       |     (P pairs of nodes)
    #   |       |       |       |       |     (one edge in each pair)
    #   O___    O       O       O    ___O
    #    \_ \   |\     / \    _/|   /  /
    #      \    |    _/    __/  |    _/       (2 * P edges)
    #       \_  |  _/   __/     |  _/         (connecting both coupling pairs
    #         \ | /  __/      \ | /            to each top-layer pair)
    #          \|/__/       \  \|/
    #           O------/  \--\--O
    #           |               |             (2 coupling pairs)
    #           |               |             (1 edge in each pair)
    #           O               O
    #          / \             / \            (2 * Q edges)
    #         /   \           /   \           (connecting each coupling pair
    #        /     \         /     \           to the Q pairs below it)
    #       O       O       O       O
    #       |       |       |       |         (2 groups of Q pairs of nodes)
    #       |       |       |       |         (1 edge in each pair)
    #       O       O       O       O
    #
    # Plan:
    #  - First, match each pair in the top layer.
    #  - Then, pull all edges between the top-layer and the
    #    left coupling pair tight (without matching anything).
    #  - Then match the left coupling pair.
    #  - Then pull all edges between the top-layer and the
    #    right coupling pair tight. (This will loosen the edges
    #    between the top-layer and the left coupling pair.)
    #  - Then match the right coupling pair.
    #  - Then alternate between the bottom left group and bottom right group
    #    of pairs:
    #     - Pick a new pair from the selected bottom group.
    #     - Pull the edge to its coupling pair tight.
    #     - Pull all edges between the coupling pair and the top-layer tight.
    #       (This will loosen the edges between the top-layer and the other
    #        coupling pair.)
    #     - Match the selected bottom pair.
    #
    # The trick is to assign edge weights that force the matching
    # algorithm to execute the plan as descibed above.
    #

    edges: list[tuple[int, int, int]] = []

    max_weight = 16 * (num_q_pairs + 1) * (num_p_pairs + 2)

    # Make the top pairs.
    for i in range(num_p_pairs):
        x = 2 * i
        y = 2 * i + 1
        w = max_weight - i
        edges.append((x, y, w))

    # Make the coupling pairs.
    for k in range(2):

        # Connect the coupling pair to the top layer.
        for i in range(num_p_pairs):
            x = 2 * i + 1
            y = 2 * num_p_pairs + 2 * k
            if k == 0:
                w = max_weight - num_p_pairs
            else:
                w = max_weight - num_p_pairs - 1 - i
            edges.append((x, y, w))

        # Make the internal edge in the coupling pair.
        x = 2 * num_p_pairs + 2 * k
        y = 2 * num_p_pairs + 2 * k + 1
        if k == 0:
            w = max_weight - 2 * num_p_pairs - 1
        else:
            w = max_weight - 4 * num_p_pairs - 2
        edges.append((x, y, w))

    # Make the bottom groups.
    for k in range(2):

        # Connect the coupling pair to the bottom layer.
        for i in range(num_q_pairs):
            x = 2 * num_p_pairs + 2 * k + 1
            y = 2 * num_p_pairs + 4 + 2 * num_q_pairs * k + 2 * i
            w = (max_weight
                 - (2 * i + k) * (2 * num_p_pairs + 4)
                 - 4 * num_p_pairs + 1)
            edges.append((x, y, w))

        # Make the pairs in this half of the bottom layer.
        for i in range(num_q_pairs):
            x = 2 * num_p_pairs + 4 + 2 * num_q_pairs * k + 2 * i
            y = 2 * num_p_pairs + 4 + 2 * num_q_pairs * k + 2 * i + 1
            w = (max_weight
                 - (2 * i + k + 1) * 8 * num_p_pairs
                 - (2 * i + k) * 12)
            edges.append((x, y, w))

    return edges


def main() -> int:
    """Main program."""

    parser = argparse.ArgumentParser()
    parser.description = "Generate a difficult graph."

    parser.add_argument("--structure",
                        action="store",
                        choices=("sparse", "dense"),
                        default="sparse",
                        help="choose graph structure")
    parser.add_argument("--check",
                        action="store_true",
                        help="solve the matching and count delta steps")
    parser.add_argument("n",
                        action="store",
                        type=int,
                        help="number of vertices")

    args = parser.parse_args()

    if args.check:
        patch_matching_code()

    if args.structure == "sparse":

        if args.n < 12:
            print("ERROR: Number of vertices must be >= 12", file=sys.stderr)
            return 1

        if args.n % 8 != 4:
            print("ERROR: Number of vertices must be 4 modulo 8",
                  file=sys.stderr)
            return 1

        edges = make_sparse_slow_graph(args.n)

    elif args.structure == "dense":

        if args.n < 4:
            print("ERROR: Number of vertices must be >= 4", file=sys.stderr)
            return 1

        if args.n % 4 != 0:
            print("ERROR: Number of vertices must be divisible by 4",
                  file=sys.stderr)
            return 1

        edges = make_dense_slow_graph(args.n)

    else:
        assert False

    if args.check:
        (pairs, num_blossom, num_delta) = run_max_weight_matching(edges)
        print(f"n={args.n} m={len(edges)} "
              f"nblossom={num_blossom} ndelta={num_delta}",
              file=sys.stderr)

    write_dimacs_graph(sys.stdout, edges)

    return 0


if __name__ == "__main__":
    sys.exit(main())