#!/usr/bin/env python3

"""
Generate a graph that belongs to a class of worst-case graphs
described by Gabow.

Reference: H. N. Gabow, "An efficient implementation of Edmonds'
           algorithm for maximum matching on graphs", JACM 23
           (1976), pp. 221-234.
 
Based on Fortran program "hardcard.f" by R. Bruce Mattingly, 1991.
Rewritten in Python by Joris van Rantwijk, 2023.

For the original Fortran code, see
  http://archive.dimacs.rutgers.edu/pub/netflow/generators/matching/hardcard.f

Output to stdout in DIMACS edge format.
All edges have weight 1.

Input parameter:    K
Number of vertices: N = 6*K
Number of edges:    M = 8*K*K

The graph is constructed so that vertices 1 - 4*K form a complete subgraph.
For 1 <= I <= 2*K, vertex (2*I-1) is joined to vertex (4*K+I).
"""

import sys
import argparse


def main():
    """Main program."""

    parser = argparse.ArgumentParser()
    parser.description = "Generate a difficult graph"

    parser.add_argument("k",
                        action="store",
                        type=int,
                        help="size parameter; N = 6*K, M = 4*K*K")
    args = parser.parse_args()

    if args.k < 1:
        print("ERROR: K must be at least 1", file=sys.stderr)
        sys.exit(1)

    k = args.k
    n = 6 * k
    m = 8 * k * k

    print(f"p edge {n} {m}")

    for i in range(1, 4*k):
        for j in range(i + 1, 4*k + 1):
            print(f"e {i} {j} 1")
        if i % 2 == 1:
            j = 4 * k + (i + 1) // 2
            print(f"e {i} {j} 1")


if __name__ == "__main__":
    main()
