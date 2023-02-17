#!/usr/bin/env python3

r"""
Generate a graph that consists of interconnected triangles.

The graph is a chain of triangles, connected either at 1 vertex
or at 3 vertices.

Example of triangles connected at one vertex:

   [1]-------[4]       [7]           [10]-------[13]
    | \       | \       | \         / |          |  \
    |  [3]    |  [6]    |  [9]---[12] |          |  [15]
    | /       | /       | /         \ |          |  /
   [2]       [5]-------[8]           [11]       [14]-----

Example of triangles connected at 3 vertices:

   [1]-------[4]-------[7]-------[10]-------[13]
    | \       | \       | \       |  \       |  \
    |  [3]----|--[6]----|--[9]----|--[12]----|--[15]
    | /       | /       | /       |  /       |  /
   [2]-------[5]-------[8]-------[11]-------[14]

Based on Fortran programs "t.f" and "tt.f"
by N. Ritchey and B. Mattingly, Youngstown State University, 1991.

Rewritten in Python by Joris van Rantwijk, 2023.

For the original Fortran code, see
  http://archive.dimacs.rutgers.edu/pub/netflow/generators/matching/t.f
  http://archive.dimacs.rutgers.edu/pub/netflow/generators/matching/tt.f

Output to stdout in DIMACS edge format.
All edges have weight 1.

Input parameter:    K = number of triangles
Input parameter:    C = 1 to connect triangles by 1 corner
                    C = 3 to connect triangles by 3 corners
Number of vertices: N = 3*K
Number of edges:    M = 3*K + C*(K-1)
"""

import argparse


def main():

    parser = argparse.ArgumentParser()
    parser.description = (
        "Generate a graph that consists of interconnected triangles.")

    parser.add_argument("k",
                        action="store",
                        type=int,
                        help="size parameter; N = 3*K, M = 3*K+C*(K-1)")
    parser.add_argument("c",
                        action="store",
                        type=int,
                        choices=(1, 3),
                        help="number of corners to connect")

    args = parser.parse_args()

    if args.k < 1:
        print("ERROR: K must be at least 1", file=sys.stderr)
        sys.exit(1)

    k = args.k
    n = 3 * k
    m = 3 * k + args.c * (k - 1)

    print(f"p edge {n} {m}")

    for i in range(k):
        x = 3 * i + 1
        print(f"e {x} {x+1} 1")
        print(f"e {x} {x+2} 1")
        print(f"e {x+1} {x+2} 1")

    if args.c == 1:
        for i in range(k - 1):
            x = 3 * i + i % 3 + 1
            print(f"e {x} {x+3} 1")

    elif args.c == 3:
        for x in range(1, 3 * k - 2):
            print(f"e {x} {x+3} 1")


if __name__ == "__main__":
    main()
