#!/bin/bash
#
# Generate a set of test graphs.
#

set -e

SRCDIR="$(dirname $0)"

make_random_graph() {
    python3 $SRCDIR/make_random_graph.py "$@"
}

make_slow_graph() {
    python3 $SRCDIR/make_slow_graph.py "$@"
}

triangles() {
    python3 $SRCDIR/triangles.py "$@"
}

make_random_graph --seed 101 1000 10000 > random_n1000_m10000.edge
make_random_graph --seed 102 2000 10000 > random_n2000_m10000.edge
make_random_graph --seed 103 4000 10000 > random_n4000_m10000.edge

make_slow_graph chain 1000 > chain_n1000.edge
make_slow_graph chain 5000 > chain_n5000.edge
make_slow_graph chain 10000 > chain_n10000.edge
make_slow_graph chain 20000 > chain_n20000.edge
make_slow_graph chain 50000 > chain_n50000.edge
make_slow_graph chain 100000 > chain_n100000.edge

make_slow_graph sparse 1004 > sparse_delta_n1004.edge
make_slow_graph sparse 2004 > sparse_delta_n2004.edge
make_slow_graph sparse 5004 > sparse_delta_n5004.edge
make_slow_graph sparse 10004 > sparse_delta_n10004.edge
make_slow_graph sparse 20004 > sparse_delta_n20004.edge
make_slow_graph sparse 40004 > sparse_delta_n40004.edge

triangles 334 1 > triangles_n1002.edge
triangles 1667 1 > triangles_n5001.edge
triangles 3334 1 > triangles_n10002.edge
triangles 16667 1 > triangles_n50001.edge
triangles 33334 1 > triangles_n100002.edge

