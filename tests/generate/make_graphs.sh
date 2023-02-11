#!/bin/bash
#
# Generate a set of test graphs.
#

set -e

SRCDIR="$(dirname $0)"

# Usage: gen_random SEED N M
gen_random() {
    python3 $SRCDIR/make_random_graph.py --seed $1 $2 $3 \
      > graph_rnd_s${1}_n${2}_m${3}.gr
}

# Usage: gen_slow STRUCTURE N M
gen_slow() {
    python3 $SRCDIR/make_slow_graph.py --structure $1 $2 \
      > graph_slow_${1}_n${2}_m${3}.gr
}


gen_random 101 250 31125
gen_random 102 250 31125
gen_random 103 250  3952
gen_random 104 250  3952
gen_random 105 250  2500
gen_random 106 250  2500

gen_random 111 500 124750
gen_random 112 500 124750
gen_random 113 500  11180
gen_random 114 500  11180
gen_random 115 500   5000
gen_random 116 500   5000

gen_random 121 1000 499500
gen_random 122 1000 499500
gen_random 123 1000  31622
gen_random 124 1000  31622
gen_random 125 1000  10000
gen_random 126 1000  10000

gen_random 133 2000   89442
gen_random 134 2000   89442
gen_random 135 2000   20000
gen_random 136 2000   20000

gen_random 143 5000  353553
gen_random 144 5000  353553
gen_random 145 5000   50000
gen_random 146 5000   50000

gen_random 155 10000 100000
gen_random 156 10000 100000

gen_slow dense    252  4095
gen_slow dense    500 15875
gen_slow dense   1000 63000
gen_slow sparse   252   312
gen_slow sparse   500   622
gen_slow sparse  1004  1252
gen_slow sparse  2004  2502
gen_slow sparse  5004  6252
gen_slow sparse 10004 12502

