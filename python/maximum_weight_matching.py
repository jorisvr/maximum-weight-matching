"""
Algorithm for finding a maximum weight matching in general graphs.
"""

from __future__ import annotations

import sys
import math
from typing import cast, NamedTuple, Optional


def maximum_weight_matching(
        edges: list[tuple[int, int, int|float]]
        ) -> list[tuple[int, int]]:
    """Compute a maximum-weighted matching in the general undirected weighted
    graph given by "edges".

    The graph is specified as a list of edges, each edge specified as a tuple
    of its two vertices and the edge weight.
    There may be at most one edge between any pair of vertices.
    No vertex may have an edge to itself.
    The graph may be non-connected (i.e. contain multiple components).

    Vertices are indexed by consecutive, non-negative integers, such that
    the first vertex has index 0 and the last vertex has index (n-1).
    Edge weights may be integers or floating point numbers.

    Isolated vertices (not incident to any edge) are allowed, but not
    recommended since such vertices consume time and memory but have
    no effect on the maximum-weight matching.

    Negative edge weights are allowed, but not recommended since such edges
    consume time and memory but have no effect on the maximum-weight matching.

    This function takes time O(n**3), where "n" is the number of vertices.
    This function uses O(n + m) memory, where "m" is the number of edges.

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, wt)"
            where "i" and "j" are vertex indices and "wt" is the edge weight.

    Returns:
        List of pairs of matched vertex indices.
        This is a subset of the list of edges in the graph.
        It contains a tuple "(i, j)" if vertex "i" is matched to vertex "j".

    Raises:
        ValueError: If the input does not satisfy the constraints.
        TypeError: If the input contains invalid data types.
    """

    # Check that the input meets all constraints.
    _check_input_types(edges)
    _check_input_graph(edges)

    # Special case for empty graphs.
    if not edges:
        return []

    # Initialize graph representation.
    graph = _GraphInfo(edges)

    # Initialize trivial partial matching without any matched edges.
    matching = _PartialMatching(graph)

    # Improve the solution until no further improvement is possible.
    #
    # Each successful pass through this loop increases the number
    # of matched edges by 1.
    #
    # This loop runs through at most (n/2 + 1) iterations.
    # Each iteration takes time O(n**2).
    while _run_stage(matching):
        pass

    # Extract the final solution.
    pairs: list[tuple[int, int]] = [
        (i, j) for (i, j, _wt) in edges if matching.vertex_mate[i] == j]

    # Verify that the matching is optimal.
    # This only works reliably for integer weights.
    # Verification is a redundant step; if the matching algorithm is correct,
    # verification will always pass.
    if graph.integer_weights:
# TODO : Maybe interesting to redesign blossom/dual data structures such
#        that this info for verification is easier to extract.
        blossom_dual_var = [
            (2 * blossom.half_dual_var if blossom is not None else 0)
            for blossom in matching.blossom]
        _verify_optimum(graph,
                        pairs,
                        cast(list[int], matching.dual_var),
                        matching.blossom_parent,
                        cast(list[int], blossom_dual_var))

    return pairs


def adjust_weights_for_maximum_cardinality_matching(
        edges: list[tuple[int, int, int|float]]
        ) -> list[tuple[int, int, int|float]]:
    """Adjust edge weights such that the maximum-weight matching of
    the adjusted graph is a maximum-cardinality matching, equal to
    a matching in the original graph that has maximum weight out of all
    matchings with maximum cardinality.

    The graph is specified as a list of edges, each edge specified as a tuple
    of its two vertices and the edge weight.

    Vertices are indexed by consecutive, non-negative integers, such that
    the first vertex has index 0 and the last vertex has index (n-1).
    Edge weights may be integers or floating point numbers.
    Negative edge weights are allowed.

    This function increases all edge weights by an equal amount such that
    the adjusted weights satisfy the following conditions:
     - All edge weights are positive;
     - The minimum edge weight is at least "n" times the difference between
       maximum and minimum edge weight.

    These conditions ensure that a maximum-cardinality matching will be found:
    The weight of any non-maximum-cardinality matching can be increased by
    matching an additional edge, even if the new edge has minimum edge weight
    and causes all other matched edges to degrade from maximum to minimum
    edge weight.

    Since we are only considering maximum-cardinality matchings, increasing
    all edge weights by an equal amount will not change the set of edges
    that makes up the maximum-weight matching.

    This function increases edge weights by an amount that is proportional
    to the product of the unadjusted weight range and the number of vertices
    in the graph. In case of a big graph with floating point weights, this
    may introduce rounding errors in the weights.

    This function takes time O(m), where "m" is the number of edges.

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, wt)"
            where "i" and "j" are vertex indices and "wt" is the edge weight.

    Returns:
        List of edges with adjusted weights. If no adjustments are necessary,
        the input list instance may be returned.

    Raises:
        ValueError: If the input does not satisfy the constraints.
        TypeError: If the input contains invalid data types.
    """

    _check_input_types(edges)

    # Don't worry about empty graphs:
    if not edges:
        return edges

    num_vertex = 1 + max(max(i, j) for (i, j, _wt) in edges)

    min_weight = min(wt for (_i, _j, wt) in edges)
    max_weight = max(wt for (_i, _j, wt) in edges)
    weight_range = max_weight - min_weight

    # Do nothing if the weights already ensure a maximum-cardinality matching.
    if min_weight > 0 and min_weight >= num_vertex * weight_range:
        return edges

    delta: int|float
    if weight_range > 0:
        # Increase weights to make minimum edge weight large enough
        # to improve any non-maximum-cardinality matching.
        delta = num_vertex * weight_range - min_weight
    else:
        # All weights are the same. Increase weights to make them positive.
        delta = 1 - min_weight

    assert delta >= 0

    # Increase all edge weights by "delta".
    return [(i, j, wt + delta) for (i, j, wt) in edges]


def _check_input_types(edges: list[tuple[int, int, int|float]]) -> None:
    """Check that the input consists of valid data types and valid
    numerical ranges.

    This function takes time O(m).

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, wt)"
            where "i" and "j" are edge indices and "wt" is the edge weight.

    Raises:
        ValueError: If the input does not satisfy the constraints.
        TypeError: If the input contains invalid data types.
    """

    float_limit = sys.float_info.max / 4

    if not isinstance(edges, list):
        raise TypeError('"edges" must be a list')

    for e in edges:
        if (not isinstance(e, tuple)) or (len(e) != 3):
            raise TypeError("Each edge must be specified as a 3-tuple")

        (i, j, wt) = e

        if (not isinstance(i, int)) or (not isinstance(j, int)):
            raise TypeError("Edge endpoints must be integers")

        if (i < 0) or (j < 0):
            raise ValueError("Edge endpoints must be non-negative integers")

        if not isinstance(wt, (int, float)):
            raise TypeError(
                "Edge weights must be integers or floating point numbers")

        if isinstance(wt, float):
            if not math.isfinite(wt):
                raise ValueError("Edge weights must be finite numbers")

            # Check that this edge weight will not cause our dual variable
            # calculations to exceed the valid floating point range.
            if wt > float_limit:
                raise ValueError("Floating point edge weights must be"
                                 f" less than {float_limit:g}")


def _check_input_graph(edges: list[tuple[int, int, int|float]]) -> None:
    """Check that the input is a valid graph, without any multi-edges and
    without any self-edges.

    This function takes time O(m * log(m)).

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, wt)"
            where "i" and "j" are edge indices and "wt" is the edge weight.

    Raises:
        ValueError: If the input does not satisfy the constraints.
    """

    # Check that the graph has no self-edges.
    for (i, j, _wt) in edges:
        if i == j:
            raise ValueError("Self-edges are not supported")

    # Check that the graph does not have multi-edges.
    # Using a set() would be more straightforward, but the runtime bounds
    # of the Python set type are not clearly specified.
    # Sorting provides guaranteed O(m * log(m)) run time.
    edge_endpoints = [((i, j) if (i < j) else (j, i))
                      for (i, j, _wt) in edges]
    edge_endpoints.sort()

    for i in range(len(edge_endpoints) - 1):
        if edge_endpoints[i] == edge_endpoints[i+1]:
            raise ValueError(f"Duplicate edge {edge_endpoints[i]}")


class _GraphInfo:
    """Representation of the input graph.

    These data remain unchanged while the algorithm runs.
    """

    def __init__(self, edges: list[tuple[int, int, int|float]]) -> None:
        """Initialize the graph representation and prepare an adjacency list.

        This function takes time O(n + m).
        """

        # Vertices are indexed by integers in range 0 .. n-1.
        # Edges are indexed by integers in range 0 .. m-1.
        #
        # Each edge is incident on two vertices.
        # Each edge also has a weight.
        #
        # "edges[e] = (i, j, wt)" where
        #     "e" is an edge index;
        #     "i" and "j" are vertex indices of the incident vertices;
        #     "wt" is the edge weight.
        #
        # These data remain unchanged while the algorithm runs.
        self.edges: list[tuple[int, int, int|float]] = edges

        # num_vertex = the number of vertices.
        if edges:
            self.num_vertex = 1 + max(max(i, j) for (i, j, _wt) in edges)
        else:
            self.num_vertex = 0

        # Each vertex is incident to zero or more edges.
        #
        # "adjacent_edges[v]" is the list of edge indices of edges incident
        # to the vertex with index "v".
        #
        # These data remain unchanged while the algorithm runs.
        self.adjacent_edges: list[list[int]] = [
            [] for v in range(self.num_vertex)]
        for (e, (i, j, _wt)) in enumerate(edges):
            self.adjacent_edges[i].append(e)
            self.adjacent_edges[j].append(e)

        # Determine whether _all_ weights are integers.
        # In this case we can avoid floating point computations entirely.
        self.integer_weights: bool = all(isinstance(wt, int)
                                         for (_i, _j, wt) in edges)


# TODO:
#   When this fucking thing is finished and working, reconsider the data structures:
#    - Is it really so important to separate StageData from PartialMatching ?
#    - Especially since S-blossom least-slack tracking might be better placed inside _Blossom.
#    - Is there a way to shortcut the indexing from blossom index to _Blossom via blossom[b] ?
#    - Consider for example that the "blossombase" array in the old code was actually pretty nice
#      since it generalizes over trivial and non-trivial blossoms.
#    - Maybe we should EITHER go for ugly and array-heavy like the previous code,
#      OR nice and object-oriented, but then also make objects for single vertices.
#      Also try to think ahead to how this could be done in C++.
#    - Is there a way to reduce allocations of tuples ?
#      (First profile if it saves any time, otherwise don't even bother.)
#
#    - Probably better to use "vb" instead of "vblossom", so we consistently use the "b" suffix to mark blossom _index_.
#    - Consider using "wt" for weight instead of "w".
#    - Maybe nice to abstract away the management of least-slack edges to separate classes.
#
# Proof that S-to-S edges have even slack when working with integer weights:
# Edge slack is even iff indicent vertex duals are both odd or both even.
# Unmatched vertices are always S vertices, therefore either all unmatched vertices are odd or all unmatched vertices are even.
# Within an alternating tree, all edges have zero slack, therefore either all vertices in the tree are even or all are odd.
# Alternating trees are rooted in unmatched vertices, therefore either all vertices in all alternating trees are even or all are odd.
# Therefore all labeled vertices are even or all labeled vertices are odd.
# Therefore S-to-S edges have even slack.
#
# Proof that dual variables will always be in range (0 .. max_edge_weight):
#  - Assuming normal maximum-weight matching, without maximum-cardinality tricks.
#  - Assuming the original formulation of dual variables and edge weights;
#    i.e. in our implementation the range would be 0 .. 2 * max_edge_weight
#    due to implicit multiplication by 2.
#  - Initially, all dual variables are 0.5 * max_edge_weight
#  - While the algorithm runs, there is at least one unmatched vertex.
#  - Any unmatched vertex has been unmatched since the start of the algorithm
#    (since a matched vertex can never become unmatched).
#  - All unmatched vertices are immediately labeled as S-vertex.
#  - Therefore all delta updates have decreased the dual variables of
#    all unmatched edges.
#  - But the dual variable of an S-vertex can never become negative
#    (due to the delta1 rule).
#  - Therefore the sum of delta updates can not exceed 0.5 * max_edge_weight.
#  - Therefore no T-vertex can get its dual variable to exceed max_edge_weight.
#  - And no S-blossom can get its dual variable to exceed max_edge_weight.
#


class _Blossom:
    """Represents a non-trivial blossom in a (partially) matched graph.

    A blossom is an odd-length alternating cycle over sub-blossoms.
    An alternating path consists of alternating matched and unmatched edges.
    An alternating cycle is an alternating path that starts and ends in
    the same sub-blossom.

    A single vertex by itself is also a blossom: a "trivial blossom".
    We use the term "non-trivial blossom" to refer to a blossom that
    contains at least 3 sub-blossoms.

    Blossoms are recursive structures: A non-trivial blossoms contains
    sub-blossoms, which may themselves contain sub-blossoms etc.

    All vertices that are contained in a non-trivial blossom are matched.
    Exactly one of these vertices is matched to a vertex outside the blossom;
    this is the "base vertex" of the blossom.

    Blossoms are created and destroyed by the matching algorithm.
    This implies that not every odd-length alternating cycle is a blossom;
    it only becomes a blossom through an explicit action of the algorithm.
    An existing blossom may also be changed when the matching is augmented
    along a path that runs through the blossom.
    """

    def __init__(
            self,
            subblossoms: list[int],
            edges: list[tuple[int, int]],
            base_vertex: int
            ) -> None:
        """Initialize a new blossom."""

        # Sanity check.
        n = len(subblossoms)
        assert len(edges) == n
        assert n >= 3
        assert n % 2 == 1

        # "subblossoms" is a list of the sub-blossoms of the blossom,
        # ordered by their appearance in the alternating cycle.
        #
        # "subblossoms[0]" is the start and end of the alternating cycle.
        # "subblossoms[0]" contains the base vertex of the blossom.
        #
        # "subblossoms[i]" is blossom index (either a non-trivial blossom
        # index, or a vertex index if the i-th sub-blossom is trivial).
        self.subblossoms: list[int] = subblossoms

        # "edges" is a list of edges linking the sub-blossoms.
        # Each edge is represented as an ordered pair "(v, w)" where "v"
        # and "w" are vertex indices.
        #
        # "edges[0] = (v, w)" where vertex "v" in "subblossoms[0]" is
        # adjacent to vertex "w" in "subblossoms[1]", etc.
        self.edges: list[tuple[int, int]] = edges

        # "base_vertex" is the vertex index of the base of the blossom.
        # This is the unique vertex which is contained in the blossom
        # but currently matched to a vertex not contained in the blossom.
        self.base_vertex: int = base_vertex

        # Every blossom has a variable in the dual LPP.
        #
        # "half_dual_var" is half of the current value of the dual variable.
        # New blossoms start with dual variable 0.
        self.half_dual_var: int|float = 0


class _PartialMatching:
    """Represents a partial solution of the matching problem.

    These data change while the algorithm runs.
    """

    def __init__(self, graph: _GraphInfo) -> None:
        """Initialize a partial solution where all vertices are unmated."""

        # Keep a reference to the graph for convenience.
        self.graph = graph

        # Each vertex is either single (unmatched) or matched to
        # another vertex.
        #
        # If vertex "v" is matched to vertex "w",
        # "vertex_mate[v] == w" and "vertex_mate[w] == v".
        # If vertex "v" is unmatched, "vertex_mate[v] == -1".
        #
        # Initially all vertices are unmatched.
        self.vertex_mate: list[int] = graph.num_vertex * [-1]

        # Blossoms are indexed by integers in range 0 .. 2*n-1.
        #
        # Blossom indices in range 0 .. n-1 refer to the trivial blossoms
        # that consist of a single vertex. In this case the blossom index
        # is simply equal to the vertex index.
        #
        # Blossom indices in range n .. 2*n-1 refer to non-trivial blossoms,
        # represented by instances of the _Blossom class.
        #
        # "blossom[b]" (for n <= b < 2*n-1) is an instance of the _Blossom
        # class that describes the non-trivial blossom with index "b".
        #
        # Blossoms are created and destroyed while the algorithm runs.
        # Initially there are no blossoms.
        self.blossom: list[Optional[_Blossom]] = [
            None for b in range(2 * graph.num_vertex)]

        # List of currently unused blossom indices.
        self.unused_blossoms: list[int] = list(
            range(graph.num_vertex, 2 * graph.num_vertex))

        # TODO : we may need a list of top-level blossom indices

        # Every vertex is part of exactly one top-level blossom,
        # possibly a trivial blossom consisting of just that vertex.
        #
        # "vertex_blossom[v]" is the index of the top-level blossom that
        # contains vertex "v".
        # "vertex_blossom[v] == v" if the "v" is a trivial top-level blossom.
        #
        # Initially all vertices are top-level trivial blossoms.
        self.vertex_blossom: list[int] = list(range(graph.num_vertex))

        # "blossom_parent[b]" is the index of the smallest blossom that
        # contains blossom "b", or
        # "blossom_parent[b] == -1" if blossom "b" is a top-level blossom.
        #
        # Initially all vertices are trivial top-level blossoms.
        self.blossom_parent: list[int] = (2 * graph.num_vertex) * [-1]

        # Every vertex has a variable in the dual LPP.
        #
        # "dual_var[v]" is the current value of the dual variable of "v".
        #
        # Vertex duals are initialized to half the maximum edge weight.
        # Note that we multiply all edge weights by 2, and half of 2 times
        # the maximum edge weight is simply the maximum edge weight.
        max_weight = max(wt for (_i, _j, wt) in graph.edges)
        self.dual_var: list[int|float] = graph.num_vertex * [max_weight]

    def edge_slack(self, e: int) -> int|float:
        """Return the slack of the edge with index "e".

        The result is only valid for edges that are not between vertices
        that belong to the same top-level blossom.

        Slack values are integers if all edge weights are even integers.
        For this reason, we multiply all edge weights by 2.
        """
        (i, j, wt) = self.graph.edges[e]
        assert self.vertex_blossom[i] != self.vertex_blossom[j]
        return self.dual_var[i] + self.dual_var[j] - 2 * wt

    def get_blossom(self, b: int) -> _Blossom:
        """Return the Blossom instance for blossom index "b"."""
        blossom = self.blossom[b]
        assert blossom is not None
        return blossom

    def blossom_vertices(self, b: int) -> list[int]:
        """Return a list of vertex indices contained in blossom "b"."""
        num_vertex = self.graph.num_vertex
        if b < num_vertex:
            return [b]
        else:
            # Use an explicit stack to avoid deep recursion.
            stack: list[int] = [b]
            nodes: list[int] = []
            while stack:
                b = stack.pop()
                for sub in self.get_blossom(b).subblossoms:
                    if sub < num_vertex:
                        nodes.append(sub)
                    else:
                        stack.append(sub)
        return nodes


# Each vertex may be labeled "S" (outer) or "T" (inner) or be unlabeled.
_LABEL_NONE = 0
_LABEL_S = 1
_LABEL_T = 2


class _StageData:
    """Data structures that are used during a stage of the algorithm."""

    def __init__(self, graph: _GraphInfo) -> None:
        """Initialize data structures for a new stage."""

        num_vertex = graph.num_vertex

        # Top-level blossoms that are part of an alternating tree are
        # labeled S or T. Unlabeled top-level blossoms are not (yet)
        # part of any alternating tree.
        #
        # "blossom_label[b]" is the label of blossom "b".
        #
        # At the beginning of a stage, all blossoms are unlabeled.
        self.blossom_label: list[int] = (2 * num_vertex) * [_LABEL_NONE]

        # For each labeled blossom, we keep track of the edge that attaches
        # it to its alternating tree.
        #
        # "blossom_link[b] = (v, w)" denotes the edge through which
        # blossom "b" is attached to the alternating tree, where "v" and "w"
        # are vertex indices and vertex "w" is contained in blossom "b".
        #
        # "blossom_link[b] = None" if "b" is the root of an alternating tree,
        # or if "b" is not a labeled, top-level blossom.
        self.blossom_link: list[Optional[tuple[int, int]]] = [
            None for b in range(2 * num_vertex)]

        # "vertex_best_edge[v]" is the edge index of the least-slack edge
        # between "v" and any S-vertex, or -1 if no such edge has been found.
        self.vertex_best_edge: list[int] = num_vertex * [-1]

        # For non-trivial top-level S-blossom "b",
        # "blossom_best_edge_set[b]" is a list of edges between blossom "b"
        # and other S-blossoms.
        self.blossom_best_edge_set: list[Optional[list[int]]] = [
            None for b in range(2 * num_vertex)]

        # For every top-level S-blossom "b",
        # "blossom_best_edge[b]" is the edge index of the least-slack edge
        # to a different S-blossom, or -1 if no such edge has been found.
        self.blossom_best_edge: list[int] = (2 * num_vertex) * [-1]

        # A list of S-vertices to be scanned.
        # We call it a queue, but it is actually a stack.
        self.queue: list[int] = []

        # Temporary array used to construct new blossoms.
        self.blossom_marker: list[bool] = (2 * num_vertex) * [False]


class _AlternatingPath(NamedTuple):
    """Represents an alternating path or an alternating cycle."""
    edges: list[tuple[int, int]]


def _trace_alternating_paths(
        matching: _PartialMatching,
        stage_data: _StageData,
        v: int,
        w: int
        ) -> _AlternatingPath:
    """Trace back through the alternating trees from vertices "v" and "w".

    If both vertices are part of the same alternating tree, this function
    discovers a new blossom. In this case it returns an alternating path
    through the blossom that starts and ends in the same sub-blossom.

    If the vertices are part of different alternating trees, this function
    discovers an augmenting path. In this case it returns an alternating
    path that starts and ends in an unmatched vertex.

    This function takes time O(k) to discover a blossom, where "k" is
    the number of sub-blossoms, or time O(n) to discover an augmenting path.

    Returns:
        Alternating path as an ordered list of edges between top-level
        blossoms.
    """

    vertex_blossom = matching.vertex_blossom
    blossom_link = stage_data.blossom_link
    blossom_marker = stage_data.blossom_marker
    marked_blossoms: list[int] = []

    # "vedges" is a list of edges used while tracing from "v".
    # "wedges" is a list of edges used while tracing from "w".
    vedges: list[tuple[int, int]] = []
    wedges: list[tuple[int, int]] = []

    # Pre-load the edge between "v" and "w" so it will end up in the right
    # place in the final path.
    vedges.append((v, w))

    # Alternate between tracing the path from "v" and the path from "w".
    # This ensures that the search time is bounded by the size of the
    # newly found blossom.
    first_common = -1
    while v != -1 or w != -1:

        # Check if we found a common ancestor.
        vblossom = vertex_blossom[v]
        if blossom_marker[vblossom]:
            first_common = vblossom
            break

        # Mark blossom as a potential common ancestor.
        blossom_marker[vblossom] = True
        marked_blossoms.append(vblossom)

        # Track back through the link in the alternating tree.
        link = blossom_link[vblossom]
        if link is None:
            # Reached the root of this alternating tree.
            v = -1
        else:
            vedges.append(link)
            v = link[0]

        # Swap "v" and "w" to alternate between paths.
        if w != -1:
            v, w = w, v
            vedges, wedges = wedges, vedges

    # Remove all markers we placed.
    for b in marked_blossoms:
        blossom_marker[b] = False

    # If we found a common ancestor, trim the paths so they end there.
    if first_common == -1:
        assert vertex_blossom[vedges[-1][0]] == first_common
        while wedges and (vertex_blossom[wedges[-1][0]] != first_common):
            wedges.pop()

    # Fuse the two paths.
    path_edges = vedges[::-1]
    path_edges.extend((q, p) for (p, q) in wedges)

    # Any S-to-S alternating path must have odd length.
    assert len(path_edges) % 2 == 1

    return _AlternatingPath(path_edges)


def _make_blossom(
        matching: _PartialMatching,
        stage_data: _StageData,
        path: _AlternatingPath
        ) -> None:
    """Create a new blossom from an alternating cycle.

    Assign label S to the new blossom.
    Relabel all T-sub-blossoms as S and add their vertices to the queue.

    This function takes time O(n).
    """

    num_vertex = matching.graph.num_vertex

    # Check that the path is odd-length.
    assert len(path.edges) % 2 == 1
    assert len(path.edges) >= 3

    # Construct the list of sub-blossoms (current top-level blossoms).
    subblossoms = [matching.vertex_blossom[v] for (v, w) in path.edges]

    # Check that the path is cyclic.
    # Note the path may not start and end with the same _vertex_,
    # but it must start and end in the same _blossom_.
    subblossoms_next = [matching.vertex_blossom[w] for (v, w) in path.edges]
    assert subblossoms[0] == subblossoms_next[-1]
    assert subblossoms[1:] == subblossoms[:-1]

    # Determine the base vertex of the new blossom.
    base_blossom = subblossoms[0]
    if base_blossom >= num_vertex:
        base_vertex = matching.get_blossom(base_blossom).base_vertex
    else:
        base_vertex = base_blossom

    # Create the new blossom object.
    blossom = _Blossom(subblossoms, path.edges, base_vertex)

    # Allocate a new blossom index and create the blossom object.
    b = matching.unused_blossoms.pop()
    matching.blossom[b] = blossom
    matching.blossom_parent[b] = -1

    # Link the subblossoms to the their new parent.
    for sub in subblossoms:
        matching.blossom_parent[sub] = b

    # Update blossom-membership of all vertices in the new blossom.
    # NOTE: This step takes O(n) time per blossom formation, and adds up
    #       to O(n**2) total time per stage.
    #       This could be improved through a union-find datastructure, or
    #       by somehow re-using the blossom index of the largest sub-blossom.
    for v in matching.blossom_vertices(b):
        matching.vertex_blossom[v] = b

    # Assign label S to the new blossom.
    assert stage_data.blossom_label[base_blossom] == _LABEL_S
    stage_data.blossom_label[b] = _LABEL_S
    stage_data.blossom_link[b] = stage_data.blossom_link[base_blossom]

    # Former T-vertices which are part of this blossom have now become
    # S-vertices. Add them to the queue.
    for sub in subblossoms[1::2]:
        if sub < num_vertex:
            stage_data.queue.append(sub)
        else:
            stage_data.queue.extend(matching.blossom_vertices(sub))

    # Calculate the set of least-slack edges to other S-blossoms.
    # We basically merge the edge lists from all sub-blossoms, but reject
    # edges that are internal to this blossom, and trim the set such that
    # there is at most one edge to each external S-blossom.
    #
    # Sub-blossoms that were formerly labeled T can be ignored, since their
    # vertices are in the queue and will discover neighbouring S-blossoms
    # via the edge scan process.
    #
    # Build a temporary array holding the least-slack edge index to
    # each top-level S-blossom.
    #
    # NOTE: This step takes O(n) time per blossom formation, and adds up
    #       to O(n**2) total time per stage.
    #       For sparse graphs, this could be improved by tracking
    #       least-slack edges in a priority queue.
    best_edge_to_blossom: list[int] = (2 * num_vertex) * [-1]
    best_slack_to_blossom: list[int|float] = [
        0 for b in range(2 * num_vertex)]

    # Add the least-slack edges of every S-sub-blossom.
    for sub in subblossoms[0::2]:
        if sub < num_vertex:
            # Trivial blossoms don't have a list of least-slack edges,
            # so we just look at all adjacent edges. This happens at most
            # once per vertex per stage. It adds up to O(m) time per stage.
            sub_edge_set = matching.graph.adjacent_edges[sub]
        else:
            # Use the edge list of the sub-blossom, then delete it from
            # the sub-blossom.
            sub_edge_set_opt = stage_data.blossom_best_edge_set[sub]
            assert sub_edge_set_opt is not None
            sub_edge_set = sub_edge_set_opt
            stage_data.blossom_best_edge_set[sub] = None

        # Add edges to the temporary array.
        for e in sub_edge_set:
            (i, j, _wt) = matching.graph.edges[e]
            iblossom = matching.vertex_blossom[i]
            jblossom = matching.vertex_blossom[j]
            assert (iblossom == b) or (jblossom == b)
            vblossom = jblossom if iblossom == b else iblossom

            # Reject internal edges in this blossom.
            if vblossom == b:
                continue

            # Reject edges that don't link to an S-blossom.
            if stage_data.blossom_label[vblossom] != _LABEL_S:
                continue

            # Keep only the least-slack edge to "vblossom".
            slack = matching.edge_slack(e)
            if ((best_edge_to_blossom[vblossom] == -1)
                    or (slack < best_slack_to_blossom[vblossom])):
                best_edge_to_blossom[vblossom] = e
                best_slack_to_blossom[vblossom] = slack

    # Extract a compact list of least-slack edge indices.
    # We can not keep the temporary array because that would blow up
    # memory use to O(n**2).
    best_edge_set = [e for e in best_edge_to_blossom if e != -1]
    stage_data.blossom_best_edge_set[b] = best_edge_set

    # Select the overall least-slack edge to any other S-blossom.
    best_edge = -1
    best_slack: int|float = 0
    for e in best_edge_set:
        slack = matching.edge_slack(e)
        if (best_edge == -1) or (slack < best_slack):
            best_edge = e
            best_slack = slack
    stage_data.blossom_best_edge[b] = best_edge


def _stage_mark_unmatched_vertices(
        matching: _PartialMatching,
        stage_data: _StageData
        ) -> None:
    """Assign label S to all unmatched vertices and put them in the
    scan queue.

    This function takes time O(n).
    """
    num_vertex = matching.graph.num_vertex

    for v in range(num_vertex):
        if matching.vertex_mate[v] < 0:

            # "v" is an unmatched vertex.
            # Double-check that it is not contained in a non-trivial blossom.
            assert matching.vertex_blossom[v] == v

            # Assign label S and mark as root of an alternating tree.
            stage_data.blossom_label[v] = _LABEL_S
            stage_data.blossom_link[v] = None

            # Add to the scan queue.
            stage_data.queue.append(v)


def _substage_add_unlabeled(
        matching: _PartialMatching,
        stage_data: _StageData,
        v: int,
        w: int
        ) -> None:
    """Add the unlabeled blossom that contains vertex "w".

    Assign label T to the blossom that contains "w" and assign label S
    to its mate. Attach both newly labeled blossoms to the alternating tree.

    Any vertices that become S-vertices through this process are added to
    the queue.

    Preconditions:
        "v" is an S-vertex.
        "w" is an unlabeled, matched vertex.
        There is a tight edge between vertices "v" and "w".
    """
    assert stage_data.blossom_label[matching.vertex_blossom[v]] == _LABEL_S

    # Assign label T to the unlabeled blossom.
    wblossom = matching.vertex_blossom[w]
    assert stage_data.blossom_label[wblossom] == _LABEL_NONE
    stage_data.blossom_label[wblossom] = _LABEL_T
    stage_data.blossom_link[wblossom] = (v, w)

    # Find the mate "y" of vertex "w".
    wbase = w if wblossom == w else matching.get_blossom(wblossom).base_vertex
    y = matching.vertex_mate[wbase]

    # Assign label S to the blossom that contains vertex "y".
    yblossom = matching.vertex_blossom[y]
    assert stage_data.blossom_label[yblossom] == _LABEL_NONE
    stage_data.blossom_label[yblossom] = _LABEL_S
    stage_data.blossom_link[yblossom] = (w, y)

    # Add all vertices inside the newly labeled S-blossom to the queue.
    if yblossom == y:
        stage_data.queue.append(y)
    else:
        stage_data.queue.extend(matching.blossom_vertices(yblossom))


def _substage_add_s_to_s_edge(
        matching: _PartialMatching,
        stage_data: _StageData,
        v: int,
        w: int
        ) -> Optional[_AlternatingPath]:
    """Add the edge between S-vertices "v" and "w".

    If the edge connects blossoms that are part of the same alternating tree,
    this function creates a new S-blossom and returns None.

    If the edge connects two different alternating trees, an augmenting
    path has been discovered. In this case the function changes nothing
    and returns the augmenting path.

    Returns:
        Augmenting path if found; otherwise None.
    """

    # Trace back through the alternating trees from "v" and "w".
    path = _trace_alternating_paths(matching, stage_data, v, w)

    # If the path is a cycle, we create a new blossom.
    # Otherwise the path is an augmenting path.
    # Note that an alternating starts and ends in the same blossom,
    # but not necessarily in the same vertex within that blossom.
    p = path.edges[0][0]
    q = path.edges[-1][1]
    if matching.vertex_blossom[p] == matching.vertex_blossom[q]:
        _make_blossom(matching, stage_data, path)
        return None
    else:
        return path


def _substage_scan(
        matching: _PartialMatching,
        stage_data: _StageData
        ) -> Optional[_AlternatingPath]:
    """Scan to expand the alternating trees.

    The scan proceeds until either an augmenting path is found,
    or the queue of S vertices becomes empty.

    During the scan, new blossoms may be discovered and added to
    the partial matching. Any such newly discovered blossoms are S-blossoms.

    Returns:
        Augmenting path if found; otherwise None.
    """

    edges = matching.graph.edges
    adjacent_edges = matching.graph.adjacent_edges

    # Process S-vertices waiting to be scanned.
    while stage_data.queue:

        # Take a vertex from the queue.
        v = stage_data.queue.pop()
        vblossom = matching.vertex_blossom[v]

        # Double-check that "v" is an S-vertex.
        assert stage_data.blossom_label[vblossom] == _LABEL_S

        # Scan the edges that are incident on "v".
        for e in adjacent_edges[v]:
            (i, j, _wt) = edges[e]
            w = i if i != v else j  # TODO : consider abstracting this

            # Consider the edge between vertices "v" and "w".
            # Try to pull this edge into an alternating tree.

            # Ignore edges that are internal to a blossom.
            wblossom = matching.vertex_blossom[w]
            if vblossom == wblossom:
                continue

            wlabel = stage_data.blossom_label[wblossom]

            # Check whether this edge is tight (has zero slack).
            # Only tight edges may be part of an alternating tree.
            slack = matching.edge_slack(e)
            if slack <= 0:
                if wlabel == _LABEL_NONE:
                    # Add the unlabeled blossom containing vertex "w" to the
                    # alternating tree and label it as T-blossom.
                    _substage_add_unlabeled(matching, stage_data, v, w)
                elif wlabel == _LABEL_S:
                    # This edge connects two S-blossoms. Use it to find
                    # either a new blossom or an augmenting path.
                    alternating_path = _substage_add_s_to_s_edge(matching,
                                                                 stage_data,
                                                                 v, w)
                    if alternating_path is not None:
                        return alternating_path

            elif wlabel == _LABEL_S:
                # Update tracking of least-slack edges between S-blossoms.
                best_edge = stage_data.blossom_best_edge[vblossom]
                if best_edge < 0 or slack < matching.edge_slack(best_edge):
                    stage_data.blossom_best_edge[vblossom] = e

                # Update the list of least-slack edges to S-blossoms for
                # the blossom that contains "v".
                # We only do this for non-trivial blossoms.
                if vblossom != v:
                    best_edge_set = stage_data.blossom_best_edge_set[vblossom]
                    assert best_edge_set is not None
                    best_edge_set.append(e)

            # Update tracking of least-slack edges between vertex "w" and
            # any S-vertex. We do this even for edges that have zero slack.
            # (If "w" is part of a T-blossom, it may become unlabeled later.
            # At that point we will need this edge to relabel vertex "w".
            # And we will find this edge through least-slack edge tracking.)
            best_edge = stage_data.vertex_best_edge[w]
            if best_edge < 0 or slack < matching.edge_slack(best_edge):
                stage_data.vertex_best_edge[w] = e

    # No further S vertices to scan, and no augmenting path found.
    return None


def _find_path_through_blossom(
        matching: _PartialMatching,
        b: int,
        s: int
        ) -> tuple[list[int], list[tuple[int, int]]]:
    """Construct a path through blossom "b" from sub-blossom "s"
    to the base of the blossom.

    Return:
        Tuple (nodes, edges).
    """
    blossom = matching.blossom[b]
    assert blossom is not None

    nodes: list[int] = [s]
    edges: list[tuple[int, int]] = []

    # Walk around the blossom from "s" to its base.
    i = blossom.subblossoms.index(s)
    nsub = len(blossom.subblossoms)
    while i != 0:
        if i % 2 == 0:
            # Stepping towards the beginning of the subblossom list.
            # Currently at subblossom (i), next position (i-2):
            #
            #  0 --- 1 === 2 --- 3 === (i-2) --- (i-1) ==(v,w)== (i)
            #                           ^^^                      ^^^
            #                               <-------------------
            #
            # We must flip edges from (v,w) to (w,v) to make them fit
            # in the path from "s" to base.
            edges.append(blossom.edges[i-1][::-1])
            nodes.append(blossom.subblossoms[i-1])
            edges.append(blossom.edges[i-2][::-1])
            nodes.append(blossom.subblossoms[i-2])
            i -= 2
        else:
            # Stepping towards the end of the subblossom list.
            # Currently at subblossom (i), next position (i+2):
            #
            #  (i) ==(v,w)== (i+1) --- (i+2) === (i+3) --- 0
            #  ^^^                      ^^^
            #      ------------------->
            edges.append(blossom.edges[i])
            nodes.append(blossom.subblossoms[i+1])
            edges.append(blossom.edges[i+1])
            nodes.append(blossom.subblossoms[(i+2) % nsub])

    return (nodes, edges)


def _augment_blossom(matching: _PartialMatching, b: int, v: int) -> None:
    """Augment along an alternating path through blossom "b", from vertex "v"
    to the base vertex of the blossom.

    This function takes time O(n).
    """

    num_vertex = matching.graph.num_vertex

    # Use an explicit stack to avoid deep recursion.
    stack = [(b, v)]

    while stack:
        (top_blossom, sub) = stack.pop()
        b = matching.blossom_parent[sub]

        if b != top_blossom:
            # Set up to continue augmenting through the parent of "b".
            stack.append((top_blossom, b))

        # Augment blossom "b" from subblossom "sub" to the base of the
        # blossom. Afterwards, "sub" contains the new base vertex.

        blossom = matching.blossom[b]
        assert blossom is not None

        # Walk through the expanded blossom from "sub" to the base vertex.
        (path_nodes, path_edges) = _find_path_through_blossom(
            matching, b, sub)

        for i in range(0, len(path_edges), 2):
            # Before augmentation:
            #   path_nodes[i] is matched to path_nodes[i+1]
            #
            #   (i) ===== (i+1) ---(v,w)--- (i+2)
            #
            # After augmentation:
            #   path_nodes[i+1] is mathed to path_nodes[i+2] via edge (v, w)
            #
            #   (i) ----- (i+1) ===(v,w)=== (i+2)
            #

            # Pull the edge (v, w) into the matching.
            (v, w) = path_edges[i+1]
            matching.vertex_mate[v] = w
            matching.vertex_mate[w] = v

            # Augment through the subblossoms touching the edge (v, w).
            # Nothing needs to be done for trivial subblossoms.
            vb = path_nodes[i+1]
            if vb >= num_vertex:
                stack.append((vb, v))

            wb = path_nodes[i+2]
            if wb >= num_vertex:
                stack.append((wb, w))

        # Rotate the subblossom list so the new base ends up in position 0.
        sub_pos = blossom.subblossoms.index(sub)
        blossom.subblossoms = (blossom.subblossoms[sub_pos:]
                               + blossom.subblossoms[:sub_pos])
        blossom.edges = blossom.edges[sub_pos:] + blossom.edges[:sub_pos]

        # Update the base vertex.
        # We can pull this from the sub-blossom where we started since
        # its augmentation has already finished.
        if sub < num_vertex:
            blossom.base_vertex = sub
        else:
            blossom.base_vertex = matching.get_blossom(sub).base_vertex


def _expand_t_blossom(
        matching: _PartialMatching,
        stage_data: _StageData,
        b: int
        ) -> None:
    """Expand the specified T-blossom.

    This function takes time O(n).
    """

    num_vertex = matching.graph.num_vertex

    blossom = matching.blossom[b]
    assert blossom is not None
    assert matching.blossom_parent[b] == -1
    assert stage_data.blossom_label[b] == _LABEL_T

    # Convert sub-blossoms into top-level blossoms.
    for sub in blossom.subblossoms:
        matching.blossom_parent[sub] = -1
        if sub < num_vertex:
            matching.vertex_blossom[sub] = sub
        else:
            for v in matching.blossom_vertices(sub):
                matching.vertex_blossom[v] = sub
        assert stage_data.blossom_label[sub] == _LABEL_NONE

    # The expanding blossom was part of an alternating tree, linked to
    # a parent node in the tree via one of its subblossoms, and linked to
    # a child node of the tree via the base vertex.
    # We must reconstruct this part of the alternating tree, which will
    # now run via sub-blossoms of the expanded blossom.

    # Determine which sub-blossom is attached to the parent tree node.
# TODO : uglyness with the assertion
    entry_link = stage_data.blossom_link[b]
    assert entry_link is not None
    (v, w) = entry_link
    sub = matching.vertex_blossom[w]

    # Assign label T to that sub-blossom.
    stage_data.blossom_label[sub] = _LABEL_T
    stage_data.blossom_link[sub] = (v, w)

    # Walk through the expanded blossom from "sub" to the base vertex.
    # Assign alternating S and T labels to the sub-blossoms and attach
    # them to the alternating tree.
    (path_nodes, path_edges) = _find_path_through_blossom(matching, b, sub)

    for i in range(0, len(path_edges), 2):
        #
        #   (i) ===== (i+1) ----- (i+2)
        #    T          S           T
        #
        # path_nodes[i] has already been labeled T.
        # We now assign labels to path_nodes[i+1] and path_nodes[i+2].

        # Assign label S to path_nodes[i+1] and attach it to path_nodes[i].
        sub = path_nodes[i+1]
        stage_data.blossom_label[sub] = _LABEL_S
        stage_data.blossom_link[sub] = path_edges[i]

        # Put vertices in the newly labeled S-blossom in the queue.
# TODO : It feels like we have seen this code pattern a few times; try to generalize it.
        if sub < num_vertex:
            stage_data.queue.append(sub)
        else:
            stage_data.queue.extend(matching.blossom_vertices(sub))

        # Assign label T to path_nodes[i+2] and attach it to path_nodes[i+1].
        sub = path_nodes[i+2]
        stage_data.blossom_label[sub] = _LABEL_T
        stage_data.blossom_link[sub] = path_edges[i+1]

    # Unlabel and delete the expanded blossom. Recycle its blossom index.
    stage_data.blossom_label[b] = _LABEL_NONE
    stage_data.blossom_link[b] = None
    matching.blossom[b] = None
    matching.unused_blossoms.append(b)


def _expand_zero_dual_blossoms(matching: _PartialMatching) -> None:
    """Expand all blossoms with zero dual variable (recursively).

    This function takes time O(n).
    """

    num_vertex = matching.graph.num_vertex

    # Find top-level blossoms with zero slack.
    stack: list[int] = []
    for b in range(num_vertex, 2 * num_vertex):
        blossom = matching.blossom[b]
        if (blossom is not None) and (matching.blossom_parent[b] == -1):
            # We basically expand only S-blossoms that were created after
            # the most recent delta step. Those blossoms have dual variable
            # _exactly_ zero. So this comparison is reliable, even in case
            # of floating point edge weights.
            if blossom.half_dual_var == 0:
                stack.append(b)

    # Use an explicit stack to avoid deep recursion.
    while stack:
        b = stack.pop()

        # Expand blossom "b".

        blossom = matching.blossom[b]
        assert blossom is not None
        assert matching.blossom_parent[b] == -1

        # Examine sub-blossoms of "b".
        for sub in blossom.subblossoms:

            # Mark the sub-blossom as a top-level blossom.
            matching.blossom_parent[sub] = -1

            if sub < num_vertex:
                # Trivial sub-blossom. Mark it as top-level vertex.
                matching.vertex_blossom[sub] = sub
            else:
                # Non-trivial sub-blossom.
                # If its dual variable is zero, we expand it recursively.
                if matching.get_blossom(sub).half_dual_var == 0:
                    stack.append(sub)
                else:
                    # This sub-blossom will not be expanded.
                    # We still need to update its "vertex_blossom" entries.
                    for v in matching.blossom_vertices(sub):
                        matching.vertex_blossom[v] = sub

        # Delete the expanded blossom. Recycle its blossom index.
        matching.blossom[b] = None
        matching.unused_blossoms.append(b)


def _augment_matching(
        matching: _PartialMatching,
        path: _AlternatingPath
        ) -> None:
    """Augment the matching through the specified augmenting path.

    This function takes time O(n).
    """

    # Check that the augmenting path starts and ends in an unmatched vertex.
    assert len(path.edges) % 2 == 1
    assert matching.vertex_mate[path.edges[0][0]] == -1
    assert matching.vertex_mate[path.edges[-1][1]] == -1

    # Consider the edges of the augmenting path that are currently not
    # part of the matching but will become part of it.
    for (v, w) in path.edges[0::2]:

        # Augment through non-trivial blossoms on either side of this edge.
        vblossom = matching.vertex_blossom[v]
        if vblossom != v:
            _augment_blossom(matching, vblossom, v)

        wblossom = matching.vertex_blossom[w]
        if wblossom != w:
            _augment_blossom(matching, wblossom, w)

        # Pull the edge into the matching.
        matching.vertex_mate[v] = w
        matching.vertex_mate[w] = v


def _calc_dual_delta(
        matching: _PartialMatching,
        stage_data: _StageData
        ) -> tuple[int, float|int, int, int]:
    """Calculate a delta step in the dual LPP problem.

    This function returns the minimum of the 4 types of delta values,
    and the type of delta which obtain the minimum, and the edge or blossom
    that produces the minimum delta, if applicable.

    The returned delta value is an integer if all edge weights are even
    integers.

    This function assumes that there is at least one S-vertex.
    This function takes time O(n).

    Returns:
        Tuple (delta_type, delta, delta_edge, delta_blossom).
    """
    num_vertex = matching.graph.num_vertex

    delta_edge = -1
    delta_blossom = -1

    # Compute delta1: minimum dual variable of any S-vertex.
    delta_type = 1
    delta = min(matching.dual_var[v]
                for v in range(num_vertex)
                if stage_data.blossom_label[matching.vertex_blossom[v]])

    # Compute delta2: minimum slack of any edge between an S-vertex and
    # an unlabeled vertex.
    for v in range(num_vertex):
        vb = matching.vertex_blossom[v]
        if stage_data.blossom_label[vb] == _LABEL_NONE:
            e = stage_data.vertex_best_edge[v]
            if e != -1:
                slack = matching.edge_slack(e)
                if slack <= delta:
                    delta_type = 2
                    delta = slack
                    delta_edge = e

    # Compute delta3: half minimum slack of any edge between two top-level
    # S-blossoms.
    for b in range(2 * matching.graph.num_vertex):
        if (stage_data.blossom_label[b] == _LABEL_S
                and matching.blossom_parent[b] == -1):
            e = stage_data.blossom_best_edge[b]
            if e != -1:
                slack = matching.edge_slack(e)
                if matching.graph.integer_weights:
                    # If all edge weights are even integers, the slack
                    # of any edge between two S blossoms is also an even
                    # integer. Therefore the delta is an integer.
                    assert slack % 2 == 0
                    slack = slack // 2
                else:
                    slack = slack / 2
                if slack <= delta:
                    delta_type = 3
                    delta = slack
                    delta_edge = e

    # Compute delta4: half minimum dual variable of any T-blossom.
    for b in range(num_vertex, 2 * num_vertex):
        if (stage_data.blossom_label[b] == _LABEL_T
                and matching.blossom_parent[b] == -1):
            slack = matching.get_blossom(b).half_dual_var
            if slack < delta:
                delta_type = 4
                delta = slack
                delta_blossom = b

    return (delta_type, delta, delta_edge, delta_blossom)


def _apply_delta_step(
        matching: _PartialMatching,
        stage_data: _StageData,
        delta: int|float
        ) -> None:
    """Apply a delta step to the dual LPP variables."""

    num_vertex = matching.graph.num_vertex

    # Apply delta to dual variables of all vertices.
    for v in range(num_vertex):
        vlabel = stage_data.blossom_label[matching.vertex_blossom[v]]
        if vlabel == _LABEL_S:
            # S-vertex: subtract delta from dual variable.
            matching.dual_var[v] -= delta
        elif vlabel == _LABEL_T:
            # T-vertex: add delta to dual variable.
            matching.dual_var[v] += delta

    # Apply delta to dual variables of top-level non-trivial blossoms.
    for b in range(num_vertex, 2 * num_vertex):
        blabel = stage_data.blossom_label[b]
        if blabel == _LABEL_S:
            # S-blossom: add 2*delta to dual variable.
            assert matching.blossom_parent[b] == -1
            matching.get_blossom(b).half_dual_var += delta
        elif blabel == _LABEL_T:
            # T-blossom: subtract 2*delta from dual variable.
            assert matching.blossom_parent[b] == -1
            matching.get_blossom(b).half_dual_var -= delta


def _run_stage(matching: _PartialMatching) -> bool:
    """Run one stage of the matching algorithm.

    The stage searches a maximum-weight augmenting path.
    If this path is found, it is used to augment the matching,
    thereby increasing the number of matched edges by 1.
    If no such path is found, the matching must already be optimal.

    This function takes time O(n**2).

    Returns:
        True if the matching was successfully augmented.
        False if no further improvement is possible.
    """

    # Initialize stage data structures.
    stage_data = _StageData(matching.graph)

    # Assign label S to all unmatched vertices and put them in the queue.
    _stage_mark_unmatched_vertices(matching, stage_data)

    # Stop if all vertices are matched.
    # No further improvement is possible in this case.
    # This avoids messy calculations of delta steps without any S-vertex.
    if not stage_data.queue:
        return False

    # Each pass through the following loop is a "substage".
    # The substage tries to find an augmenting path. If such a path is found,
    # we augment the matching and end the stage. Otherwise we update the
    # dual LPP problem and enter the next substage.
    #
    # This loop runs through at most O(n) iterations per stage.
    augmenting_path = None
    while True:

        # Scan to expand the alternating trees.
        # End the stage if an augmenting path is found.
        augmenting_path = _substage_scan(matching, stage_data)
        if augmenting_path is not None:
            break

        # Calculate delta step in the dual LPP problem.
        (delta_type, delta, delta_edge, delta_blossom
            ) = _calc_dual_delta(matching, stage_data)

        # Apply the delta step to the dual variables.
        _apply_delta_step(matching, stage_data, delta)

        if delta_type == 2:
            # Use the edge from S-vertex to unlabeled vertex that got
            # unlocked through the delta update.
            (v, w, _wt) = matching.graph.edges[delta_edge]
            if (stage_data.blossom_label[matching.vertex_blossom[v]] !=
                    _LABEL_S):
                (v, w) = (w, v)
            _substage_add_unlabeled(matching, stage_data, v, w)

        elif delta_type == 3:
            # Use the S-to-S edge that got unlocked through the delta update.
            # This may reveal an augmenting path.
            (v, w, _wt) = matching.graph.edges[delta_edge]
            augmenting_path = _substage_add_s_to_s_edge(
                matching, stage_data, v, w)
            if augmenting_path is not None:
                break

        elif delta_type == 4:
            # Expand the T-blossom that reached dual value 0 through
            # the delta update.
            _expand_t_blossom(matching, stage_data, delta_blossom)

        else:
            # No further improvement possible. End the stage.
            assert delta_type == 1
            break

    # Augment the matching if an augmenting path was found.
    if augmenting_path is not None:
        _augment_matching(matching, augmenting_path)

    # At the end of the stage, expand all blossoms with dual variable zero.
    # In practice, these are always S-blossoms, since T-blossoms typically
    # get expanded as soon as their dual variable hits zero.
    _expand_zero_dual_blossoms(matching)

    # Return True if the matching was augmented.
    return (augmenting_path is not None)


def _verify_optimum(
        graph: _GraphInfo,
        pairs: list[tuple[int, int]],
        vertex_dual_var: list[int],
        blossom_parent: list[int],
        blossom_dual_var: list[int]
        ) -> None:
    """Verify that the optimum solution has been found.

    This function takes time O(m*n).

    Raises:
        AssertionError: If the solution is not optimal.
    """

    # Find mate of each matched vertex.
    # Double-check that each vertex is matched to at most one other.
    vertex_mate = (graph.num_vertex) * [-1]
    for (i, j) in pairs:
        assert vertex_mate[i] == -1
        assert vertex_mate[j] == -1
        vertex_mate[i] = j
        vertex_mate[j] = i

    # Double-check that each matching edge actually exists in the graph.
    nmatched = 0
    for (i, j, _wt) in graph.edges:
        if vertex_mate[i] == j:
            nmatched += 1
    assert len(pairs) == nmatched

    # Check that all dual variables are non-negative.
    assert min(vertex_dual_var) >= 0
    assert min(blossom_dual_var) >= 0

    # Count the number of vertices in each blossom.
    blossom_nvertex = (2 * graph.num_vertex) * [0]
    for v in range(graph.num_vertex):
        b = blossom_parent[v]
        while b != -1:
            blossom_nvertex[b] += 1
            b = blossom_parent[b]

    # Calculate slack of each edge.
    # Also count the number of matched edges in each blossom.
    blossom_nmatched = (2 * graph.num_vertex) * [0]

    for (i, j, wt) in graph.edges:

        # List blossoms that contain vertex "i".
        iblossoms = []
        bi = blossom_parent[i]
        while b != -1:
            iblossoms.append(b)
            b = blossom_parent[b]

        # List blossoms that contain vertex "j".
        jblossoms = []
        bj = blossom_parent[j]
        while b != -1:
            jblossoms.append(b)
            b = blossom_parent[b]

        # List blossoms that contain the edge (i, j).
        edge_blossoms = []
        for (bi, bj) in zip(iblossoms, jblossoms):
            if bi != bj:
                break
            edge_blossoms.append(bi)

        # Calculate edge slack =
        #   dual[i] + dual[j] - weight
        #     + sum(dual[b] for blossoms "b" containing the edge)
        #
        # Note we always multiply edge weights by 2.
        slack = vertex_dual_var[i] + vertex_dual_var[j] - 2 * wt
        slack += sum(blossom_dual_var[b] for b in edge_blossoms)

        # Check that all edges have non-negative slack.
        assert slack >= 0

        # Check that all matched edges have zero slack.
        if vertex_mate[i] == j:
            assert slack == 0

        # Update number of matched edges in each blossom.
        if vertex_mate[i] == j:
            for b in edge_blossoms:
                blossom_nmatched[b] += 1

    # Check that all unmatched vertices have zero dual.
    for v in range(graph.num_vertex):
        if vertex_mate[v] == -1:
            assert vertex_dual_var[v] == 0

    # Check that all blossoms with positive dual are "full".
    # A blossom is full if all except one of its vertices are matched
    # to another vertex in the same blossom.
    for b in range(graph.num_vertex, 2 * graph.num_vertex):
        if blossom_dual_var[b] > 0:
            assert blossom_nvertex[b] == 2 * blossom_nmatched[b] + 1

    # Optimum solution confirmed.
