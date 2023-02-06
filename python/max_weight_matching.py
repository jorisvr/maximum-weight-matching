"""
Algorithm for finding a maximum weight matching in general graphs.
"""

from __future__ import annotations

import sys
import math
from typing import NamedTuple, Optional


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
    Edges with negative weight are ignored.

    This function takes time O(n**3), where "n" is the number of vertices.
    This function uses O(n + m) memory, where "m" is the number of edges.

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, w)"
            where "i" and "j" are vertex indices and "w" is the edge weight.

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

    # Remove edges with negative weight.
    edges = _remove_negative_weight_edges(edges)

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
        (i, j) for (i, j, _w) in edges if matching.vertex_mate[i] == j]

    # Verify that the matching is optimal.
    # This only works reliably for integer weights.
    # Verification is a redundant step; if the matching algorithm is correct,
    # verification will always pass.
    if graph.integer_weights:
        _verify_optimum(matching)

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
        edges: List of edges, each edge specified as a tuple "(i, j, w)"
            where "i" and "j" are vertex indices and "w" is the edge weight.

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

    num_vertex = 1 + max(max(i, j) for (i, j, _w) in edges)

    min_weight = min(w for (_i, _j, w) in edges)
    max_weight = max(w for (_i, _j, w) in edges)
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
    return [(i, j, w + delta) for (i, j, w) in edges]


def _check_input_types(edges: list[tuple[int, int, int|float]]) -> None:
    """Check that the input consists of valid data types and valid
    numerical ranges.

    This function takes time O(m).

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, w)"
            where "i" and "j" are edge indices and "w" is the edge weight.

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

        (i, j, w) = e

        if (not isinstance(i, int)) or (not isinstance(j, int)):
            raise TypeError("Edge endpoints must be integers")

        if (i < 0) or (j < 0):
            raise ValueError("Edge endpoints must be non-negative integers")

        if not isinstance(w, (int, float)):
            raise TypeError(
                "Edge weights must be integers or floating point numbers")

        if isinstance(w, float):
            if not math.isfinite(w):
                raise ValueError("Edge weights must be finite numbers")

            # Check that this edge weight will not cause our dual variable
            # calculations to exceed the valid floating point range.
            if w > float_limit:
                raise ValueError("Floating point edge weights must be"
                                 f" less than {float_limit:g}")


def _check_input_graph(edges: list[tuple[int, int, int|float]]) -> None:
    """Check that the input is a valid graph, without any multi-edges and
    without any self-edges.

    This function takes time O(m * log(m)).

    Parameters:
        edges: List of edges, each edge specified as a tuple "(i, j, w)"
            where "i" and "j" are edge indices and "w" is the edge weight.

    Raises:
        ValueError: If the input does not satisfy the constraints.
    """

    # Check that the graph has no self-edges.
    for (i, j, _w) in edges:
        if i == j:
            raise ValueError("Self-edges are not supported")

    # Check that the graph does not have multi-edges.
    # Using a set() would be more straightforward, but the runtime bounds
    # of the Python set type are not clearly specified.
    # Sorting provides guaranteed O(m * log(m)) run time.
    edge_endpoints = [((i, j) if (i < j) else (j, i)) for (i, j, _w) in edges]
    edge_endpoints.sort()

    for i in range(len(edge_endpoints) - 1):
        if edge_endpoints[i] == edge_endpoints[i+1]:
            raise ValueError(f"Duplicate edge {edge_endpoints[i]}")


def _remove_negative_weight_edges(
        edges: list[tuple[int, int, int|float]]
        ) -> list[tuple[int, int, int|float]]:
    """Remove edges with negative weight.

    This does not change the solution of the maximum-weight matching problem,
    but prevents complications in the algorithm.
    """
    if any(e[2] < 0 for e in edges):
        return [e for e in edges if e[2] >= 0]
    else:
        return edges


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
        # "edges[e] = (i, j, w)" where
        #     "e" is an edge index;
        #     "i" and "j" are vertex indices of the incident vertices;
        #     "w" is the edge weight.
        #
        # These data remain unchanged while the algorithm runs.
        self.edges: list[tuple[int, int, int|float]] = edges

        # num_vertex = the number of vertices.
        if edges:
            self.num_vertex = 1 + max(max(i, j) for (i, j, _w) in edges)
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
        for (e, (i, j, _w)) in enumerate(edges):
            self.adjacent_edges[i].append(e)
            self.adjacent_edges[j].append(e)

        # Determine whether _all_ weights are integers.
        # In this case we can avoid floating point computations entirely.
        self.integer_weights: bool = all(isinstance(w, int)
                                         for (_i, _j, w) in edges)


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

    Each blossom contains exactly one vertex that is not matched to another
    vertex in the same blossom. This is the "base vertex" of the blossom.

    Blossoms are created and destroyed by the matching algorithm.
    This implies that not every odd-length alternating cycle is a blossom;
    it only becomes a blossom through an explicit action of the algorithm.
    An existing blossom may be changed when the matching is augmented
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
        # Each edge is represented as an ordered pair "(i, j)" where "i"
        # and "j" are vertex indices.
        #
        # "edges[0] = (i, j)" where vertex "i" in "subblossoms[0]" is
        # adjacent to vertex "j" in "subblossoms[1]", etc.
        self.edges: list[tuple[int, int]] = edges

        # "base_vertex" is the vertex index of the base of the blossom.
        # This is the unique vertex which is contained in the blossom
        # but not matched to another vertex in the same blossom.
        self.base_vertex: int = base_vertex

        # Every blossom has a variable in the dual LPP.
        #
        # "dual_var" is the current value of the dual variable.
        # New blossoms start with dual variable 0.
        self.dual_var: int|float = 0


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
        # If vertex "i" is matched to vertex "j",
        # "vertex_mate[i] == j" and "vertex_mate[j] == i".
        # If vertex "i" is unmatched, "vertex_mate[i] == -1".
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
            reversed(range(graph.num_vertex, 2 * graph.num_vertex)))

        # TODO : we may need a list of top-level blossom indices

        # Every vertex is part of exactly one top-level blossom,
        # possibly a trivial blossom consisting of just that vertex.
        #
        # "vertex_blossom[i]" is the index of the top-level blossom that
        # contains vertex "i".
        # "vertex_blossom[i] == i" if the "i" is a trivial top-level blossom.
        #
        # Initially all vertices are top-level trivial blossoms.
        self.vertex_blossom: list[int] = list(range(graph.num_vertex))

        # "blossom_parent[b]" is the index of the smallest blossom that
        # is a strict superset of blossom "b", or
        # "blossom_parent[b] == -1" if blossom "b" is a top-level blossom.
        #
        # Initially all vertices are trivial top-level blossoms.
        self.blossom_parent: list[int] = (2 * graph.num_vertex) * [-1]

        # Every vertex has a variable in the dual LPP.
        #
        # "dual_var_2x[i]" is 2 times the dual variable of vertex "i".
        # Multiplication by 2 ensures that the values are integers
        # if all edge weights are integers.
        #
        # Vertex duals are initialized to half the maximum edge weight.
        max_weight = max(w for (_i, _j, w) in graph.edges)
        self.dual_var_2x: list[int|float] = graph.num_vertex * [max_weight]

    def edge_slack_2x(self, e: int) -> int|float:
        """Return 2 times the slack of the edge with index "e".

        The result is only valid for edges that are not between vertices
        that belong to the same top-level blossom.

        Multiplication by 2 ensures that the return value is an integer
        if all edge weights are integers.
        """
        (i, j, w) = self.graph.edges[e]
        assert self.vertex_blossom[i] != self.vertex_blossom[j]
        return self.dual_var_2x[i] + self.dual_var_2x[j] - 2 * w

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
        # "blossom_link[b] = (i, j)" denotes the edge through which
        # blossom "b" is attached to the alternating tree, where "i" and "j"
        # are vertex indices and vertex "j" is contained in blossom "b".
        #
        # "blossom_link[b] = None" if "b" is the root of an alternating tree,
        # or if "b" is not a labeled, top-level blossom.
        self.blossom_link: list[Optional[tuple[int, int]]] = [
            None for b in range(2 * num_vertex)]

        # "vertex_best_edge[i]" is the edge index of the least-slack edge
        # between "i" and any S-vertex, or -1 if no such edge has been found.
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
        i: int,
        j: int
        ) -> _AlternatingPath:
    """Trace back through the alternating trees from vertices "i" and "j".

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

    # "iedges" is a list of edges used while tracing from "i".
    # "jedges" is a list of edges used while tracing from "j".
    iedges: list[tuple[int, int]] = []
    jedges: list[tuple[int, int]] = []

    # Pre-load the edge between "i" and "j" so it will end up in the right
    # place in the final path.
    iedges.append((i, j))

    # Alternate between tracing the path from "i" and the path from "j".
    # This ensures that the search time is bounded by the size of the
    # newly found blossom.
    first_common = -1
    while i != -1 or j != -1:

        # Check if we found a common ancestor.
        bi = vertex_blossom[i]
        if blossom_marker[bi]:
            first_common = bi
            break

        # Mark blossom as a potential common ancestor.
        blossom_marker[bi] = True
        marked_blossoms.append(bi)

        # Track back through the link in the alternating tree.
        link = blossom_link[bi]
        if link is None:
            # Reached the root of this alternating tree.
            i = -1
        else:
            iedges.append(link)
            i = link[0]

        # Swap "i" and "j" to alternate between paths.
        if j != -1:
            i, j = j, i
            iedges, jedges = jedges, iedges

    # Remove all markers we placed.
    for b in marked_blossoms:
        blossom_marker[b] = False

    # If we found a common ancestor, trim the paths so they end there.
    if first_common != -1:
        assert vertex_blossom[iedges[-1][0]] == first_common
        while jedges and (vertex_blossom[jedges[-1][0]] != first_common):
            jedges.pop()

    # Fuse the two paths.
    # Flip the order of one path and the edge tuples in the other path
    # to obtain a single continuous path with correctly ordered edge tuples.
    path_edges = iedges[::-1] + [(j, i) for (i, j) in jedges]

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
    subblossoms = [matching.vertex_blossom[i] for (i, j) in path.edges]

    # Check that the path is cyclic.
    # Note the path may not start and end with the same _vertex_,
    # but it must start and end in the same _blossom_.
    subblossoms_next = [matching.vertex_blossom[j] for (i, j) in path.edges]
    assert subblossoms[0] == subblossoms_next[-1]
    assert subblossoms[1:] == subblossoms_next[:-1]

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
    for i in matching.blossom_vertices(b):
        matching.vertex_blossom[i] = b

    # Assign label S to the new blossom.
    assert stage_data.blossom_label[base_blossom] == _LABEL_S
    stage_data.blossom_label[b] = _LABEL_S
    stage_data.blossom_link[b] = stage_data.blossom_link[base_blossom]

    # Former T-vertices which are part of this blossom have now become
    # S-vertices. Add them to the queue.
    for sub in subblossoms:
        if stage_data.blossom_label[sub] == _LABEL_T:
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
    zero_slack: int|float = 0
    best_slack_to_blossom: list[int|float] = (2 * num_vertex) * [zero_slack]

    # Add the least-slack edges of every S-sub-blossom.
    for sub in subblossoms:
        if stage_data.blossom_label[sub] != _LABEL_S:
            continue
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
            (i, j, _w) = matching.graph.edges[e]
            bi = matching.vertex_blossom[i]
            bj = matching.vertex_blossom[j]
            assert (bi == b) or (bj == b)

            # Reject internal edges in this blossom.
            if bi == bj:
                continue

            # Set bi = other blossom which is reachable through this edge.
            bi = bj if (bi == b) else bi

            # Reject edges that don't link to an S-blossom.
            if stage_data.blossom_label[bi] != _LABEL_S:
                continue

            # Keep only the least-slack edge to "vblossom".
            slack = matching.edge_slack_2x(e)
            if ((best_edge_to_blossom[bi] == -1)
                    or (slack < best_slack_to_blossom[bi])):
                best_edge_to_blossom[bi] = e
                best_slack_to_blossom[bi] = slack

    # Extract a compact list of least-slack edge indices.
    # We can not keep the temporary array because that would blow up
    # memory use to O(n**2).
    best_edge_set = [e for e in best_edge_to_blossom if e != -1]
    stage_data.blossom_best_edge_set[b] = best_edge_set

    # Select the overall least-slack edge to any other S-blossom.
    best_edge = -1
    best_slack: int|float = 0
    for e in best_edge_set:
        slack = matching.edge_slack_2x(e)
        if (best_edge == -1) or (slack < best_slack):
            best_edge = e
            best_slack = slack
    stage_data.blossom_best_edge[b] = best_edge


def _substage_assign_label_s(
        matching: _PartialMatching,
        stage_data: _StageData,
        i: int,
        ) -> None:
    """Assign label S to the unlabeled blossom that contains vertex "i".

    If vertex "i" is matched, it is attached to the alternating tree via its
    matched edge. If vertex "i" is unmatched, it becomes the root of
    an alternating tree.

    All vertices in the newly labeled S-blossom are added to the scan queue.

    Precondition:
        "i" is an unlabeled vertex, either unmatched or matched to
        a T-vertex via a tight edge.
    """

    # Assign label S to the blossom that contains vertex "v".
    bi = matching.vertex_blossom[i]
    assert stage_data.blossom_label[bi] == _LABEL_NONE
    stage_data.blossom_label[bi] = _LABEL_S

    j = matching.vertex_mate[i]
    if j == -1:
        # Vertex "i" is unmatched.
        # It must be either a top-level vertex or the base vertex of
        # a top-level blossom.
        assert (bi == i) or (matching.get_blossom(bi).base_vertex == i)

        # Mark the blossom that contains "v" as root of an alternating tree.
        stage_data.blossom_link[bi] = None

    else:
        # Vertex "i" is matched to T-vertex "j".
        bj = matching.vertex_blossom[j]
        assert stage_data.blossom_label[bj] == _LABEL_T

        # Attach the blossom that contains "i" to the alternating tree.
        stage_data.blossom_link[bi] = (j, i)

    # Initialize the list of least-slack edges of the newly labeled blossom.
    # This list will be filled by scanning the vertices of the blossom.
    stage_data.blossom_best_edge_set[bi] = []

    # Add all vertices inside the newly labeled S-blossom to the queue.
    if bi == i:
        stage_data.queue.append(i)
    else:
        stage_data.queue.extend(matching.blossom_vertices(bi))


def _substage_assign_label_t(
        matching: _PartialMatching,
        stage_data: _StageData,
        i: int,
        j: int
        ) -> None:
    """Assign label T to the unlabeled blossom that contains vertex "j".

    Attach it to the alternating tree via edge (i, j).
    Then immediately assign label S to the mate of vertex "j".

    Preconditions:
     - "i" is an S-vertex.
     - "j" is an unlabeled, matched vertex.
     - There is a tight edge between vertices "i" and "j".
    """
    assert stage_data.blossom_label[matching.vertex_blossom[i]] == _LABEL_S

    # Assign label T to the unlabeled blossom.
    bj = matching.vertex_blossom[j]
    assert stage_data.blossom_label[bj] == _LABEL_NONE
    stage_data.blossom_label[bj] = _LABEL_T
    stage_data.blossom_link[bj] = (i, j)

    # Assign label S to the blossom that contains the mate of vertex "j".
    jbase = j if bj == j else matching.get_blossom(bj).base_vertex
    k = matching.vertex_mate[jbase]
    assert k != -1
    _substage_assign_label_s(matching, stage_data, k)


def _substage_add_s_to_s_edge(
        matching: _PartialMatching,
        stage_data: _StageData,
        i: int,
        j: int
        ) -> Optional[_AlternatingPath]:
    """Add the edge between S-vertices "i" and "j".

    If the edge connects blossoms that are part of the same alternating tree,
    this function creates a new S-blossom and returns None.

    If the edge connects two different alternating trees, an augmenting
    path has been discovered. In this case the function changes nothing
    and returns the augmenting path.

    Returns:
        Augmenting path if found; otherwise None.
    """

    # Trace back through the alternating trees from "i" and "j".
    path = _trace_alternating_paths(matching, stage_data, i, j)

    # If the path is a cycle, create a new blossom.
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
        i = stage_data.queue.pop()

        # Double-check that "v" is an S-vertex.
        bi = matching.vertex_blossom[i]
        assert stage_data.blossom_label[bi] == _LABEL_S

        # Scan the edges that are incident on "v".
        for e in adjacent_edges[i]:
            (p, q, _w) = edges[e]
            j = p if p != i else q  # TODO : consider abstracting this

            # Consider the edge between vertices "i" and "j".
            # Try to pull this edge into an alternating tree.

            # Note: blossom index of vertex "i" may change during this loop,
            # so we need to refresh it here.
            bi = matching.vertex_blossom[i]
            bj = matching.vertex_blossom[j]

            # Ignore edges that are internal to a blossom.
            if bi == bj:
                continue

            jlabel = stage_data.blossom_label[bj]

            # Check whether this edge is tight (has zero slack).
            # Only tight edges may be part of an alternating tree.
            slack = matching.edge_slack_2x(e)
            if slack <= 0:
                if jlabel == _LABEL_NONE:
                    # Assign label T to the blossom that contains vertex "j".
                    _substage_assign_label_t(matching, stage_data, i, j)
                elif jlabel == _LABEL_S:
                    # This edge connects two S-blossoms. Use it to find
                    # either a new blossom or an augmenting path.
                    alternating_path = _substage_add_s_to_s_edge(matching,
                                                                 stage_data,
                                                                 i, j)
                    if alternating_path is not None:
                        return alternating_path

            elif jlabel == _LABEL_S:
                # Update tracking of least-slack edges between S-blossoms.
                best_edge = stage_data.blossom_best_edge[bi]
                if ((best_edge < 0)
                        or (slack < matching.edge_slack_2x(best_edge))):
                    stage_data.blossom_best_edge[bi] = e

                # Update the list of least-slack edges to S-blossoms for
                # the blossom that contains "i".
                # We only do this for non-trivial blossoms.
                if bi != i:
                    best_edge_set = stage_data.blossom_best_edge_set[bi]
                    assert best_edge_set is not None
                    best_edge_set.append(e)

            if jlabel != _LABEL_S:
                # Update tracking of least-slack edges from vertex "j" to
                # any S-vertex. We do this only for T-vertices and unlabeled
                # vertices. Edges which already have zero slack are still
                # tracked.
                best_edge = stage_data.vertex_best_edge[j]
                if best_edge < 0 or slack < matching.edge_slack_2x(best_edge):
                    stage_data.vertex_best_edge[j] = e

    # No further S vertices to scan, and no augmenting path found.
    return None


def _find_path_through_blossom(
        matching: _PartialMatching,
        b: int,
        sub: int
        ) -> tuple[list[int], list[tuple[int, int]]]:
    """Construct a path through blossom "b" from sub-blossom "sub"
    to the base of the blossom.

    Return:
        Tuple (nodes, edges).
    """
    blossom = matching.blossom[b]
    assert blossom is not None

    nodes: list[int] = [sub]
    edges: list[tuple[int, int]] = []

    # Walk around the blossom from "s" to its base.
    p = blossom.subblossoms.index(sub)
    nsub = len(blossom.subblossoms)
    while p != 0:
        if p % 2 == 0:
            # Stepping towards the beginning of the subblossom list.
            # Currently at subblossom (p), next position (p-2):
            #
            #  0 --- 1 === 2 --- 3 === (p-2) --- (p-1) ==(i,j)== (p)
            #                           ^^^                      ^^^
            #                               <-------------------
            #
            # We flip edges from (i,j) to (j,i) to make them fit
            # in the path from "s" to base.
            edges.append(blossom.edges[p-1][::-1])
            nodes.append(blossom.subblossoms[p-1])
            edges.append(blossom.edges[p-2][::-1])
            nodes.append(blossom.subblossoms[p-2])
            p -= 2
        else:
            # Stepping towards the end of the subblossom list.
            # Currently at subblossom (p), next position (p+2):
            #
            #  (p) ==(i,j)== (p+1) --- (p+2) === (p+3) --- 0
            #  ^^^                      ^^^
            #      ------------------->
            edges.append(blossom.edges[p])
            nodes.append(blossom.subblossoms[p+1])
            edges.append(blossom.edges[p+1])
            nodes.append(blossom.subblossoms[(p+2) % nsub])
            p = (p + 2) % nsub

    return (nodes, edges)


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
            for i in matching.blossom_vertices(sub):
                matching.vertex_blossom[i] = sub
        assert stage_data.blossom_label[sub] == _LABEL_NONE

    # The expanding blossom was part of an alternating tree, linked to
    # a parent node in the tree via one of its subblossoms, and linked to
    # a child node of the tree via the base vertex.
    # We must reconstruct this part of the alternating tree, which will
    # now run via sub-blossoms of the expanded blossom.

    # Find the sub-blossom that is attached to the parent node in
    # the alternating tree.
# TODO : uglyness with the assertion
    entry_link = stage_data.blossom_link[b]
    assert entry_link is not None
    (i, j) = entry_link
    sub = matching.vertex_blossom[j]

    # Assign label T to that sub-blossom.
    stage_data.blossom_label[sub] = _LABEL_T
    stage_data.blossom_link[sub] = (i, j)

    # Walk through the expanded blossom from "sub" to the base vertex.
    # Assign alternating S and T labels to the sub-blossoms and attach
    # them to the alternating tree.
    (path_nodes, path_edges) = _find_path_through_blossom(matching, b, sub)

    for p in range(0, len(path_edges), 2):
        #
        #   (p) ==(j,i)== (p+1) ----- (p+2)
        #    T              S           T
        #
        # path_nodes[p] has already been labeled T.
        # We now assign labels to path_nodes[p+1] and path_nodes[p+2].

        # Assign label S to path_nodes[p+1].
        (j, i) = path_edges[p]
        _substage_assign_label_s(matching, stage_data, i)

        # Assign label T to path_nodes[i+2] and attach it to path_nodes[p+1].
        sub = path_nodes[p+2]
        stage_data.blossom_label[sub] = _LABEL_T
        stage_data.blossom_link[sub] = path_edges[p+1]

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
            if blossom.dual_var == 0:
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
                # If its dual variable is zero, expand it recursively.
                if matching.get_blossom(sub).dual_var == 0:
                    stack.append(sub)
                else:
                    # This sub-blossom will not be expanded;
                    # it now becomes top-level. Update its vertices
                    # to point to this sub-blossom.
                    for i in matching.blossom_vertices(sub):
                        matching.vertex_blossom[i] = sub

        # Delete the expanded blossom. Recycle its blossom index.
        matching.blossom[b] = None
        matching.unused_blossoms.append(b)


def _augment_blossom(matching: _PartialMatching, b: int, i: int) -> None:
    """Augment along an alternating path through blossom "b", from vertex "i"
    to the base vertex of the blossom.

    This function takes time O(n).
    """
    num_vertex = matching.graph.num_vertex

    # Use an explicit stack to avoid deep recursion.
    stack = [(b, i)]

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
        (path_nodes, path_edges) = _find_path_through_blossom(matching,
                                                              b, sub)

        for p in range(0, len(path_edges), 2):
            # Before augmentation:
            #   path_nodes[p] is matched to path_nodes[p+1]
            #
            #   (p) ===== (p+1) ---(i,j)--- (p+2)
            #
            # After augmentation:
            #   path_nodes[p+1] is mathed to path_nodes[p+2] via edge (i, j)
            #
            #   (p) ----- (p+1) ===(i,j)=== (p+2)
            #

            # Pull the edge (i, j) into the matching.
            (i, j) = path_edges[p+1]
            matching.vertex_mate[i] = j
            matching.vertex_mate[j] = i

            # Augment through the subblossoms touching the edge (i, j).
            # Nothing needs to be done for trivial subblossoms.
            bi = path_nodes[p+1]
            if bi >= num_vertex:
                stack.append((bi, i))

            bj = path_nodes[p+2]
            if bj >= num_vertex:
                stack.append((bj, j))

        # Rotate the subblossom list so the new base ends up in position 0.
        p = blossom.subblossoms.index(sub)
        blossom.subblossoms = (
            blossom.subblossoms[p:] + blossom.subblossoms[:p])
        blossom.edges = blossom.edges[p:] + blossom.edges[:p]

        # Update the base vertex.
        # We can pull this from the sub-blossom where we started since
        # its augmentation has already finished.
        if sub < num_vertex:
            blossom.base_vertex = sub
        else:
            blossom.base_vertex = matching.get_blossom(sub).base_vertex


def _augment_matching(
        matching: _PartialMatching,
        path: _AlternatingPath
        ) -> None:
    """Augment the matching through the specified augmenting path.

    This function takes time O(n).
    """

    # Check that the augmenting path starts and ends in an unmatched vertex
    # or a blossom with unmatched base.
    assert len(path.edges) % 2 == 1
    for i in (path.edges[0][0], path.edges[-1][1]):
        b = matching.vertex_blossom[i]
        if b != i:
            i = matching.get_blossom(b).base_vertex
        assert matching.vertex_mate[i] == -1

    # Consider the edges of the augmenting path that are currently not
    # part of the matching but will become part of it.
    for (i, j) in path.edges[0::2]:

        # Augment through non-trivial blossoms on either side of this edge.
        bi = matching.vertex_blossom[i]
        if bi != i:
            _augment_blossom(matching, bi, i)

        bj = matching.vertex_blossom[j]
        if bj != j:
            _augment_blossom(matching, bj, j)

        # Pull the edge into the matching.
        matching.vertex_mate[i] = j
        matching.vertex_mate[j] = i


def _calc_dual_delta(
        matching: _PartialMatching,
        stage_data: _StageData
        ) -> tuple[int, float|int, int, int]:
    """Calculate a delta step in the dual LPP problem.

    This function returns the minimum of the 4 types of delta values,
    and the type of delta which obtain the minimum, and the edge or blossom
    that produces the minimum delta, if applicable.

    The returned value is 2 times the actual delta value.
    Multiplication by 2 ensures that the result is an integer if all edge
    weights are integers.

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
    delta_2x = min(
        matching.dual_var_2x[i]
        for i in range(num_vertex)
        if stage_data.blossom_label[matching.vertex_blossom[i]] == _LABEL_S)

    # Compute delta2: minimum slack of any edge between an S-vertex and
    # an unlabeled vertex.
    for i in range(num_vertex):
        bi = matching.vertex_blossom[i]
        if stage_data.blossom_label[bi] == _LABEL_NONE:
            e = stage_data.vertex_best_edge[i]
            if e != -1:
                slack = matching.edge_slack_2x(e)
                if slack <= delta_2x:
                    delta_type = 2
                    delta_2x = slack
                    delta_edge = e

    # Compute delta3: half minimum slack of any edge between two top-level
    # S-blossoms.
    for b in range(2 * matching.graph.num_vertex):
        if (stage_data.blossom_label[b] == _LABEL_S
                and matching.blossom_parent[b] == -1):
            e = stage_data.blossom_best_edge[b]
            if e != -1:
                slack = matching.edge_slack_2x(e)
                if matching.graph.integer_weights:
                    # If all edge weights are even integers, the slack
                    # of any edge between two S blossoms is also an even
                    # integer. Therefore the delta is an integer.
                    assert slack % 2 == 0
                    slack = slack // 2
                else:
                    slack = slack / 2
                if slack <= delta_2x:
                    delta_type = 3
                    delta_2x = slack
                    delta_edge = e

    # Compute delta4: half minimum dual variable of any T-blossom.
    for b in range(num_vertex, 2 * num_vertex):
        if (stage_data.blossom_label[b] == _LABEL_T
                and matching.blossom_parent[b] == -1):
            slack = matching.get_blossom(b).dual_var
            if slack < delta_2x:
                delta_type = 4
                delta_2x = slack
                delta_blossom = b

    return (delta_type, delta_2x, delta_edge, delta_blossom)


def _apply_delta_step(
        matching: _PartialMatching,
        stage_data: _StageData,
        delta_2x: int|float
        ) -> None:
    """Apply a delta step to the dual LPP variables."""

    num_vertex = matching.graph.num_vertex

    # Apply delta to dual variables of all vertices.
    for i in range(num_vertex):
        ilabel = stage_data.blossom_label[matching.vertex_blossom[i]]
        if ilabel == _LABEL_S:
            # S-vertex: subtract delta from dual variable.
            matching.dual_var_2x[i] -= delta_2x
        elif ilabel == _LABEL_T:
            # T-vertex: add delta to dual variable.
            matching.dual_var_2x[i] += delta_2x

    # Apply delta to dual variables of top-level non-trivial blossoms.
    for b in range(num_vertex, 2 * num_vertex):
        if matching.blossom_parent[b] == -1:
            blabel = stage_data.blossom_label[b]
            if blabel == _LABEL_S:
                # S-blossom: add 2*delta to dual variable.
                matching.get_blossom(b).dual_var += delta_2x
            elif blabel == _LABEL_T:
                # T-blossom: subtract 2*delta from dual variable.
                matching.get_blossom(b).dual_var -= delta_2x


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

    num_vertex = matching.graph.num_vertex

    # Initialize stage data structures.
    stage_data = _StageData(matching.graph)

    # Assign label S to all unmatched vertices and put them in the queue.
    for i in range(num_vertex):
        if matching.vertex_mate[i] == -1:
            _substage_assign_label_s(matching, stage_data, i)

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
        (delta_type, delta_2x, delta_edge, delta_blossom
            ) = _calc_dual_delta(matching, stage_data)

        # Apply the delta step to the dual variables.
        _apply_delta_step(matching, stage_data, delta_2x)

        if delta_type == 2:
            # Use the edge from S-vertex to unlabeled vertex that got
            # unlocked through the delta update.
            (i, j, _w) = matching.graph.edges[delta_edge]
            if (stage_data.blossom_label[matching.vertex_blossom[i]] !=
                    _LABEL_S):
                (i, j) = (j, i)
            _substage_assign_label_t(matching, stage_data, i, j)

        elif delta_type == 3:
            # Use the S-to-S edge that got unlocked through the delta update.
            # This may reveal an augmenting path.
            (i, j, _w) = matching.graph.edges[delta_edge]
            augmenting_path = _substage_add_s_to_s_edge(
                matching, stage_data, i, j)
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


def _verify_optimum(matching: _PartialMatching) -> None:
    """Verify that the optimum solution has been found.

    This function takes time O(m * n).

    Raises:
        AssertionError: If the solution is not optimal.
    """

    num_vertex = matching.graph.num_vertex

    vertex_mate = matching.vertex_mate
    vertex_dual_var_2x = matching.dual_var_2x

    # Extract dual values of blossoms
    blossom_dual_var = [
        (blossom.dual_var if blossom is not None else 0)
        for blossom in matching.blossom]

    # Double-check that each matching edge actually exists in the graph.
    num_matched_vertex = 0
    for i in range(num_vertex):
        if vertex_mate[i] != -1:
            num_matched_vertex += 1

    num_matched_edge = 0
    for (i, j, _w) in matching.graph.edges:
        if vertex_mate[i] == j:
            num_matched_edge += 1

    assert num_matched_vertex == 2 * num_matched_edge

    # Check that all dual variables are non-negative.
    assert min(vertex_dual_var_2x) >= 0
    assert min(blossom_dual_var) >= 0

    # Count the number of vertices in each blossom.
    blossom_nvertex = (2 * num_vertex) * [0]
    for i in range(num_vertex):
        b = matching.blossom_parent[i]
        while b != -1:
            blossom_nvertex[b] += 1
            b = matching.blossom_parent[b]

    # Calculate slack of each edge.
    # Also count the number of matched edges in each blossom.
    blossom_nmatched = (2 * num_vertex) * [0]

    for (i, j, w) in matching.graph.edges:

        # List blossoms that contain vertex "i".
        iblossoms = []
        bi = matching.blossom_parent[i]
        while bi != -1:
            iblossoms.append(bi)
            bi = matching.blossom_parent[bi]

        # List blossoms that contain vertex "j".
        jblossoms = []
        bj = matching.blossom_parent[j]
        while bj != -1:
            jblossoms.append(bj)
            bj = matching.blossom_parent[bj]

        # List blossoms that contain the edge (i, j).
        edge_blossoms = []
        for (bi, bj) in zip(reversed(iblossoms), reversed(jblossoms)):
            if bi != bj:
                break
            edge_blossoms.append(bi)

        # Calculate edge slack =
        #   dual[i] + dual[j] - weight
        #     + sum(dual[b] for blossoms "b" containing the edge)
        #
        # Multiply weights by 2 to ensure integer values.
        slack = vertex_dual_var_2x[i] + vertex_dual_var_2x[j] - 2 * w
        slack += 2 * sum(blossom_dual_var[b] for b in edge_blossoms)

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
    for i in range(num_vertex):
        if vertex_mate[i] == -1:
            assert vertex_dual_var_2x[i] == 0

    # Check that all blossoms with positive dual are "full".
    # A blossom is full if all except one of its vertices are matched
    # to another vertex in the same blossom.
    for b in range(num_vertex, 2 * num_vertex):
        if blossom_dual_var[b] > 0:
            assert blossom_nvertex[b] == 2 * blossom_nmatched[b] + 1

    # Optimum solution confirmed.
