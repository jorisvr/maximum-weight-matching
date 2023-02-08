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
        edges: List of edges, each edge specified as a tuple "(x, y, w)"
            where "x" and "y" are vertex indices and "w" is the edge weight.

    Returns:
        List of pairs of matched vertex indices.
        This is a subset of the edges in the graph.
        It contains a tuple "(x, y)" if vertex "x" is matched to vertex "y".

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

    # Initialize the matching algorithm.
    ctx = _MatchingContext(graph)

    # Improve the solution until no further improvement is possible.
    #
    # Each successful pass through this loop increases the number
    # of matched edges by 1.
    #
    # This loop runs through at most (n/2 + 1) iterations.
    # Each iteration takes time O(n**2).
    while ctx.run_stage():
        pass

    # Extract the final solution.
    pairs: list[tuple[int, int]] = [
        (x, y) for (x, y, _w) in edges if ctx.vertex_mate[x] == y]

    # Verify that the matching is optimal.
    # This only works reliably for integer weights.
    # Verification is a redundant step; if the matching algorithm is correct,
    # verification will always pass.
    if graph.integer_weights:
        # TODO : pass selection of data to verification
        #        passing the whole context does not inspire trust that this is an independent verification
        _verify_optimum(ctx)

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
        edges: List of edges, each edge specified as a tuple "(x, y, w)"
            where "x" and "y" are vertex indices and "w" is the edge weight.

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

    num_vertex = 1 + max(max(x, y) for (x, y, _w) in edges)

    min_weight = min(w for (_x, _y, w) in edges)
    max_weight = max(w for (_x, _y, w) in edges)
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
    return [(x, y, w + delta) for (x, y, w) in edges]


def _check_input_types(edges: list[tuple[int, int, int|float]]) -> None:
    """Check that the input consists of valid data types and valid
    numerical ranges.

    This function takes time O(m).

    Parameters:
        edges: List of edges, each edge specified as a tuple "(x, y, w)"
            where "x" and "y" are edge indices and "w" is the edge weight.

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

        (x, y, w) = e

        if (not isinstance(x, int)) or (not isinstance(y, int)):
            raise TypeError("Edge endpoints must be integers")

        if (x < 0) or (y < 0):
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
        edges: List of edges, each edge specified as a tuple "(x, y, w)"
            where "x" and "y" are edge indices and "w" is the edge weight.

    Raises:
        ValueError: If the input does not satisfy the constraints.
    """

    # Check that the graph has no self-edges.
    for (x, y, _w) in edges:
        if x == y:
            raise ValueError("Self-edges are not supported")

    # Check that the graph does not have multi-edges.
    # Using a set() would be more straightforward, but the runtime bounds
    # of the Python set type are not clearly specified.
    # Sorting provides guaranteed O(m * log(m)) run time.
    edge_endpoints = [((x, y) if (x < y) else (y, x)) for (x, y, _w) in edges]
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
        # "edges[e] = (x, y, w)" where
        #     "e" is an edge index;
        #     "x" and "y" are vertex indices of the incident vertices;
        #     "w" is the edge weight.
        #
        # These data remain unchanged while the algorithm runs.
        self.edges: list[tuple[int, int, int|float]] = edges

        # num_vertex = the number of vertices.
        if edges:
            self.num_vertex = 1 + max(max(x, y) for (x, y, _w) in edges)
        else:
            self.num_vertex = 0

        # Each vertex is incident to zero or more edges.
        #
        # "adjacent_edges[x]" is the list of edge indices of edges incident
        # to the vertex with index "x".
        #
        # These data remain unchanged while the algorithm runs.
        self.adjacent_edges: list[list[int]] = [
            [] for v in range(self.num_vertex)]
        for (e, (x, y, _w)) in enumerate(edges):
            self.adjacent_edges[x].append(e)
            self.adjacent_edges[y].append(e)

        # Determine whether _all_ weights are integers.
        # In this case we can avoid floating point computations entirely.
        self.integer_weights: bool = all(isinstance(w, int)
                                         for (_x, _y, w) in edges)


# Each vertex may be labeled "S" (outer) or "T" (inner) or be unlabeled.
_LABEL_NONE = 0
_LABEL_S = 1
_LABEL_T = 2


class _Blossom:
    """Represents a blossom in a partially matched graph.

    A blossom is an odd-length alternating cycle over sub-blossoms.
    An alternating path consists of alternating matched and unmatched edges.
    An alternating cycle is an alternating path that starts and ends in
    the same sub-blossom.

    A single vertex by itself is also a blossom: a "trivial blossom".

    An instance of this class represents either a trivial blossom,
    or a non-trivial blossom which consists of multiple sub-blossoms.

    Blossoms are recursive structures: A non-trivial blossoms contains
    sub-blossoms, which may themselves contain sub-blossoms etc.

    Each blossom contains exactly one vertex that is not matched to another
    vertex in the same blossom. This is the "base vertex" of the blossom.
    """

    def __init__(self, base_vertex: int) -> None:
        """Initialize a new blossom."""

        # If this is not a top-level blossom,
        # "parent" is the blossom in which this blossom is a sub-blossom.
        #
        # If this is a top-level blossom,
        # "parent = None".
        self.parent: Optional[_NonTrivialBlossom] = None

        # "base_vertex" is the vertex index of the base of the blossom.
        # This is the unique vertex which is contained in the blossom
        # but not matched to another vertex in the same blossom.
        #
        # For trivial blossoms, the base vertex is simply the only
        # vertex in the blossom.
        self.base_vertex: int = base_vertex

        # A top-level blossom that are part of an alternating tree,
        # has labels S or T. Unlabeled top-level blossoms are not (yet)
        # part of any alternating tree.
        self.label: int = _LABEL_NONE

        # Labeled top-level blossoms keep track of the edge through which
        # they are attached to an alternating tree.
        #
        # "tree_edge = (x, y)" if the blossom is attached to an alternating
        # tree via edge "(x, y)" and vertex "y" is contained in the blossom.
        #
        # "tree_edge = None" if the blossom is the root of an alternating tree.
        self.tree_edge: Optional[tuple[int, int]] = None

        # For a top-level S-blossom,
        # "best_edge" is the edge index of the least-slack edge to
        # a different S-blossom, or -1 if no such edge has been found.
        self.best_edge: int = -1

        # "marker" is a temporary variable used to discover common
        # ancestors in the blossom tree. It is normally False, except
        # when used by "trace_alternating_paths()".
        self.marker: bool = False

    def vertices(self) -> list[int]:
        """Return a list of vertex indices contained in the blossom."""
        return [self.base_vertex]


class _NonTrivialBlossom(_Blossom):
    """Represents a non-trivial blossom in a partially matched graph.

    A non-trivial blossom is a blossom that contains multiple sub-blossoms
    (at least 3 sub-blossoms, since all blossoms have odd length).

    Non-trivial blossoms maintain a list of their sub-blossoms and the edges
    between their subblossoms.

    Unlike trivial blossoms, each non-trivial blossom is associated with
    a variable in the dual LPP problem.

    Non-trivial blossoms are created and destroyed by the matching algorithm.
    This implies that not every odd-length alternating cycle is a blossom;
    it only becomes a blossom through an explicit action of the algorithm.
    An existing blossom may change when the matching is augmented along
    a path that runs through the blossom.
    """

    def __init__(
            self,
            subblossoms: list[_Blossom],
            edges: list[tuple[int, int]]
            ) -> None:
        """Initialize a new blossom."""

        super().__init__(subblossoms[0].base_vertex)

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
        self.subblossoms: list[_Blossom] = subblossoms

        # "edges" is a list of edges linking the sub-blossoms.
        # Each edge is represented as an ordered pair "(x, y)" where "x"
        # and "y" are vertex indices.
        #
        # "edges[0] = (x, y)" where vertex "x" in "subblossoms[0]" is
        # adjacent to vertex "y" in "subblossoms[1]", etc.
        self.edges: list[tuple[int, int]] = edges

        # Every non-trivial blossom has a variable in the dual LPP.
        #
        # "dual_var" is the current value of the dual variable.
        # New blossoms start with dual variable 0.
        self.dual_var: int|float = 0

        # For a non-trivial, top-level S-blossom,
        # "best_edge_set" is a list of least-slack edges between this blossom
        # and other S-blossoms.
        self.best_edge_set: Optional[list[int]] = None

    def vertices(self) -> list[int]:
        """Return a list of vertex indices contained in the blossom."""

        # Use an explicit stack to avoid deep recursion.
        stack: list[_NonTrivialBlossom] = [self]
        nodes: list[int] = []

        while stack:
            b = stack.pop()
            for sub in b.subblossoms:
                if isinstance(sub, _NonTrivialBlossom):
                    stack.append(sub)
                else:
                    nodes.append(sub.base_vertex)

        return nodes


class _AlternatingPath(NamedTuple):
    """Represents a list of edges forming an alternating path or an
    alternating cycle."""
    edges: list[tuple[int, int]]


class _MatchingContext:
    """Holds all data used by the matching algorithm.

    It contains a partial solution of the matching problem and several
    auxiliary data structures.
    """

    def __init__(self, graph: _GraphInfo) -> None:
        """Set up the initial state of the matching algorithm."""

        num_vertex = graph.num_vertex

        # Reference to the input graph.
        # The graph does not change while the algorithm runs.
        self.graph = graph

        # Each vertex is either single (unmatched) or matched to
        # another vertex.
        #
        # If vertex "x" is matched to vertex "y",
        # "vertex_mate[x] == y" and "vertex_mate[y] == x".
        #
        # If vertex "x" is unmatched, "vertex_mate[x] == -1".
        #
        # Initially all vertices are unmatched.
        self.vertex_mate: list[int] = num_vertex * [-1]

        # Each vertex is associated with a trivial blossom.
        # In addition, non-trivial blossoms may be created and destroyed
        # during the course of the matching algorithm.
        #
        # "trivial_blossom[x]" is the trivial blossom that contains only
        # vertex "x".
        self.trivial_blossom: list[_Blossom] = [_Blossom(x)
                                                for x in range(num_vertex)]

        # Non-trivial blossoms may be created and destroyed during
        # the course of the algorithm.
        #
        # Initially there are no non-trivial blossoms.
        self.nontrivial_blossom: list[_NonTrivialBlossom] = []

        # Every vertex is contained in exactly one top-level blossom
        # (possibly the trivial blossom that contains just that vertex).
        #
        # "vertex_top_blossom[x]" is the top-level blossom that contains
        # vertex "x".
        #
        # Initially all vertices are trivial top-level blossoms.
        self.vertex_top_blossom: list[_Blossom] = self.trivial_blossom.copy()

        # Every vertex has a variable in the dual LPP.
        #
        # "vertex_dual_2x[x]" is 2 times the dual variable of vertex "x".
        # Multiplication by 2 ensures that the values are integers
        # if all edge weights are integers.
        #
        # Vertex duals are initialized to half the maximum edge weight.
        max_weight = max(w for (_x, _y, w) in graph.edges)
        self.vertex_dual_2x: list[int|float] = num_vertex * [max_weight]

        # For each T-vertex or unlabeled vertex "x",
        # "vertex_best_edge[x]" is the edge index of the least-slack edge
        # between "x" and any S-vertex, or -1 if no such edge has been found.
        self.vertex_best_edge: list[int] = num_vertex * [-1]

        # "queue" is a list of S-vertices that must be scanned.
        # We call it a queue, but it is actually a stack.
        self.queue: list[int] = []

    def edge_slack_2x(self, e: int) -> int|float:
        """Return 2 times the slack of the edge with index "e".

        The result is only valid for edges that are not between vertices
        that belong to the same top-level blossom.

        Multiplication by 2 ensures that the return value is an integer
        if all edge weights are integers.

        This function is called O(m) times per stage.
        """
        (x, y, w) = self.graph.edges[e]
        assert self.vertex_top_blossom[x] is not self.vertex_top_blossom[y]
        return self.vertex_dual_2x[x] + self.vertex_dual_2x[y] - 2 * w

    #
    # Least-slack edge tracking:
    #
    # To calculate delta steps, the matching algorithm needs to find
    #  - the least-slack edge between any S-vertex and an unlabeled vertex;
    #  - the least-slack edge between any pair of top-level S-blossoms.
    #
    # For each unlabeled vertex and each T-vertex, we keep track of the
    # least-slack edge to any S-vertex. Tracking for unlabeled vertices
    # serves to provide the least-slack edge for the delta step.
    # Tracking for T-vertices is done because such vertices can turn into
    # unlabeled vertices if they are part of a T-blossom that gets expanded.
    #
    # For each top-level S-blossom, we keep track of the least-slack edge
    # to any S-vertex not in the same blossom.
    #
    # Furthermore, for each top-level S-blossom, we keep a list of least-slack
    # edges to other top-level S-blossoms. For any pair of top-level
    # S-blossoms, the least-slack edge between them is contained in the edge
    # list of at least one of the blossoms. An edge list may contain multiple
    # edges to the same S-blossom. Such redundant edges are pruned during
    # blossom merging to limit the number of tracked edges.
    #
    # Note: For a given vertex or blossom, the identity of the least-slack
    # edge to any S-blossom remains unchanged during a delta step.
    # Although the delta step changes edge slacks, it changes the slack
    # of every edge to an S-vertex by the same amount. Therefore the edge
    # that had least slack before the delta step, will still have least slack
    # after the delta step.
    #

    def lset_reset(self) -> None:
        """Reset least-slack edge tracking.

        This function takes time O(n).
        """
        num_vertex = self.graph.num_vertex

        for x in range(num_vertex):
            self.vertex_best_edge[x] = -1

        for blossom in self.trivial_blossom:
            blossom.best_edge = -1

        for blossom in self.nontrivial_blossom:
            blossom.best_edge = -1
            blossom.best_edge_set = None

    def lset_add_vertex_edge(self, y: int, e: int, slack: int|float) -> None:
        """Add edge "e" from an S-vertex to unlabeled vertex or T-vertex "y".

        This function takes time O(1) per call.
        This function is called O(m) times per stage.
        """
        best_edge = self.vertex_best_edge[y]
        if best_edge == -1:
            self.vertex_best_edge[y] = e
        else:
            best_slack = self.edge_slack_2x(best_edge)
            if slack < best_slack:
                self.vertex_best_edge[y] = e

    def lset_get_best_vertex_edge(self) -> tuple[int, int|float]:
        """Return the index and slack of the least-slack edge between
        any S-vertex and unlabeled vertex.

        This function takes time O(n) per call.
        This function takes total time O(n**2) per stage.

        Returns:
            Tuple (edge_index, slack_2x) if there is a least-slack edge,
            or (-1, 0) if there is no suitable edge.
        """
        best_index = -1
        best_slack: int|float = 0

        for x in range(self.graph.num_vertex):
            if self.vertex_top_blossom[x].label == _LABEL_NONE:
                e = self.vertex_best_edge[x]
                if e != -1:
                    slack = self.edge_slack_2x(e)
                    if (best_index == -1) or (slack < best_slack):
                        best_index = e
                        best_slack = slack

        return (best_index, best_slack)

    def lset_new_blossom(self, blossom: _Blossom) -> None:
        """Start tracking edges for a new S-blossom."""
        assert blossom.best_edge == -1
        if isinstance(blossom, _NonTrivialBlossom):
            assert blossom.best_edge_set is None
            blossom.best_edge_set = []

    def lset_add_blossom_edge(
            self,
            blossom: _Blossom,
            e: int,
            slack: int|float
            ) -> None:
        """Add edge "e" between the specified S-blossom and another S-blossom.

        This function takes time O(1) per call.
        This function is called O(m) times per stage.
        """
        # Track least-slack edge between this blossom and any other S-blossom.
        if blossom.best_edge == -1:
            blossom.best_edge = e
        else:
            best_slack = self.edge_slack_2x(blossom.best_edge)
            if slack < best_slack:
                blossom.best_edge = e

        # Regardless of whether this edge is currently the least-slack edge,
        # this edge may later become the least-slack edge if other edges
        # become unavailable when S-blossoms are merged.
        #
        # We therefore add the edge to a list of potential future least-slack
        # edges for this blossom. We do this only for non-trivial blossoms.
        if isinstance(blossom, _NonTrivialBlossom):
            assert blossom.best_edge_set is not None
            blossom.best_edge_set.append(e)

    def lset_merge_blossoms(self, blossom: _NonTrivialBlossom) -> None:
        """Update least-slack edge tracking after merging sub-blossoms
        into a new S-blossom.

        This function takes time O(n) per call.
        This function takes total time O(n**2) per stage.
        """
        num_vertex = self.graph.num_vertex

        # Calculate the set of least-slack edges to other S-blossoms.
        # We basically merge the edge lists from all sub-blossoms, but reject
        # edges that are internal to this blossom, and trim the set such that
        # there is at most one edge to each external S-blossom.
        #
        # Sub-blossoms that were formerly labeled T can be ignored; their
        # vertices are in the queue and will discover neighbouring S-blossoms
        # via the edge scan process.
        #
        # Build a temporary array holding the least-slack edge index to
        # each top-level S-blossom. This array is indexed by the base vertex
        # of the blossoms.
# TODO : consider using pre-allocated arrays for this purpose
        best_edge_to_blossom: list[int] = num_vertex * [-1]
        zero_slack: int|float = 0
        best_slack_to_blossom: list[int|float] = num_vertex * [zero_slack]

        # And find the overall least-slack edge to any other S-blossom.
        best_edge = -1
        best_slack: int|float = 0

        # Add the least-slack edges of every S-sub-blossom.
        for sub in blossom.subblossoms:

            if sub.label != _LABEL_S:
                continue

            if isinstance(sub, _NonTrivialBlossom):
                # Pull the edge list from the sub-blossom.
                assert sub.best_edge_set is not None
                sub_edge_set = sub.best_edge_set
                # Delete the edge list from the sub-blossom.
                sub.best_edge_set = None
            else:
                # Trivial blossoms don't have a list of least-slack edges,
                # so we just look at all adjacent edges. This happens at most
                # once per vertex per stage.
                # It adds up to O(m) time per stage.
                sub_edge_set = self.graph.adjacent_edges[sub.base_vertex]

            # Add edges to the temporary array.
            for e in sub_edge_set:
                (x, y, _w) = self.graph.edges[e]
                bx = self.vertex_top_blossom[x]
                by = self.vertex_top_blossom[y]
                assert (bx is blossom) or (by is blossom)

                # Reject internal edges in this blossom.
                if bx is by:
                    continue

                # Set bx = blossom at the other end of this edge.
                bx = by if (bx is blossom) else bx

                # Reject edges that don't link to an S-blossom.
                if bx.label != _LABEL_S:
                    continue

                # Keep only the least-slack edge to "bx".
                slack = self.edge_slack_2x(e)
                bx_base = bx.base_vertex
                if ((best_edge_to_blossom[bx_base] == -1)
                        or (slack < best_slack_to_blossom[bx_base])):
                    best_edge_to_blossom[bx_base] = e
                    best_slack_to_blossom[bx_base] = slack

                # Update the overall least-slack edge to any S-blossom.
                if (best_edge == -1) or (slack < best_slack):
                    best_edge = e
                    best_slack = slack

        # Extract a compact list of least-slack edge indices.
        # We can not keep the temporary array because that would blow up
        # memory use to O(n**2).
        best_edge_set = [e for e in best_edge_to_blossom if e != -1]
        blossom.best_edge_set = best_edge_set

        # Keep the overall least-slack edge.
        blossom.best_edge = best_edge

    def lset_get_best_blossom_edge(self) -> tuple[int, int|float]:
        """Return the index and slack of the least-slack edge between
        any pair of top-level S-blossoms.

        This function takes time O(n) per call.
        This function takes total time O(n**2) per stage.

        Returns:
            Tuple (edge_index, slack_2x) if there is a least-slack edge,
            or (-1, 0) if there is no suitable edge.
        """
        best_index = -1
        best_slack: int|float = 0

# TODO : do we really want to split trivial/nontrivial blossoms ?
        for blossom in self.trivial_blossom + self.nontrivial_blossom:
            if (blossom.label == _LABEL_S) and (blossom.parent is None):
                e = blossom.best_edge
                if e != -1:
                    slack = self.edge_slack_2x(e)
                    if (best_index == -1) or (slack < best_slack):
                        best_index = e
                        best_slack = slack

        return (best_index, best_slack)

    #
    # General support routines:
    #

    def reset_stage(self) -> None:
        """Reset data which are only valid during a stage.

        Marks all blossoms as unlabeled, clears the queue,
        and resets tracking of least-slack edges.
        """

        # Remove blossom labels.
        for blossom in self.trivial_blossom + self.nontrivial_blossom:
            blossom.label = _LABEL_NONE
            blossom.tree_edge = None

        # Clear the queue.
        self.queue.clear()

        # Reset least-slack edge tracking.
        self.lset_reset()

    def trace_alternating_paths(self, x: int, y: int) -> _AlternatingPath:
        """Trace back through the alternating trees from vertices "x" and "y".

        If both vertices are part of the same alternating tree, this function
        discovers a new blossom. In this case it returns an alternating path
        through the blossom that starts and ends in the same sub-blossom.

        If the vertices are part of different alternating trees, this function
        discovers an augmenting path. In this case it returns an alternating
        path that starts and ends in an unmatched vertex.

        This function takes time O(k) to discover a blossom, where "k" is the
        number of sub-blossoms, or time O(n) to discover an augmenting path.

        Returns:
            Alternating path as an ordered list of edges between top-level
            blossoms.
        """

        marked_blossoms: list[_Blossom] = []

        # "xedges" is a list of edges used while tracing from "x".
        # "yedges" is a list of edges used while tracing from "y".
        # Pre-load the edge (x, y) on both lists.
        xedges: list[tuple[int, int]] = [(x, y)]
        yedges: list[tuple[int, int]] = [(y, x)]

        # "first_common" is the first common ancestor of "x" and "y"
        # in the alternating tree, or None if there is no common ancestor.
        first_common: Optional[_Blossom] = None

        # Alternate between tracing the path from "x" and the path from "y".
        # This ensures that the search time is bounded by the size of the
        # newly found blossom.
        while x != -1 or y != -1:

            # Check if we found a common ancestor.
            bx = self.vertex_top_blossom[x]
            if bx.marker:
                first_common = bx
                break

            # Mark blossom as a potential common ancestor.
            bx.marker = True
            marked_blossoms.append(bx)

            # Track back through the link in the alternating tree.
            if bx.tree_edge is None:
                # Reached the root of this alternating tree.
                x = -1
            else:
                xedges.append(bx.tree_edge)
                x = bx.tree_edge[0]

            # Swap "x" and "y" to alternate between paths.
            if y != -1:
                (x, y) = (y, x)
                (xedges, yedges) = (yedges, xedges)

        # Remove all markers we placed.
        for b in marked_blossoms:
            b.marker = False

        # If we found a common ancestor, trim the paths so they end there.
        if first_common is not None:
            assert self.vertex_top_blossom[xedges[-1][0]] is first_common
            while self.vertex_top_blossom[yedges[-1][0]] is not first_common:
                yedges.pop()

        # Fuse the two paths.
        # Flip the order of one path, and flip the edge tuples in the other
        # path to obtain a continuous path with correctly ordered edge tuples.
        # Skip the duplicate edge in one of the paths.
        path_edges = xedges[::-1] + [(y, x) for (x, y) in yedges[1:]]

        # Any S-to-S alternating path must have odd length.
        assert len(path_edges) % 2 == 1

        return _AlternatingPath(path_edges)

    #
    # Merge and expand blossoms:
    #

    def make_blossom(self, path: _AlternatingPath) -> None:
        """Create a new blossom from an alternating cycle.

        Assign label S to the new blossom.
        Relabel all T-sub-blossoms as S and add their vertices to the queue.

        This function takes time O(n) per call.
        This function takes total time O(n**2) per stage.
        """

        # Check that the path is odd-length.
        assert len(path.edges) % 2 == 1
        assert len(path.edges) >= 3

        # Construct the list of sub-blossoms (current top-level blossoms).
        subblossoms = [self.vertex_top_blossom[x] for (x, y) in path.edges]

        # Check that the path is cyclic.
        # Note the path may not start and end with the same _vertex_,
        # but it must start and end in the same _blossom_.
        subblossoms_next = [self.vertex_top_blossom[y]
                            for (x, y) in path.edges]
        assert subblossoms[0] == subblossoms_next[-1]
        assert subblossoms[1:] == subblossoms_next[:-1]

        # Create the new blossom object.
        blossom = _NonTrivialBlossom(subblossoms, path.edges)

        # Insert into the blossom array.
# TODO : rework this
        self.nontrivial_blossom.append(blossom)

        # Link the subblossoms to the their new parent.
        for sub in subblossoms:
            sub.parent = blossom

        # Update blossom-membership of all vertices in the new blossom.
        # NOTE: This step takes O(n) time per blossom formation, and adds up
        #       to O(n**2) total time per stage.
        #       This could be improved through a union-find datastructure, or
        #       by re-using the blossom index of the largest sub-blossom.
        for x in blossom.vertices():
            self.vertex_top_blossom[x] = blossom

        # Assign label S to the new blossom.
        assert subblossoms[0].label == _LABEL_S
        blossom.label = _LABEL_S
        blossom.tree_edge = subblossoms[0].tree_edge

        # Former T-vertices which are part of this blossom now become
        # S-vertices. Add them to the queue.
        for sub in subblossoms:
            if sub.label == _LABEL_T:
                self.queue.extend(sub.vertices())

        # Merge least-slack edges for the S-sub-blossoms.
        self.lset_merge_blossoms(blossom)

    def find_path_through_blossom(
            self,
            blossom: _NonTrivialBlossom,
            sub: _Blossom
            ) -> tuple[list[_Blossom], list[tuple[int, int]]]:
        """Construct a path through the specified blossom,
        from sub-blossom "sub" to the base of the blossom.

        Return:
            Tuple (nodes, edges).
        """

        nodes: list[_Blossom] = [sub]
        edges: list[tuple[int, int]] = []

        # Walk around the blossom from "sub" to its base.
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

    def expand_t_blossom(self, blossom: _NonTrivialBlossom) -> None:
        """Expand the specified T-blossom.

        This function takes time O(n).
        """

        assert blossom.parent is None
        assert blossom.label == _LABEL_T

        # Convert sub-blossoms into top-level blossoms.
        for sub in blossom.subblossoms:
            assert sub.label == _LABEL_NONE
            sub.parent = None
            if isinstance(sub, _NonTrivialBlossom):
                for x in sub.vertices():
                    self.vertex_top_blossom[x] = sub
            else:
                self.vertex_top_blossom[sub.base_vertex] = sub

        # The expanding blossom was part of an alternating tree, linked to
        # a parent node in the tree via one of its subblossoms, and linked to
        # a child node of the tree via the base vertex.
        # We must reconstruct this part of the alternating tree, which will
        # now run via sub-blossoms of the expanded blossom.

        # Find the sub-blossom that is attached to the parent node in
        # the alternating tree.
# TODO : uglyness with the assertion
        entry_link = blossom.tree_edge
        assert entry_link is not None
        (x, y) = entry_link
        sub = self.vertex_top_blossom[y]

        # Assign label T to that sub-blossom.
        sub.label = _LABEL_T
        sub.tree_edge = blossom.tree_edge

        # Walk through the expanded blossom from "sub" to the base vertex.
        # Assign alternating S and T labels to the sub-blossoms and attach
        # them to the alternating tree.
        (path_nodes, path_edges) = self.find_path_through_blossom(blossom,
                                                                  sub)

        for p in range(0, len(path_edges), 2):
            #
            #   (p) ==(y,x)== (p+1) ----- (p+2)
            #    T              S           T
            #
            # path_nodes[p] has already been labeled T.
            # We now assign labels to path_nodes[p+1] and path_nodes[p+2].

            # Assign label S to path_nodes[p+1].
            (y, x) = path_edges[p]
            self.assign_label_s(x)

            # Assign label T to path_nodes[i+2] and attach it
            # to path_nodes[p+1].
            sub = path_nodes[p+2]
            sub.label = _LABEL_T
            sub.tree_edge = path_edges[p+1]

        # Delete the expanded blossom.
# TODO : avoid O(n) here if possible
        self.nontrivial_blossom.remove(blossom)

    def expand_blossom_rec(
            self,
            blossom: _NonTrivialBlossom,
            stack: list[_NonTrivialBlossom]
            ) -> None:
        """Expand the specified blossom and recursively expand any
        sub-blossoms that have dual variable zero.

        Use the stack object instead of making direct recursive calls.
        """

        assert blossom.parent is None

        # Examine sub-blossoms.
        for sub in blossom.subblossoms:

            # Mark the sub-blossom as a top-level blossom.
            sub.parent = None

            if isinstance(sub, _NonTrivialBlossom):
                # Non-trivial sub-blossom.
                # If its dual variable is zero, expand it recursively.
                if sub.dual_var == 0:
                    stack.append(sub)
                else:
                    # This sub-blossom will not be expanded;
                    # it now becomes top-level. Update its vertices
                    # to point to this sub-blossom.
                    for x in sub.vertices():
                        self.vertex_top_blossom[x] = sub
            else:
                # Trivial sub-blossom. Mark it as top-level vertex.
                self.vertex_top_blossom[sub.base_vertex] = sub

        # Delete the expanded blossom.
# TODO : avoid O(n) here if possible
        self.nontrivial_blossom.remove(blossom)

    def expand_zero_dual_blossoms(self) -> None:
        """Expand all blossoms with zero dual variable (recursively).

        Note that this function runs at the end of a stage.
        Blossoms are not labeled. Least-slack edges are not tracked.

        This function takes time O(n).
        """

        # Use an explicit stack to avoid deep recursion.
        # The stack contains a list of blossoms to be expanded.
        stack: list[_NonTrivialBlossom] = []

        # Find top-level blossoms with zero slack.
        for blossom in self.nontrivial_blossom:
            if blossom.parent is None:
                # We typically expand only S-blossoms that were created after
                # the most recent delta step. Those blossoms have _exactly_
                # zero dual. So this comparison is reliable, even in case
                # of floating point edge weights.
                if blossom.dual_var == 0:
                    stack.append(blossom)

        # Expand blossoms.
        while stack:
            blossom = stack.pop()
            self.expand_blossom_rec(blossom, stack)

    #
    # Augmenting:
    #

    def augment_blossom_rec(
            self,
            blossom: _NonTrivialBlossom,
            sub: _Blossom,
            stack: list[tuple[_NonTrivialBlossom, _Blossom]]
            ) -> None:
        """Augment along an alternating path through the specified blossom,
        from sub-blossom "sub" to the base vertex of the blossom.

        Modify the blossom to reflect that sub-blossom "sub" contains
        the base vertex after augmenting.

        Mark any sub-blossoms on the alternating path for recursive
        augmentation, except for sub-blossom "sub" which has already been
        augmented. Use the stack instead of making direct recursive calls.
        """

        # Walk through the blossom from "sub" to the base vertex.
        (path_nodes, path_edges) = self.find_path_through_blossom(blossom,
                                                                  sub)

        for p in range(0, len(path_edges), 2):
            # Before augmentation:
            #   path_nodes[p] is matched to path_nodes[p+1]
            #
            #   (p) ===== (p+1) ---(x,y)--- (p+2)
            #
            # After augmentation:
            #   path_nodes[p+1] matched to path_nodes[p+2] via edge (i,j)
            #
            #   (p) ----- (p+1) ===(x,y)=== (p+2)
            #

            # Pull the edge (x, y) into the matching.
            (x, y) = path_edges[p+1]
            self.vertex_mate[x] = y
            self.vertex_mate[y] = x

            # Augment through the subblossoms touching the edge (x, y).
            # Nothing needs to be done for trivial subblossoms.
            bx = path_nodes[p+1]
            if isinstance(bx, _NonTrivialBlossom):
                stack.append((bx, self.trivial_blossom[x]))

            by = path_nodes[p+2]
            if isinstance(by, _NonTrivialBlossom):
                stack.append((by, self.trivial_blossom[y]))

        # Rotate the subblossom list so the new base ends up in position 0.
        p = blossom.subblossoms.index(sub)
        blossom.subblossoms = (
            blossom.subblossoms[p:] + blossom.subblossoms[:p])
        blossom.edges = blossom.edges[p:] + blossom.edges[:p]

        # Update the base vertex.
        # We can pull this from the sub-blossom where we started since
        # its augmentation has already finished.
        blossom.base_vertex = sub.base_vertex

    def augment_blossom(
            self,
            blossom: _NonTrivialBlossom,
            sub: _Blossom
            ) -> None:
        """Augment along an alternating path through the specified blossom,
        from sub-blossom "sub" to the base vertex of the blossom.

        Recursively augment any sub-blossoms on the alternating path.

        This function takes time O(n).
        """

        # Use an explicit stack to avoid deep recursion.
        stack = [(blossom, sub)]

        while stack:
            (outer_blossom, sub) = stack.pop()
            assert sub.parent is not None
            blossom = sub.parent

            if blossom != outer_blossom:
                # Sub-blossom "sub" is an indirect (nested) child of
                # the "outer_blossom" we are supposed to be augmenting.
                #
                # "blossom" is the direct parent of "sub".
                # Let's first augment "blossom" from "sub" to its base vertex.
                # Then continue by augmenting the parent of "blossom",
                # from "blossom" to its base vertex, and so on until we
                # get to the "outer_blossom".
                #
                # Set up to continue augmenting through the parent of
                # "blossom".
                stack.append((outer_blossom, blossom))

            # Augment "blossom" from "sub" to the base vertex.
            self.augment_blossom_rec(blossom, sub, stack)

    def augment_matching(self, path: _AlternatingPath) -> None:
        """Augment the matching through the specified augmenting path.

        This function takes time O(n).
        """

        # Check that the augmenting path starts and ends in
        # an unmatched vertex or a blossom with unmatched base.
        assert len(path.edges) % 2 == 1
        for x in (path.edges[0][0], path.edges[-1][1]):
            b = self.vertex_top_blossom[x]
            assert self.vertex_mate[b.base_vertex] == -1

        # The augmenting path looks like this:
        #
        #   (unmatched) ---- (B) ==== (B) ---- (B) ==== (B) ---- (unmatched)
        #
        # The first and last vertex (or blossom) of the path are unmatched
        # (or have unmatched base vertex). After augmenting, those vertices
        # will be matched. All matched edges on the path become unmatched,
        # and unmatched edges become matched.
        #
        # This loop walks along the edges of this path that were not matched
        # before augmenting.
        for (x, y) in path.edges[0::2]:

            # Augment the non-trivial blossoms on either side of this edge.
            # No action is necessary for trivial blossoms.
            bx = self.vertex_top_blossom[x]
            if isinstance(bx, _NonTrivialBlossom):
                self.augment_blossom(bx, self.trivial_blossom[x])

            by = self.vertex_top_blossom[y]
            if isinstance(by, _NonTrivialBlossom):
                self.augment_blossom(by, self.trivial_blossom[y])

            # Pull the edge into the matching.
            self.vertex_mate[x] = y
            self.vertex_mate[y] = x

    #
    # Labeling and alternating tree expansion:
    #

    def assign_label_s(self, x: int) -> None:
        """Assign label S to the unlabeled blossom that contains vertex "x".

        If vertex "x" is matched, it is attached to the alternating tree
        via its matched edge. If vertex "x" is unmatched, it becomes the root
        of an alternating tree.

        All vertices in the newly labeled blossom are added to the scan queue.

        Precondition:
            "x" is an unlabeled vertex, either unmatched or matched to
            a T-vertex via a tight edge.
        """

        # Assign label S to the blossom that contains vertex "x".
        bx = self.vertex_top_blossom[x]
        assert bx.label == _LABEL_NONE
        bx.label = _LABEL_S

        y = self.vertex_mate[x]
        if y == -1:
            # Vertex "x" is unmatched.
            # It must be either a top-level vertex or the base vertex of
            # a top-level blossom.
            assert bx.base_vertex == x

            # Mark the blossom as root of an alternating tree.
            bx.tree_edge = None

        else:
            # Vertex "x" is matched to T-vertex "y".
            by = self.vertex_top_blossom[y]
            assert by.label == _LABEL_T

            # Attach the blossom that contains "x" to the alternating tree.
            bx.tree_edge = (y, x)

        # Start least-slack edge tracking for the S-blossom.
        self.lset_new_blossom(bx)

        # Add all vertices inside the newly labeled S-blossom to the queue.
        self.queue.extend(bx.vertices())

    def assign_label_t(self, x: int, y: int) -> None:
        """Assign label T to the unlabeled blossom that contains vertex "y".

        Attach it to the alternating tree via edge (x, y).
        Then immediately assign label S to the mate of vertex "y".

        Preconditions:
         - "x" is an S-vertex.
         - "y" is an unlabeled, matched vertex.
         - There is a tight edge between vertices "x" and "y".
        """
        assert self.vertex_top_blossom[x].label == _LABEL_S

        # Assign label T to the unlabeled blossom.
        by = self.vertex_top_blossom[y]
        assert by.label == _LABEL_NONE
        by.label = _LABEL_T
        by.tree_edge = (x, y)

        # Assign label S to the blossom that contains the mate of vertex "y".
        z = self.vertex_mate[by.base_vertex]
        assert z != -1
        self.assign_label_s(z)

    def add_s_to_s_edge(self, x: int, y: int) -> Optional[_AlternatingPath]:
        """Add the edge between S-vertices "x" and "y".

        If the edge connects blossoms that are part of the same alternating
        tree, this function creates a new S-blossom and returns None.

        If the edge connects two different alternating trees, an augmenting
        path has been discovered. In this case the function changes nothing
        and returns the augmenting path.

        Returns:
            Augmenting path if found; otherwise None.
        """

        # Trace back through the alternating trees from "x" and "y".
        path = self.trace_alternating_paths(x, y)

        # If the path is a cycle, create a new blossom.
        # Otherwise the path is an augmenting path.
        # Note that an alternating starts and ends in the same blossom,
        # but not necessarily in the same vertex within that blossom.
        p = path.edges[0][0]
        q = path.edges[-1][1]
        if self.vertex_top_blossom[p] is self.vertex_top_blossom[q]:
            self.make_blossom(path)
            return None
        else:
            return path

    def substage_scan(self) -> Optional[_AlternatingPath]:
        """Scan queued S-vertices to expand the alternating trees.

        The scan proceeds until either an augmenting path is found,
        or the queue of S-vertices becomes empty.

        New blossoms may be created during the scan.

        Returns:
            Augmenting path if found; otherwise None.
        """

        edges = self.graph.edges
        adjacent_edges = self.graph.adjacent_edges

        # Process S-vertices waiting to be scanned.
        # This loop runs through O(n) iterations per stage.
        while self.queue:

            # Take a vertex from the queue.
            x = self.queue.pop()

            # Double-check that "x" is an S-vertex.
            bx = self.vertex_top_blossom[x]
            assert bx.label == _LABEL_S

            # Scan the edges that are incident on "x".
            # This loop runs through O(m) iterations per stage.
            for e in adjacent_edges[x]:
                (p, q, _w) = edges[e]
                y = p if p != x else q

                # Consider the edge between vertices "x" and "y".
                # Try to pull this edge into an alternating tree.

                # Note: blossom index of vertex "x" may change during
                # this loop, so we need to refresh it here.
                bx = self.vertex_top_blossom[x]
                by = self.vertex_top_blossom[y]

                # Ignore edges that are internal to a blossom.
                if bx is by:
                    continue

                ylabel = by.label

                # Check whether this edge is tight (has zero slack).
                # Only tight edges may be part of an alternating tree.
                slack = self.edge_slack_2x(e)
                if slack <= 0:
                    if ylabel == _LABEL_NONE:
                        # Assign label T to the blossom that contains "y".
                        self.assign_label_t(x, y)
                    elif ylabel == _LABEL_S:
                        # This edge connects two S-blossoms. Use it to find
                        # either a new blossom or an augmenting path.
                        alternating_path = self.add_s_to_s_edge(x, y)
                        if alternating_path is not None:
                            return alternating_path

                elif ylabel == _LABEL_S:
                    # Update tracking of least-slack edges between S-blossoms.
                    self.lset_add_blossom_edge(bx, e, slack)

                if ylabel != _LABEL_S:
                    # Update tracking of least-slack edges from vertex "y" to
                    # any S-vertex. We do this for T-vertices and unlabeled
                    # vertices. Edges which already have zero slack are still
                    # tracked.
                    self.lset_add_vertex_edge(y, e, slack)

        # No further S vertices to scan, and no augmenting path found.
        return None

    #
    # Delta steps:
    #

    def substage_calc_dual_delta(
            self
            ) -> tuple[int, float|int, int, Optional[_NonTrivialBlossom]]:
        """Calculate a delta step in the dual LPP problem.

        This function returns the minimum of the 4 types of delta values,
        and the type of delta which obtain the minimum, and the edge or
        blossom that produces the minimum delta, if applicable.

        The returned value is 2 times the actual delta value.
        Multiplication by 2 ensures that the result is an integer if all edge
        weights are integers.

        This function assumes that there is at least one S-vertex.
        This function takes time O(n).

        Returns:
            Tuple (delta_type, delta_2x, delta_edge, delta_blossom).
        """
        num_vertex = self.graph.num_vertex

        delta_edge = -1
        delta_blossom: Optional[_NonTrivialBlossom] = None

        # Compute delta1: minimum dual variable of any S-vertex.
        delta_type = 1
        delta_2x = min(
            self.vertex_dual_2x[x]
            for x in range(num_vertex)
            if self.vertex_top_blossom[x].label == _LABEL_S)

        # Compute delta2: minimum slack of any edge between an S-vertex and
        # an unlabeled vertex.
        (e, slack) = self.lset_get_best_vertex_edge()
        if (e != -1) and (slack <= delta_2x):
            delta_type = 2
            delta_2x = slack
            delta_edge = e

        # Compute delta3: half minimum slack of any edge between two top-level
        # S-blossoms.
        (e, slack) = self.lset_get_best_blossom_edge()
        if e != -1:
            if self.graph.integer_weights:
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

        # Compute delta4: half minimum dual variable of a top-level T-blossom.
        for blossom in self.nontrivial_blossom:
            if (blossom.label == _LABEL_T) and (blossom.parent is None):
                if blossom.dual_var <= delta_2x:
                    delta_type = 4
                    delta_2x = blossom.dual_var
                    delta_blossom = blossom

        return (delta_type, delta_2x, delta_edge, delta_blossom)

    def substage_apply_delta_step(self, delta_2x: int|float) -> None:
        """Apply a delta step to the dual LPP variables."""

        num_vertex = self.graph.num_vertex

        # Apply delta to dual variables of all vertices.
        for x in range(num_vertex):
            xlabel = self.vertex_top_blossom[x].label
            if xlabel == _LABEL_S:
                # S-vertex: subtract delta from dual variable.
                self.vertex_dual_2x[x] -= delta_2x
            elif xlabel == _LABEL_T:
                # T-vertex: add delta to dual variable.
                self.vertex_dual_2x[x] += delta_2x

        # Apply delta to dual variables of top-level non-trivial blossoms.
        for blossom in self.nontrivial_blossom:
            if blossom.parent is None:
                blabel = blossom.label
                if blabel == _LABEL_S:
                    # S-blossom: add 2*delta to dual variable.
                    blossom.dual_var += delta_2x
                elif blabel == _LABEL_T:
                    # T-blossom: subtract 2*delta from dual variable.
                    blossom.dual_var -= delta_2x

    #
    # Main stage function:
    #

    def run_stage(self) -> bool:
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

        num_vertex = self.graph.num_vertex

        # Assign label S to all unmatched vertices and put them in the queue.
        for x in range(num_vertex):
            if self.vertex_mate[x] == -1:
                self.assign_label_s(x)

        # Stop if all vertices are matched.
        # No further improvement is possible in that case.
        # This avoids messy calculations of delta steps without any S-vertex.
        if not self.queue:
            return False

        # Each pass through the following loop is a "substage".
        # The substage tries to find an augmenting path.
        # If an augmenting path is found, we augment the matching and end
        # the stage. Otherwise we update the dual LPP problem and enter the
        # next substage, or stop if no further improvement is possible.
        #
        # This loop runs through at most O(n) iterations per stage.
        augmenting_path = None
        while True:

            # Expand alternating trees.
            # End the stage if an augmenting path is found.
            augmenting_path = self.substage_scan()
            if augmenting_path is not None:
                break

            # Calculate delta step in the dual LPP problem.
            (delta_type, delta_2x, delta_edge, delta_blossom
                ) = self.substage_calc_dual_delta()

            # Apply the delta step to the dual variables.
            self.substage_apply_delta_step(delta_2x)

            if delta_type == 2:
                # Use the edge from S-vertex to unlabeled vertex that got
                # unlocked through the delta update.
                (x, y, _w) = self.graph.edges[delta_edge]
                if self.vertex_top_blossom[x].label != _LABEL_S:
                    (x, y) = (y, x)
                self.assign_label_t(x, y)

            elif delta_type == 3:
                # Use the S-to-S edge that got unlocked by the delta update.
                # This may reveal an augmenting path.
                (x, y, _w) = self.graph.edges[delta_edge]
                augmenting_path = self.add_s_to_s_edge(x, y)
                if augmenting_path is not None:
                    break

            elif delta_type == 4:
                # Expand the T-blossom that reached dual value 0 through
                # the delta update.
                assert delta_blossom is not None
                self.expand_t_blossom(delta_blossom)

            else:
                # No further improvement possible. End the stage.
                assert delta_type == 1
                break

        # Augment the matching if an augmenting path was found.
        if augmenting_path is not None:
            self.augment_matching(augmenting_path)

        # Expand all blossoms with dual variable zero.
        # These are typically S-blossoms, since T-blossoms normally
        # get expanded as soon as their dual variable hits zero.
        self.expand_zero_dual_blossoms()

        # Remove all labels, clear queue.
        self.reset_stage()

        # Return True if the matching was augmented.
        return (augmenting_path is not None)


# TODO : clean up this whole mess

def _verify_optimum(ctx: _MatchingContext) -> None:
    """Verify that the optimum solution has been found.

    This function takes time O(m * n).
TODO : really ??

    Raises:
        AssertionError: If the solution is not optimal.
    """

    num_vertex = ctx.graph.num_vertex

    vertex_mate = ctx.vertex_mate
    vertex_dual_var_2x = ctx.vertex_dual_2x

    # Double-check that each matching edge actually exists in the graph.
    num_matched_vertex = 0
    for x in range(num_vertex):
        if vertex_mate[x] != -1:
            num_matched_vertex += 1

    num_matched_edge = 0
    for (x, y, _w) in ctx.graph.edges:
        if vertex_mate[x] == y:
            num_matched_edge += 1

    assert num_matched_vertex == 2 * num_matched_edge

    # Check that all dual variables are non-negative.
    assert min(vertex_dual_var_2x) >= 0
    for blossom in ctx.nontrivial_blossom:
        assert blossom.dual_var >= 0

    # Count the number of vertices in each blossom.
    blossom_nvertex = {id(blossom): 0 for blossom in ctx.nontrivial_blossom}
    for x in range(num_vertex):
        b = ctx.trivial_blossom[x]
        while b.parent is not None:
            b = b.parent
            blossom_nvertex[id(b)] += 1

    # Calculate slack of each edge.
    # Also count the number of matched edges in each blossom.
    blossom_nmatched = {id(blossom): 0 for blossom in ctx.nontrivial_blossom}

    for (x, y, w) in ctx.graph.edges:

        # List blossoms that contain vertex "x".
        xblossoms = []
        bx = ctx.trivial_blossom[x]
        while bx.parent is not None:
            bx = bx.parent
            xblossoms.append(bx)

        # List blossoms that contain vertex "y".
        yblossoms = []
        by = ctx.trivial_blossom[y]
        while by.parent is not None:
            by = by.parent
            yblossoms.append(by)

        # List blossoms that contain the edge (x, y).
        edge_blossoms: list[_NonTrivialBlossom] = []
        for (bx, by) in zip(reversed(xblossoms), reversed(yblossoms)):
            if bx is not by:
                break
            edge_blossoms.append(bx)

        # Calculate edge slack =
        #   dual[x] + dual[y] - weight
        #     + sum(blossom.dual_var for "blossom" containing the edge)
        #
        # Multiply weights by 2 to ensure integer values.
        slack = vertex_dual_var_2x[x] + vertex_dual_var_2x[y] - 2 * w
        slack += 2 * sum(blossom.dual_var for blossom in edge_blossoms)

        # Check that all edges have non-negative slack.
        assert slack >= 0

        # Check that all matched edges have zero slack.
        if vertex_mate[x] == y:
            assert slack == 0

        # Update number of matched edges in each blossom.
        if vertex_mate[x] == y:
            for b in edge_blossoms:
                blossom_nmatched[id(b)] += 1

    # Check that all unmatched vertices have zero dual.
    for x in range(num_vertex):
        if vertex_mate[x] == -1:
            assert vertex_dual_var_2x[x] == 0

    # Check that all blossoms with positive dual are "full".
    # A blossom is full if all except one of its vertices are matched
    # to another vertex in the same blossom.
    for blossom in ctx.nontrivial_blossom:
        if blossom.dual_var > 0:
            assert blossom_nvertex[id(blossom)] == 2 * blossom_nmatched[id(blossom)] + 1

    # Optimum solution confirmed.
