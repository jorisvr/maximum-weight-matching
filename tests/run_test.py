#!/usr/bin/env python3

"""
Run matching solvers on test graphs.
"""

from __future__ import annotations

import sys
import argparse
import enum
import io
import math
import os.path
import random
import shlex
import subprocess
import time
from typing import NamedTuple, Optional, Sequence, TextIO


class InvalidMatchingError(Exception):
    """Raised when the solver output is not a valid matching in the graph."""


class SolverError(Exception):
    """Raised if the solver returns an error or outputs an invalid format."""


class Graph(NamedTuple):
    """Represents a graph. Vertex indices start from 0."""
    edges: list[tuple[int, int, int|float]]

    def num_vertex(self) -> int:
        """Count number of vertices."""
        if self.edges:
            return 1 + max(max(x, y) for (x, y, _w) in self.edges)
        else:
            return 0


class Matching(NamedTuple):
    """Represents a matching."""
    pairs: list[tuple[int, int]]


class RunStatus(enum.IntEnum):
    """Result categories for running a solver."""
    OK = 0
    FAILED = 1
    WRONG_ANSWER = 2
    TIMEOUT = 3


class RunResult(NamedTuple):
    """Represent the result of running a solver on a graph."""
    status: RunStatus = RunStatus.OK
    weight: int|float = 0
    run_time: Sequence[float] = ()


def parse_int_or_float(s: str) -> int|float:
    """Convert a string to integer or float value."""
    try:
        return int(s)
    except ValueError:
        pass
    return float(s)


def read_dimacs_graph(f: TextIO) -> Graph:
    """Read a graph in DIMACS edge list format."""

    edges: list[tuple[int, int, float]] = []

    for line in f:
        s = line.strip()
        words = s.split()

        if not words[0]:
            # Skip empty line.
            pass

        elif words[0].startswith("c"):
            # Skip comment line.
            pass

        elif words[0] == "p":
            # Handle "problem" line.
            if len(words) != 4:
                raise ValueError(
                    f"Expecting DIMACS edge format but got {s!r}")
            if words[1] != "edge":
                raise ValueError(
                    f"Expecting DIMACS edge format but got {words[1]!r}")

        elif words[0] == "e":
            # Handle "edge" line.
            if len(words) != 4:
                raise ValueError(f"Expecting edge but got {s!r}")
            x = int(words[1])
            y = int(words[2])
            w = parse_int_or_float(words[3])
            if x < 1:
                raise ValueError("Invalid vertex index {x}")
            if y < 1:
                raise ValueError("Invalid vertex index {y}")
            edges.append((x - 1, y - 1, w))

        else:
            raise ValueError(f"Unknown line type {words[0]!r}")

    return Graph(edges)


def write_dimacs_graph(f: TextIO, graph: Graph) -> None:
    """Write a graph in DIMACS edge list format."""

    num_vertex = graph.num_vertex()
    num_edge = len(graph.edges)

    print(f"p edge {num_vertex} {num_edge}", file=f)

    integer_weights = all(isinstance(w, int) for (_x, _y, w) in graph.edges)

    for (x, y, w) in graph.edges:
        if integer_weights:
            print(f"e {x+1} {y+1} {w}", file=f)
        else:
            print(f"e {x+1} {y+1} {w:.12g}", file=f)


def read_dimacs_matching(f: TextIO) -> tuple[int|float, Matching]:
    """Read a matching solution in DIMACS format."""

    have_weight = False
    weight: int|float = 0
    pairs: list[tuple[int, int]] = []

    for line in f:
        s = line.strip()
        words = s.split()

        if not words[0]:
            # Skip empty line.
            continue

        if words[0].startswith("c"):
            # Skip comment line.
            pass

        elif words[0] == "s":
            # Handle "solution" line.
            if len(words) != 2:
                raise ValueError(
                    f"Expecting solution line but got {s!r}")
            if have_weight:
                raise ValueError("Duplicate solution line")
            have_weight = True
            weight = parse_int_or_float(words[1])

        elif words[0] == "m":
            # Handle "matching" line.
            if len(words) != 3:
                raise ValueError(
                    f"Expecting matched edge but got {s!r}")
            x = int(words[1])
            y = int(words[2])
            if x < 1:
                raise ValueError("Invalid vertex index {x}")
            if y < 1:
                raise ValueError("Invalid vertex index {y}")
            pairs.append((x - 1, y - 1))

        else:
            raise ValueError(f"Unknown line type {words[0]!r}")

    if not have_weight:
        raise ValueError("Missing solution line")

    return (weight, Matching(pairs))


def make_random_graph(
        n: int,
        m: int,
        max_weight: float,
        float_weights: bool,
        rng: random.Random
        ) -> Graph:
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

    return Graph(edges)


def check_matching(graph: Graph, matching: Matching) -> int|float:
    """Verify that the matching is valid and calculate its weight."""

    edge_map: dict[tuple[int, int], int|float] = {}
    for (x, y, w) in graph.edges:
        edge_map[(min(x, y), max(x, y))] = w

    weight: int|float = 0
    nodes_used: set[int] = set()

    for pair in matching.pairs:
        x = min(pair)
        y = max(pair)
        if x in nodes_used:
            raise InvalidMatchingError(f"Matching uses vertex {x} twice")
        if y in nodes_used:
            raise InvalidMatchingError(f"Matching uses vertex {y} twice")
        if (x, y) not in edge_map:
            raise InvalidMatchingError(
                f"Matching contains non-existing edge ({x}, {y})")
        weight += edge_map[(x, y)]
        nodes_used.add(x)
        nodes_used.add(y)

    return weight


def compare_weight(weight1: int|float, weight2: int|float) -> int:
    """Compare weights of matchings.

    Returns:
        0 if weights are equal,
        -1 if weight1 < weight2,
        +1 if weight1 > weight2.
    """
    if isinstance(weight1, int) and isinstance(weight2, int):
        if weight1 == weight2:
            return 0
    else:
        if math.isclose(weight1, weight2, rel_tol=1e-9, abs_tol=1e-9):
            return 0
    if weight1 < weight2:
        return -1
    if weight1 > weight2:
        return 1
    return 0


class Solver:
    """Abstract base class for interaction with solvers."""

    def __init__(self, name: str, timeout: Optional[float]) -> None:
        self.name = name
        self.timeout = timeout

    def __str__(self) -> str:
        return self.name

    def prepare_input(self, graph: Graph) -> str:
        """Serialize a graph instance to input bytes for the solver."""
        raise NotImplementedError

    def parse_output(self, output_data: str) -> Matching:
        """Parse output bytes from the solver to a set of matched edges."""
        raise NotImplementedError

    def solver_command(self) -> list[str]:
        """Return the command and arguments to invoke the solver."""
        raise NotImplementedError

    def run(self, graph: Graph) -> tuple[Matching, float]:
        """Run the solver on a graph instance.

        Returns:
            Tuple (matching, elapsed_time).

        Raises:
            SolverError: If the solver returns an error status,
                         or produces incorrectly formatted output.
            TimeoutEpxired: If the solver exceeds the timeout.
        """

        input_str = self.prepare_input(graph)
        input_bytes = input_str.encode("ascii")

        command = self.solver_command()

        t0 = time.monotonic()

        proc = subprocess.run(
            command,
            stdout=subprocess.PIPE,
            input=input_bytes,
            timeout=self.timeout,
            check=False)

        t1 = time.monotonic()
        elapsed = t1 - t0

        if proc.returncode < 0:
            raise SolverError(f"Solver aborted on signal {-proc.returncode}")
        if proc.returncode > 0:
            raise SolverError(f"Solver exit status {proc.returncode}")

        try:
            output_str = proc.stdout.decode("ascii")
        except ValueError:
            raise SolverError("Non-ASCII data in solver output") from None

        pairs = self.parse_output(output_str)

        return (pairs, elapsed)


class DimacsSolver(Solver):
    """Interaction with a solver that uses DIMACS input/output."""

    def __init__(
            self,
            name: str,
            solver_command: str,
            timeout: Optional[float]
            ) -> None:
        """Initialize solver wrapper.

        The solver must read a graph in DIMACS format from stdin.
        The solver must write a matching in DIMACS format to stdout.

        Parameters:
            name:           Short label to refer to the solver.
            solver_command: Command to invoke the solver.
                            Arguments may be provided, separated by spaces.
                            Shell-style quoting may be used.
            timeout:        Optional run time limit.
        """
        super().__init__(name, timeout)
        self._solver_command = shlex.split(solver_command)

    def solver_command(self) -> list[str]:
        return self._solver_command

    def prepare_input(self, graph: Graph) -> str:
        buf = io.StringIO()
        write_dimacs_graph(buf, graph)
        return buf.getvalue()

    def parse_output(self, output_data: str) -> Matching:
        if not output_data:
            raise SolverError("Empty output from solver")
        buf = io.StringIO(output_data)
        try:
            (_weight, matching) = read_dimacs_matching(buf)
        except ValueError as exc:
            raise SolverError(exc) from None
        return matching


class WmatchSolver(Solver):
    """Interaction with the Wmatch solver."""

    def __init__(
            self,
            name: str,
            solver_command: str,
            timeout: Optional[float]
            ) -> None:
        """Initialize solver wrapper.

        Parameters:
            name:           Short label to refer to the solver.
            solver_command: Command to invoke the solver.
                            It should be a single binary without arguments.
            timeout:        Optional run time limit.
        """
        super().__init__(name, timeout)
        self._solver_command = [solver_command, "/dev/stdin"]

    def solver_command(self) -> list[str]:
        return self._solver_command

    def prepare_input(self, graph: Graph) -> str:

        num_vertex = graph.num_vertex()
        num_edge = len(graph.edges)

        all_integer = True
        adjacent: list[list[tuple[int, int|float]]] = [
            [] for i in range(num_vertex)]

        for (x, y, w) in graph.edges:
            adjacent[x].append((y, w))
            adjacent[y].append((x, w))
            all_integer = all_integer and isinstance(w, int)

        lines: list[str] = []
        lines.append(f"{num_vertex} {num_edge} U")

        for x in range(num_vertex):
            degree = len(adjacent[x])
            lines.append(f"{degree} {x+1} 0 0")
            for (y, w) in adjacent[x]:
                if all_integer:
                    lines.append(f"{y+1} {w}")
                else:
                    lines.append(f"{y+1} {w:.12g}")

        return "\n".join(lines) + "\n"

    def parse_output(self, output_data: str) -> Matching:

        pairs: list[tuple[int, int]] = []

        for line in output_data.splitlines():
            words = line.split()
            if len(words) != 2:
                raise SolverError("Invalid format in solver output")
            try:
                x = int(words[0])
                y = int(words[1])
            except ValueError:
                raise SolverError("Invalid format in solver output") from None
            if 0 < x < y:
                pairs.append((x - 1, y - 1))

        return Matching(pairs)


def test_solver_on_graph(
        solver: Solver,
        graph: Graph,
        graph_desc: str,
        gold_weight: Optional[int|float],
        num_run: int
        ) -> RunResult:
    """Test the specified solver with the specified graph."""

    solver_run_time: list[float] = []
    solver_weight: Optional[int|float] = None

    for i in range(num_run):

        print(f"Running {solver} on {graph_desc}, run {i+1}/{num_run} ...")
        sys.stdout.flush()

        try:
            (matching, elapsed) = solver.run(graph)
            weight = check_matching(graph, matching)
        except SolverError as exc:
            print("FAILED:", exc)
            return RunResult(status=RunStatus.FAILED)
        except InvalidMatchingError as exc:
            print("WRONG:", exc)
            return RunResult(status=RunStatus.WRONG_ANSWER)
        except subprocess.TimeoutExpired as exc:
            print("TIMEOUT:", exc)
            return RunResult(status=RunStatus.TIMEOUT)

        if gold_weight is not None:
            weight_diff = compare_weight(weight, gold_weight)
            if weight_diff > 0:
                raise ValueError(f"IMPOSSIBLE: Solver found weight {weight}"
                                 f" but reference answer is {gold_weight}")
            if weight_diff < 0:
                print(f"WRONG: matching has weight {weight}"
                      f" but expected {gold_weight}")
                return RunResult(status=RunStatus.WRONG_ANSWER)

        elif solver_weight is not None:
            weight_diff = compare_weight(weight, solver_weight)
            if weight_diff != 0:
                print(f"WRONG: matching has weight {weight}"
                      f" but previous run had weight {solver_weight}")
                return RunResult(status=RunStatus.WRONG_ANSWER)

        solver_weight = weight
        solver_run_time.append(elapsed)

    sys.stdout.flush()

    assert solver_weight is not None
    return RunResult(weight=solver_weight, run_time=solver_run_time)


def test_graph(
        solvers: list[Solver],
        graph: Graph,
        graph_desc: str,
        gold_weight: Optional[int|float],
        num_run: int
        ) -> list[RunResult]:
    """Test all solvers with the specified graph."""

    results: list[RunResult] = []

    for solver in solvers:
        result = test_solver_on_graph(
            solver,
            graph,
            graph_desc,
            gold_weight,
            num_run)
        results.append(result)

    if gold_weight is None:

        best_weight: Optional[int|float] = None
        for result in results:
            if result.status == RunStatus.OK:
                if (best_weight is None) or (result.weight > best_weight):
                    best_weight = result.weight

        if best_weight is not None:
            for (i, solver) in enumerate(solvers):
                result = results[i]
                if result.status == RunStatus.OK:
                    if compare_weight(result.weight, best_weight) != 0:
                        print(f"WRONG: Solver {solver} found weight"
                              f" {result.weight} but other solver found"
                              f" {best_weight}")
                        results[i] = RunResult(status=RunStatus.WRONG_ANSWER)

    return results


def test_random(
        solvers: list[Solver],
        num_vertex: int,
        num_edge: int,
        max_weight: int,
        float_weights: bool,
        num_run: int,
        seed: Optional[int]
        ) -> int:
    """Run a test with randomly generated graphs."""

    # Choose a random seed if the user does not provide a seed.
    if seed is None:
        seed = random.getrandbits(48)

    solver_num_fail: list[int] = [0 for solver in solvers]
    solver_num_wrong: list[int] = [0 for solver in solvers]
    solver_num_timeout: list[int] = [0 for solver in solvers]
    solver_run_time: list[list[float]] = [[] for solver in solvers]
    solver_errors: list[list[tuple[str, RunStatus]]] = [
        [] for solver in solvers]

    num_errors = 0

    for i in range(num_run):

        rng = random.Random(seed + i)

        graph_desc = f"graph {i+1}/{num_run} seed={seed+i}"
        graph = make_random_graph(
            num_vertex,
            num_edge,
            max_weight,
            float_weights,
            rng)

        results = test_graph(
            solvers=solvers,
            graph=graph,
            graph_desc=graph_desc,
            gold_weight=None,
            num_run=1)

        assert len(results) == len(solvers)
        for (j, result) in enumerate(results):
            if result.status == RunStatus.FAILED:
                solver_num_fail[j] += 1
            elif result.status == RunStatus.WRONG_ANSWER:
                solver_num_wrong[j] += 1
            elif result.status == RunStatus.TIMEOUT:
                solver_num_timeout[j] += 1
            else:
                solver_run_time[j] += result.run_time
            if result.status != RunStatus.OK:
                num_errors += 1
                solver_errors[j].append((graph_desc, result.status))

    print("Test finished.")
    print()

    if num_errors > 0:
        print("Failed tests")
        print("------------")
        for (j, solver) in enumerate(solvers):
            for (graph_desc, status) in solver_errors[j]:
                print(f"solver: {solver}, {graph_desc},"
                      f" status: {status.name}")
        print()

    print("Results")
    print("-------")
    print()

    print("N =", num_vertex)
    print("M =", num_edge)
    print("float" if float_weights else "integer", "weights;",
          "max_weight =", max_weight)
    print("runs =", num_run)
    print()

    for (i, solver) in enumerate(solvers):
        print(f"  {str(solver):32s} ", end="")
        num_fail = solver_num_fail[i]
        num_wrong = solver_num_wrong[i]
        num_timeout = solver_num_timeout[i]
        if (num_fail > 0) or (num_wrong > 0) or (num_timeout > 0):
            print(f"nFAIL={num_fail}  nWRONG={num_wrong}"
                  f"  nTIMEOUT={num_timeout}")
        else:
            run_time = solver_run_time[i]
            tmin = min(run_time)
            tmax = max(run_time)
            tavg = sum(run_time) / len(run_time)
            print(f"min={tmin:8.2f}  max={tmax:8.2f}  avg={tavg:8.2f}")

    print()
    if num_errors == 0:
        print("All tests passed")
    else:
        print(num_errors, "tests failed")

    return 0 if (num_errors == 0) else 2


def test_input(
        solvers: list[Solver],
        files: list[str],
        verify: bool,
        num_run: int
        ) -> int:
    """Run a test with graphs from input files."""

    result_table: list[list[RunResult]] = []

    for filename in files:

        try:
            with open(filename, "r", encoding="ascii") as f:
                graph = read_dimacs_graph(f)
        except (OSError, ValueError) as exc:
            print(f"ERROR: Can not read graph {filename!r} ({exc})",
                  file=sys.stderr)
            return 1

        gold_weight: Optional[int|float] = None
        if verify:
            reffile = os.path.splitext(filename)[0] + ".out"
            try:
                with open(reffile, "r", encoding="ascii") as f:
                    (gold_weight, _matching) = read_dimacs_matching(f)
            except (OSError, ValueError) as exc:
                print(f"ERROR: Can not read matching {reffile!r} ({exc})",
                      file=sys.stderr)
                return 1

        results = test_graph(solvers, graph, filename, gold_weight, num_run)
        result_table.append(results)

    print("Test finished.")
    print()
    print("Results")
    print("-------")
    print()

    errors = 0

    for (i, filename) in enumerate(files):
        print("Graph:", filename)
        for (j, solver) in enumerate(solvers):
            result = result_table[i][j]
            print(f"  {str(solver):32s} ", end="")
            if result.status == RunStatus.FAILED:
                print("FAILED")
                errors += 1
            elif result.status == RunStatus.WRONG_ANSWER:
                print("WRONG ANSWER")
                errors += 1
            elif result.status == RunStatus.TIMEOUT:
                print("TIMEOUT")
                errors += 1
            else:
                tmin = min(result.run_time)
                tmax = max(result.run_time)
                tavg = sum(result.run_time) / len(result.run_time)
                print(f"min={tmin:8.2f}  max={tmax:8.2f}  avg={tavg:8.2f}")
        print()

    if errors == 0:
        print("All tests passed")
    else:
        print(errors, "tests failed")

    return 0 if (errors == 0) else 2


def main() -> int:
    """Main program."""

    parser = argparse.ArgumentParser()
    parser.description = "Run matching solvers on test graphs."

    parser.add_argument("--solver",
                        action="append",
                        type=str,
                        nargs=3,
                        metavar=("NAME", "TYPE", "CMD"),
                        help="add a solver; NAME is a label;"
                             " TYPE is either 'dimacs' or 'wmatch';"
                             " CMD is the command to run the solver")
    parser.add_argument("--timeout",
                        action="store",
                        type=float,
                        help="abort when solver runs longer than TIMEOUT seconds")
    parser.add_argument("--runs",
                        action="store",
                        type=int,
                        metavar="R",
                        default=1,
                        help="run R random graphs or run input files R times")
    parser.add_argument("--random",
                        action="store_true",
                        help="generate random test graphs")
    parser.add_argument("--seed",
                        action="store",
                        type=int,
                        help="starting random seed")
    parser.add_argument("-n", "--n",
                        action="store",
                        type=int,
                        help="number of vertices for random graphs")
    parser.add_argument("-m", "--m",
                        action="store",
                        type=int,
                        help="number of edges for random graphs")
    parser.add_argument("--maxweight",
                        action="store",
                        type=int,
                        default=1000000,
                        metavar="W",
                        help="maximum edge weight for random graphs")
    parser.add_argument("--float",
                        action="store_true",
                        help="use floating point weights in random graphs")
    parser.add_argument("--verify",
                        action="store_true",
                        help="compare answers to existing output files")
    parser.add_argument("input",
                        nargs="*",
                        help="read test graphs from input files")

    args = parser.parse_args()

    if (not args.input) and (not args.random):
        print("ERROR: Specify at least one input file or --random",
              file=sys.stderr)
        print(file=sys.stderr)
        parser.print_help(sys.stderr)
        return 1

    if not args.solver:
        print("ERROR: Specify at least one solver", file=sys.stderr)
        return 1

    solvers: list[Solver] = []
    for (name, typ, cmd) in args.solver:
        if typ.lower() == "dimacs":
            solvers.append(DimacsSolver(name, cmd, args.timeout))
        elif typ.lower() == "wmatch":
            solvers.append(WmatchSolver(name, cmd, args.timeout))
        else:
            print("ERROR: Unknown solver type {type!r}", file=sys.stderr)
            return 1

    if args.runs < 1:
        print("ERROR: Number of runs must be >= 1", file=sys.stderr)
        return 1

    if args.random:

        if args.input:
            print("ERROR: Specify either --random or input files, not both",
                  file=sys.stderr)
            return 1

        if args.verify:
            print("ERROR: --verify not supported with --random",
                  file=sys.stderr)
            return 1

        if not args.n:
            print("ERROR: Missing required option --n", file=sys.stderr)
            return 1

        if not args.m:
            print("ERROR: Missing required option --m", file=sys.stderr)
            return 1

        if args.n < 2:
            print("ERROR: Number of vertices must be >= 2", file=sys.stderr)
            return 1

        if args.m < 1:
            print("ERROR: Number of edges must be >= 1", file=sys.stderr)
            return 1

        if args.m > args.n * (args.n - 1) // 2:
            print("ERROR: Too many edges", file=sys.stderr)
            return 1

        if args.maxweight < 1:
            print("ERROR: Maximum weight must be >= 1", file=sys.stderr)
            return 1

        return test_random(
            solvers=solvers,
            num_vertex=args.n,
            num_edge=args.m,
            max_weight=args.maxweight,
            float_weights=args.float,
            num_run=args.runs,
            seed=args.seed)

    else:

        return test_input(
            solvers=solvers,
            files=args.input,
            verify=args.verify,
            num_run=args.runs)


if __name__ == "__main__":
    sys.exit(main())
