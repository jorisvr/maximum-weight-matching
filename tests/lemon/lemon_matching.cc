/*
 * Use LEMON to solve a maximum-weight matching problem.
 *
 * This program is designed for LEMON version 1.3.1.
 *
 * Download LEMON from https://lemon.cs.elte.hu/trac/lemon
 * or install Debian package "liblemon-dev".
 *
 * This program reads input from stdin or from a specified filename.
 * Input should be a weighted graph in DIMACS edge-list format.
 *
 * This program writes output to stdout.
 * The output is a matching in DIMACS matching format.
 */

#include <cstdlib>
#include <cerrno>
#include <cstring>
#include <fstream>
#include <iostream>
#include <string>
#include <sstream>
#include <unordered_map>
#include <utility>

#include <lemon/smart_graph.h>
#include <lemon/matching.h>


namespace {  // anonymous namespace

typedef long long IntWeight;
typedef double FloatWeight;


/* Weighted edge. */
template <typename WeightType>
struct WeightedEdge
{
    int x, y;
    WeightType w;

    WeightedEdge(int x, int y, WeightType w)
      : x(x), y(y), w(w)
    { }
};


/* Graph defined by a list of weighted edges. */
template <typename WeightType>
using WeightedEdgeList = std::vector<WeightedEdge<WeightType>>;


/* Wrapper that holds an edge list with either integer or float weights. */
struct WeightedEdgeListVariant
{
    bool is_int_weight;
    WeightedEdgeList<IntWeight> int_edges;
    WeightedEdgeList<FloatWeight> float_edges;
};


/* Matching defined by a list of matched edges. */
struct Matching
{
    bool is_int_weight;
    IntWeight int_weight;
    FloatWeight float_weight;
    std::vector<std::pair<int, int>> pairs;

    Matching() = default;

    Matching(IntWeight weight, std::vector<std::pair<int, int>>&& pairs)
      : is_int_weight(true),
        int_weight(weight),
        float_weight(0),
        pairs(pairs)
    { }

    Matching(FloatWeight weight, std::vector<std::pair<int, int>>&& pairs)
      : is_int_weight(false),
        int_weight(0),
        float_weight(weight),
        pairs(pairs)
    { }
};


/* Wrapper for a LEMON graph with edge weights and node indices. */
template <typename WeightType>
struct LemonGraph
{
    lemon::SmartGraph graph;
    lemon::SmartGraph::NodeMap<int> node_index;
    lemon::SmartGraph::EdgeMap<WeightType> edge_weight;

    /* Construct a LEMON graph from a list of weighted edges. */
    explicit LemonGraph(const WeightedEdgeList<WeightType>& edges)
      : graph(),
        node_index(graph),
        edge_weight(graph)
    {
        std::unordered_map<int, lemon::SmartGraph::Node> node_map;

        for (const WeightedEdge<WeightType>& e : edges) {
            lemon::SmartGraph::Node node_x = node_from_index(node_map, e.x);
            lemon::SmartGraph::Node node_y = node_from_index(node_map, e.y);
            lemon::SmartGraph::Edge edge = graph.addEdge(node_x, node_y);
            edge_weight.set(edge, e.w);
        }
    }

private:
    /* Return the Node that corresponds to the specified vertex index. */
    lemon::SmartGraph::Node
    node_from_index(std::unordered_map<int, lemon::SmartGraph::Node>& node_map,
                    int x)
    {
        auto it = node_map.find(x);
        if (it == node_map.end()) {
            lemon::SmartGraph::Node node = graph.addNode();
            node_index.set(node, x);
            node_map[x] = node;
            return node;
        } else {
            return it->second;
        }
    }
};


/* Parse an integer edge weight. */
bool parse_int_weight(const std::string& s, long long& v)
{
    const char *p = s.c_str();
    char *q;
    errno = 0;
    v = std::strtoll(p, &q, 10);
    return (q != p && q[0] == '\0' && errno == 0);
}


/* Parse a floating point edge weight. */
bool parse_float_weight(const std::string& s, double& v)
{
    const char *p = s.c_str();
    char *q;
    errno = 0;
    v = std::strtod(p, &q);
    return (q != p && q[0] == '\0' && errno == 0);

}


/* Read a graph in DIMACS edge list format. */
int read_dimacs_graph(std::istream& input, WeightedEdgeListVariant& res)
{
    res.is_int_weight = true;
    res.int_edges.clear();
    res.float_edges.clear();

    while (!input.eof()) {
        std::string line;
        std::getline(input, line);
        if (!input) {
            if (input.eof()) {
                break;
            } else {
                return -1;
            }
        }

        line.erase(0, line.find_first_not_of(" \n\r\t"));
        line.erase(line.find_last_not_of(" \n\r\t") + 1);

        if (line.empty()) {
            // skip empty line
        } else if (line[0] == 'c') {
            // skip comment line
        } else if (line[0] == 'p') {
            // handle problem line
            std::istringstream is(line);
            std::string cmd, fmt;
            is >> cmd >> fmt;
            if (!is || cmd != "p" || fmt != "edge") {
                std::cerr << "ERROR: Expected DIMACS edge format but got '"
                          << line << "'" << std::endl;
                return -1;
            }
        } else if (line[0] == 'e') {
            // handle edge
            std::istringstream is(line);

            std::string cmd, weight;
            unsigned int x, y;
            IntWeight wi;
            FloatWeight wf;

            is >> cmd >> x >> y >> weight;
            if (!is || cmd != "e" || x < 1 || y < 1) {
                std::cerr << "ERROR: Expected edge but got '"
                          << line << "'" << std::endl;
                return -1;
            }

            if (res.is_int_weight && parse_int_weight(weight, wi)) {
                wf = wi;
                res.int_edges.push_back(WeightedEdge<IntWeight>(x, y, wi));
            } else {
                if (!parse_float_weight(weight, wf)) {
                    std::cerr << "ERROR: Expected edge but got '"
                              << line << "'" << std::endl;
                    return -1;
                }
                res.is_int_weight = false;
            }
            res.float_edges.push_back(WeightedEdge<FloatWeight>(x, y, wf));

        } else {
            std::cerr << "ERROR: Unknown line format '" << line << "'"
                      << std::endl;
            return -1;
        }
    }

    if (res.is_int_weight) {
        res.float_edges.clear();
    } else {
        res.int_edges.clear();
    }

    return 0;
}


/* Write matching in DIMACS format. */
void write_dimacs_matching(std::ostream& f, const Matching& matching)
{
    f << "s ";
    if (matching.is_int_weight) {
        f << matching.int_weight;
    } else {
        f.precision(12);
        f << matching.float_weight;
    }
    f << std::endl;
    for (auto pair : matching.pairs) {
        f << "m " << pair.first << " " << pair.second << std::endl;
    }
}


/* Solve a maximum-weight-matching problem. */
template <typename WeightType>
Matching run_matching(const WeightedEdgeList<WeightType>& edges)
{
    LemonGraph<WeightType> lemon_graph(edges);

    lemon::MaxWeightedMatching<lemon::SmartGraph,
                               lemon::SmartGraph::EdgeMap<WeightType>>
        match(lemon_graph.graph, lemon_graph.edge_weight);
    match.run();

    std::vector<std::pair<int, int>> pairs;

    for (lemon::SmartGraph::NodeIt node_x(lemon_graph.graph);
            node_x != lemon::INVALID;
            ++node_x) {
        lemon::SmartGraph::Node node_y = match.mate(node_x);
        if (node_y != lemon::INVALID) {
            int x = lemon_graph.node_index[node_x];
            int y = lemon_graph.node_index[node_y];
            if (x < y) {
                pairs.push_back(std::make_pair(x, y));
            }
        }
    }

    return Matching(match.matchingWeight(), std::move(pairs));
}


void usage()
{
    std::cerr
        << std::endl
        << "Solves a maximum-weight matching problem with LEMON."
        << std::endl << std::endl
        << "Usage: lemon_matching < inputfile.gr" << std::endl
        << "   or: lemon_matching inputfile.gr" << std::endl
        << std::endl
        << "    inputfile.gr    Input file in DIMACS edge-list format"
        << std::endl << std::endl;
}

};  // anonymous namespace


int main(int argc, const char **argv)
{
    std::string input_file;
    bool end_options = false;

    for (int i = 1; i < argc; i++) {
        if (!end_options && argv[i][0] == '-') {
            if (std::string("--help") == argv[i]
                    || std::string("-h") == argv[i]) {
                usage();
                return 0;
            } else if (std::string("--") == argv[i]) {
                end_options = true;
            } else {
                std::cerr << "ERROR: Unknown option " << argv[i] << std::endl;
                usage();
                return 1;
            }
        } else {
            if (!input_file.empty()) {
                std::cerr << "ERROR: Multiple input files not supported."
                          << std::endl;
                usage();
                return 1;
            }
            input_file = argv[i];
        }
    }

    WeightedEdgeListVariant edges;

    if (input_file.empty()) {

        if (read_dimacs_graph(std::cin, edges) != 0) {
            if (!std::cin) {
                std::cerr << "ERROR: Can not read from stdin ("
                          << std::strerror(errno) << ")" << std::endl;
            }
            return 1;
        }

    } else {

        std::ifstream f(input_file);
        if (!f) {
            std::cerr << "ERROR: Can not open '" << input_file << "' ("
                      << std::strerror(errno) << ")" << std::endl;
            return 1;
        }
        if (read_dimacs_graph(f, edges) != 0) {
            if (!f) {
                std::cerr << "ERROR: Can not read from '" << input_file
                          << "' (" << std::strerror(errno) << ")" << std::endl;
            }
            return 1;
        }

    }

    Matching matching;

    if (edges.is_int_weight) {
        matching = run_matching(edges.int_edges);
    } else {
        matching = run_matching(edges.float_edges);
    }

    write_dimacs_matching(std::cout, matching);

    return 0;
}
