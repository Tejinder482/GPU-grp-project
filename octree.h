// octree.h
// Barnes-Hut octree. It is rebuilt on the CPU every time step and then
// flattened into plain arrays that get copied to the GPU.
//
// Flattened layout (depth-first, parent before children):
//   child[k]  first child of node k, or -1 if k is a leaf
//   next[k]   where the walk continues once k has been used or skipped
//
// In this order the whole subtree of k sits right after k, so
//   next[k] = k + (number of nodes in k's subtree)
// and the walk is finished when the index reaches the node count.
// With this the GPU can walk the tree with a single integer (no recursion,
// no stack), which is the main reason for flattening it like this.

#pragma once

#include <algorithm>
#include <cfloat>
#include <vector>

struct FlatTree {
    std::vector<float4> com;   // centre of mass in xyz, total mass in w
    std::vector<float>  width; // side length of the cell (0 for leaves)
    std::vector<int>    child;
    std::vector<int>    next;
    std::vector<int>    order; // particle indices in the order their leaves appear
    int numNodes() const { return (int)com.size(); }
};

class Octree {
public:
    // insertOrder (optional): order in which to insert the particles. Passing
    // the leaf order of the previous step's tree means particles that are
    // close in space get inserted one after another, so they walk the same
    // nodes and the cache is used much better. Made the build ~3x faster in my
    // tests. The tree itself comes out the same either way.
    void build(const std::vector<float4>& pos, const std::vector<int>* insertOrder = nullptr)
    {
        P = &pos;
        const int n = (int)pos.size();
        nodes.clear();
        nextBody.assign(n, -1);
        maxDepth = 0;

        // smallest cube around all particles (made slightly bigger so nothing
        // sits exactly on the outer boundary)
        float lo[3] = {FLT_MAX, FLT_MAX, FLT_MAX};
        float hi[3] = {-FLT_MAX, -FLT_MAX, -FLT_MAX};
        for (const float4& b : pos) {
            lo[0] = std::min(lo[0], b.x); hi[0] = std::max(hi[0], b.x);
            lo[1] = std::min(lo[1], b.y); hi[1] = std::max(hi[1], b.y);
            lo[2] = std::min(lo[2], b.z); hi[2] = std::max(hi[2], b.z);
        }
        float half = 0.5f * std::max(hi[0] - lo[0], std::max(hi[1] - lo[1], hi[2] - lo[2]));
        half = half * 1.001f + 1e-6f;
        newNode(0.5f * (lo[0] + hi[0]), 0.5f * (lo[1] + hi[1]), 0.5f * (lo[2] + hi[2]), half);

        if (insertOrder && (int)insertOrder->size() == n) {
            for (int b : *insertOrder) insert(b);
        } else {
            for (int b = 0; b < n; b++) insert(b);
        }
        computeMass(0);
    }

    void flatten(FlatTree& t)
    {
        t.com.clear();
        t.width.clear();
        t.child.clear();
        t.next.clear();
        t.order.clear();
        flattenRec(0, t);
    }

    int depth() const { return maxDepth; }

private:
    struct Node {
        float  cx, cy, cz, half; // cell centre and half of its side length
        int    kids[8];          // -1 = empty octant
        int    body;             // particle in this leaf, -1 for internal nodes
        double m, mx, my, mz;    // total mass and mass-weighted position sum
    };

    // If two particles are (almost) at the same spot, splitting would go on
    // forever. Past this depth we stop splitting and keep them in one leaf,
    // chained through nextBody. With random float input this basically never
    // happens, it is just a safety net.
    static constexpr int MAX_DEPTH = 30;

    std::vector<Node> nodes;
    std::vector<int> nextBody;
    const std::vector<float4>* P = nullptr;
    int maxDepth = 0;

    int newNode(float cx, float cy, float cz, float half)
    {
        Node nd;
        nd.cx = cx; nd.cy = cy; nd.cz = cz; nd.half = half;
        for (int k = 0; k < 8; k++) nd.kids[k] = -1;
        nd.body = -1;
        nd.m = nd.mx = nd.my = nd.mz = 0.0;
        nodes.push_back(nd);
        return (int)nodes.size() - 1;
    }

    int octant(int k, const float4& b) const
    {
        const Node& nd = nodes[k];
        return (b.x >= nd.cx ? 1 : 0) | (b.y >= nd.cy ? 2 : 0) | (b.z >= nd.cz ? 4 : 0);
    }

    // creates child o of node k (and links it). Careful: push_back can move
    // the vector, so only indices are used here, never references.
    int makeChild(int k, int o)
    {
        float h = 0.5f * nodes[k].half;
        float cx = nodes[k].cx + ((o & 1) ? h : -h);
        float cy = nodes[k].cy + ((o & 2) ? h : -h);
        float cz = nodes[k].cz + ((o & 4) ? h : -h);
        int c = newNode(cx, cy, cz, h);
        nodes[k].kids[o] = c;
        return c;
    }

    void insert(int b)
    {
        const float4 pb = (*P)[b];
        int k = 0, depth = 0;
        while (true) {
            if (nodes[k].body >= 0) {
                // k is a leaf that already has a particle -> split it
                if (depth >= MAX_DEPTH) {
                    nextBody[b] = nodes[k].body;
                    nodes[k].body = b;
                    return;
                }
                int old = nodes[k].body;
                nodes[k].body = -1;
                int c = makeChild(k, octant(k, (*P)[old]));
                nodes[c].body = old;
                maxDepth = std::max(maxDepth, depth + 1);
                // k is internal now, carry on inserting b below
            }
            int o = octant(k, pb);
            int c = nodes[k].kids[o];
            if (c < 0) {
                c = makeChild(k, o);
                nodes[c].body = b;
                maxDepth = std::max(maxDepth, depth + 1);
                return;
            }
            k = c;
            depth++;
        }
    }

    void computeMass(int k)
    {
        Node& nd = nodes[k]; // fine: no push_back happens in here
        if (nd.body >= 0) {
            for (int b = nd.body; b >= 0; b = nextBody[b]) {
                const float4& pb = (*P)[b];
                nd.m += pb.w;
                nd.mx += (double)pb.w * pb.x;
                nd.my += (double)pb.w * pb.y;
                nd.mz += (double)pb.w * pb.z;
            }
            return;
        }
        for (int o = 0; o < 8; o++) {
            int c = nd.kids[o];
            if (c < 0) continue;
            computeMass(c);
            nd.m += nodes[c].m;
            nd.mx += nodes[c].mx;
            nd.my += nodes[c].my;
            nd.mz += nodes[c].mz;
        }
    }

    // returns how many flat nodes this subtree produced
    int flattenRec(int k, FlatTree& t)
    {
        const Node& nd = nodes[k];
        const bool leaf = (nd.body >= 0);
        const int me = t.numNodes();

        float4 c;
        if (nd.m > 0.0) {
            c = make_float4((float)(nd.mx / nd.m), (float)(nd.my / nd.m), (float)(nd.mz / nd.m), (float)nd.m);
        } else {
            c = make_float4(nd.cx, nd.cy, nd.cz, 0.0f); // empty / massless cell
        }
        t.com.push_back(c);
        t.width.push_back(leaf ? 0.0f : 2.0f * nd.half);
        t.child.push_back(-1);
        t.next.push_back(-1);

        int count = 1;
        if (leaf) {
            for (int b = nd.body; b >= 0; b = nextBody[b]) t.order.push_back(b);
        } else {
            t.child[me] = me + 1; // first child always comes right after its parent
            // octants in order 0..7, so the leaves come out in Morton (Z-curve)
            // order, i.e. particles that are close in space get close indices
            for (int o = 0; o < 8; o++)
                if (nd.kids[o] >= 0) count += flattenRec(nd.kids[o], t);
        }
        t.next[me] = me + count;
        return count;
    }
};
