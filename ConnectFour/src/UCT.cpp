#include "UCT.h"

UCT::UCT(int current_h, int current_w, const Point& current_ban, int **current_board, int *current_top)
    : h(current_h), w(current_w), ban(current_ban), total_weight(0) {
    
    root = new UCTNode(State(h, w, current_board, current_top));
    weight = new int[w];

    for (int i = 0; i < w; i++) {
        if (i <= w / 2)
            weight[i] = (i + 1) * (i + 1);
        else
            weight[i] = weight[w - i - 1];
        total_weight += weight[i];
    }
}

UCT::~UCT() {
    delete root;
    delete[] weight;
}

UCTNode* UCT::UCTSearch() {
    int next_y = root->solveCheckmate(root->state.chance);
    // 必须走的情况：如果有必赢的一步，直接返回；如果需要防御对方的将军，也直接返回
    if (next_y != -1) {
        int next_x = --(root->state.top[next_y]);
        root->state.board[next_x][next_y] = root->state.chance ? 2 : 1;
        if (Point(next_x - 1, next_y) == ban)
            --(root->state.top[next_y]);
        return root->ch[next_y] = new UCTNode(State(root->state.h, root->state.w,
                                                   root->state.board,
                                                   root->state.top, !(root->state.chance), Point(next_x, next_y)), root);
    }
    // 非必须走的情况：进行搜索
    clock_t start_time = clock();
    while (clock() - start_time < TIME_LIMIT) {
        UCTNode *next_node = TreePolicy(root);
        double reward = DefaultPolicy(next_node);
        BackUp(next_node, reward);
    }
    return BestChild(root, 0);
}

UCTNode* UCT::TreePolicy(UCTNode *v) {
    while (!v->isTerminal()) {
        if (!v->expandable_nodes.empty()) {
            return Expand(v);
        } else {
            v = BestChild(v, PARAMETER);
        }
    }
    return v;
}

UCTNode* UCT::Expand(UCTNode *v) {
    int random_index = rand() % (v->expandable_nodes.size());

    State next_state = v->state;

    next_state.chance = !next_state.chance;
    next_state.position.y = v->expandable_nodes[random_index];
    next_state.position.x = --next_state.top[next_state.position.y];
    next_state.board[next_state.position.x][next_state.position.y] = next_state.chance ? 2 : 1;

    if (Point(next_state.position.x - 1, next_state.position.y) == ban)
        --next_state.top[next_state.position.y];

    v->ch[next_state.position.y] = new UCTNode(next_state, v);
    v->expandable_nodes[random_index] = v->expandable_nodes.back();
    v->expandable_nodes.pop_back();
    return v->ch[next_state.position.y];
}

UCTNode* UCT::BestChild(UCTNode *v, double c) {
    double best_score = 0;
    UCTNode *best_node = nullptr;
    for (int i = 0; i < v->state.w; ++i) {
        UCTNode* u = v->ch[i];
        if (nullptr != u) {
            double current_score = (v->state.chance ? -1 : 1) *
                                    u->reward / u->visit +
                                    c * sqrt(2 * log(v->visit) / u->visit);
            if (nullptr == best_node || current_score > best_score) {
                best_node = u;
                best_score = current_score;
            }
        }
    }
    return best_node;
}

double UCT::DefaultPolicy(UCTNode *v) {
    State next_state = v->state;
    while (!next_state.isTerminal()) {
        next_state.chance = !next_state.chance;
        next_state.position.y = 0;
        while (next_state.top[next_state.position.y] == 0) {
            int random_index = rand() % total_weight;
            int sum = 0;
            for (int i = 0; i < w; ++i) {
                sum += weight[i];
                if (sum > random_index) {
                    next_state.position.y = i;
                    break;
                }
            }
        }
        next_state.position.x = --next_state.top[next_state.position.y];
        next_state.board[next_state.position.x][next_state.position.y] = next_state.chance ? 2 : 1;
        if (Point(next_state.position.x - 1, next_state.position.y) == ban)
            --next_state.top[next_state.position.y];
    }
    return next_state.reward();
}

void UCT::BackUp(UCTNode *v, double Delta) {
    while (nullptr != v) {
        v->visit++;
        v->reward += Delta;
        v = v->p;
    }
}
