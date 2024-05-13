#include "UCTNode.h"

UCTNode::UCTNode(State current_state, UCTNode *current_p)
    : state(current_state), p(current_p) {
    reward = visit = 0;
    expandable_nodes.clear();

    ch = new UCTNode*[state.w];
    for (int i = 0; i < state.w; ++i) {
        if (state.top[i] > 0) {
            expandable_nodes.push_back(i);
        }
        ch[i] = nullptr;
    }
}

UCTNode::~UCTNode() {
    for (int i = 0; i < state.w; ++i) {
        if (nullptr != ch[i]) {
            delete ch[i];
        }
    }
    delete[] ch;
}

typedef bool (*WinFunction)(int, int, int, int, int* const*);
WinFunction winFunctions[] = {userWin, machineWin};

int UCTNode::solveCheckmate(bool chance) {
    for (int idx = 0; idx <= 1; idx++) {
        for (int y = 0; y < state.w; ++y) {
            if (state.top[y] > 0) {
                int x = state.top[y] - 1;
                state.board[x][y] = 1 + (idx == chance);
                if (winFunctions[idx == chance](x, y, state.h, state.w, state.board)) {
                    state.board[x][y] = 0;
                    return y;
                }
                state.board[x][y] = 0;
            }
        }
    }
    return -1;
}

bool UCTNode::isTerminal() {
    return nullptr != p && state.isTerminal();
}
