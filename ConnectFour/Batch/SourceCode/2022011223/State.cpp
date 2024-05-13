#include "State.h"

State::State(int current_h, int current_w, int **current_board, int *current_top,
            int current_chance, const Point& current_position) :
    h(current_h), w(current_w), chance(current_chance), position(current_position) {
    board = new int*[h];
    for (int i = 0; i < h; ++i) {
        board[i] = new int[w];
        for (int j = 0; j < w; ++j) {
            board[i][j] = current_board[i][j];
        }
    }
    top = new int[w];
    for (int i = 0; i < w; ++i) {
        top[i] = current_top[i];
    }
};

State:: ~State() {
    for (int i = 0; i < h; ++i) {
        delete[] board[i];
    }
    delete[] board;
    delete[] top;
};

State::State(const State& other) {
    h = other.h;
    w = other.w;
    board = new int*[h];
    for (int i = 0; i < h; ++i) {
        board[i] = new int[w];
        for (int j = 0; j < w; ++j) {
            board[i][j] = other.board[i][j];
        }
    }
    top = new int[w];
    for (int i = 0; i < w; ++i) {
        top[i] = other.top[i];
    }
    chance = other.chance;
    position = other.position;
}

bool State::isTerminal() {
    return (chance && machineWin(position.x, position.y, h, w, board)) ||
            (!chance && userWin(position.x, position.y, h, w, board)) ||
            isTie(w, top);
}

int State::reward() {
    if (chance && machineWin(position.x, position.y, h, w, board))
        return 1;
    else if (!chance && userWin(position.x, position.y, h, w, board))
        return -1;
    else if (isTie(w, top))
        return 0;
}
