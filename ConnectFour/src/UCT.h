#ifndef UCT_H
#define UCT_H

#include "Point.h"
#include "UCTNode.h"

#include <cmath>
#include <ctime>
#include <vector>
using namespace std;

constexpr double PARAMETER = 0.75;
constexpr double TIME_LIMIT = 2.0 * CLOCKS_PER_SEC;

class UCT {
public:
    int h, w;               // 棋盘高度和宽度
    Point ban;              // 被去除的点位
    UCTNode *root;

    int *weight;
    int total_weight;

    UCT(int current_h, int current_w, const Point& current_ban, int **current_board, int *current_top);
    virtual ~UCT();

    UCTNode *UCTSearch();
    UCTNode *TreePolicy(UCTNode *v);
    UCTNode *Expand(UCTNode *v);
    UCTNode *BestChild(UCTNode *v, double c);
    double DefaultPolicy(UCTNode *node);
    void BackUp(UCTNode *v, double Delta);
};
#endif
