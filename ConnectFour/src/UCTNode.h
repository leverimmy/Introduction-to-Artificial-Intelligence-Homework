#ifndef UCTNODE_H
#define UCTNODE_H

#include "Judge.h"
#include "Point.h"
#include "State.h"

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <ctime>
#include <iostream>
#include <vector>
using namespace std;

class UCTNode {
public:
    double reward;                      // 当前节点的胜率
    int visit;                          // 总访问次数
    State state;                        // 当前状态

    UCTNode *p;                         // 父节点
    UCTNode **ch;                       // 子节点

    vector<int> expandable_nodes;       // 可扩展节点

    UCTNode(State current_state, UCTNode *current_p = nullptr);
    virtual ~UCTNode();

    int solveCheckmate(bool chance);
    bool isTerminal();
};
#endif
