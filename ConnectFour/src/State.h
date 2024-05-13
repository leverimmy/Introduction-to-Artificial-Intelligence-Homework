#ifndef STATE_H
#define STATE_H

#include "Judge.h"
#include "Point.h"

class State {
public:
	int h, w;							// 棋盘的宽和高
	int **board;						// 棋盘的状态
	int *top;							// 每一列的顶部
	bool chance;						// 是否为己方棋子
	Point position;						// 落子位置

	State(int current_h, int current_w, int **current_board, int *current_top,
			int current_chance = false, const Point& current_position = Point(-1, -1));
	virtual ~State();
	State(const State& other);

	bool isTerminal();
	int reward();
};
#endif
