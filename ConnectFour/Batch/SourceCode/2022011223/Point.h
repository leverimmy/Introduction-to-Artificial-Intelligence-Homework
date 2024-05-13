#ifndef POINT_H
#define POINT_H

class Point {
public:
    int x, y;

    Point(int xx = -1, int yy = -1) : x(xx), y(yy) {}
    virtual ~Point() {}

    bool operator==(const Point &p) const {
        return x == p.x && y == p.y;
    }
};
#endif
