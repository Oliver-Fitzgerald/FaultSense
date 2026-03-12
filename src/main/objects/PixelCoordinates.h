#ifndef PixelCoordinates_H
#define PixelCoordinates_H

// Standard
#include <vector>

struct PixelCoordinate {
    int x;
    int y;
};
struct Bounds {
    int min;
    int max;
    int row;
};

struct PixelGroup {
    std::vector<PixelCoordinate> group;
    std::vector<Bounds> bounds;
    std::vector<Bounds> tempBounds;
    int row = -1;

    void append(PixelGroup& theOtherGroup, int currentRow) {

        if (currentRow - 1 > row) {
            row == currentRow;
            bounds = tempBounds;
            tempBounds = {};

        } else {

            tempBounds.insert(tempBounds.end(), 
                 theOtherGroup.bounds.begin(), 
                 theOtherGroup.bounds.begin() + std::size(theOtherGroup.bounds));
        }

        group.insert(group.end(), 
             theOtherGroup.group.begin(), 
             theOtherGroup.group.begin() + std::size(theOtherGroup.group));
    }

    bool newRow(int currentRow) {

        if (row == currentRow) {
            return true;

        } else if (row == currentRow - 1) {
            bounds = tempBounds;
            tempBounds = {};
            return true;

        } else if (row <= currentRow - 2) {
            return false;
        }

        throw std::runtime_error("row has decremented\n");
    }
};

#endif
