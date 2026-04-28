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
    std::vector<Bounds> bounds = {};
    std::vector<Bounds> tempBounds = {};
    int row = -1;

    void append(PixelGroup& theOtherGroup, int currentRow) {

        if (currentRow == row) {
            tempBounds.insert(tempBounds.end(), 
                 theOtherGroup.bounds.begin(), 
                 theOtherGroup.bounds.begin() + std::size(theOtherGroup.bounds));

        } else if (currentRow - 1 == row) {

            tempBounds.insert(tempBounds.end(), 
                 theOtherGroup.bounds.begin(), 
                 theOtherGroup.bounds.begin() + std::size(theOtherGroup.bounds));

            row = currentRow;

        } else
            throw std::runtime_error("Attempting to merge rows that are not connected");

        group.insert(group.end(), 
             theOtherGroup.group.begin(), 
             theOtherGroup.group.begin() + std::size(theOtherGroup.group));
    }

    bool newRow(int currentRow) {

        bounds = tempBounds;
        tempBounds = {};
        if (row == currentRow) {
            return true;

        } else if (row == currentRow - 1) {
            return false;

        }

        throw std::runtime_error("row has decremented\n");
    }
};

#endif
