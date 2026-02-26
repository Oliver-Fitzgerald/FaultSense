#ifndef PixelCoordinates_H
#define PixelCoordinates_H

// Standard
#include <vector>

struct pixelCoordinate {
    int x;
    int y;
};
struct Bounds {
    int min;
    int max;
    int row;
};

struct pixelGroup {
    std::vector<pixelCoordinate> group;
    std::vector<Bounds> bounds;
    std::vector<Bounds> tempBounds;
    int row = -1;

    void append(pixelGroup& theOtherGroup, int currentRow) {

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
};

struct objectCoordinates {
    int xMin;
    int xMax;
    int yMin;
    int yMax;
};

#endif
