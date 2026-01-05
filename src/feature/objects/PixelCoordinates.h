#ifndef PixelCoordinates_H
#define PixelCoordinates_H

struct pixelCoordinate {
    int x;
    int y;
};

struct pixelGroup {
    std::vector<pixelCoordinate> group;
    int min;
    int max;
    bool redundant;

    void append(pixelGroup& theOtherGroup) {
        // Does not account for the case of a fork
        max = theOtherGroup.max;
        min = theOtherGroup.min;

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
