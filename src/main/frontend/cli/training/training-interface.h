#ifndef training_interface_H
#define training_interface_H

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/FeaturesCollection.h"

void train(std::map<std::string, bool> flags, FeaturesCollection& features);

#endif
