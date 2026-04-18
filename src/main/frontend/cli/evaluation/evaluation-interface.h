#ifndef evaluation_interface_H
#define evaluation_interface_H

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/FeaturesCollection.h"

void evaluation(std::map<std::string, bool>& flags, FeaturesCollection& features);

#endif
