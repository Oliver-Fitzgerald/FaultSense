#ifndef training_interface_H
#define training_interface_H

// Standard
#include <map>
#include <string>
// Fault Sense
#include "../../../objects/PreProcessingPipeline.h"

void train(std::map<std::string, bool> flags, PreProcessingPipeline& preProcessingPipeline);

#endif
