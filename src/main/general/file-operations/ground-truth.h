#ifndef ground_truth_H
#define ground_truth_H

// Standard
#include <vector>
#include <string>
// Fault Sense
#include "../../objects/SampleMat.h"

void readVisaLabels(const std::string objectCategory, std::map<std::string, std::array<std::string, 5>> &objectLabels);

#endif
