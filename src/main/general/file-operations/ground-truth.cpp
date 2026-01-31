/*
 * ground-truth
 * Contains file operations for the parsing of data-labels of samples from the 
 * visual anomaly dataset (VisA)
 */

// Standard
#include <fstream>
#include <vector>
#include <string>
// Fault Sense
#include "../../objects/SampleMat.h"

void readVisaLabels(const std::string objectCategory, std::map<std::string, std::array<std::string, 5>> &objectLabels);

const std::string DATA_ROOT = "../../../data/";

/*
 * readVisaLabels
 * Reads the annotations of an object category into a vector of SampleMat
 * where a classification can be "normal" or "anomaly"
 *
 * @param objectCategory The object type for which annoations will be fetched
 * @param objectLabels The vector to be populated
 */
void readVisaLabels(const std::string objectCategory, std::map<std::string, std::array<std::string, 5>> &objectLabels) {

    std::ifstream file(DATA_ROOT + objectCategory + "/image_anno.csv");

    if (!file.is_open()) {
        throw std::runtime_error("Couldn't open file: " + DATA_ROOT + objectCategory + "image_anno.csv");
    }

    std::string line;
    std::string filename;
    while (std::getline(file, line)) {

        // Skip header line
        if (line == "image,label,mask") continue;

        filename = line;
        filename.erase(0, filename.find_last_of("/") + 1);
        if (filename.find(",") != filename.npos)
            filename.erase(filename.find(","));
        if (filename.find("\r") != filename.npos)
            filename.erase(filename.find("\r"));

        // Get Anomaly types
        if (line.find("Anomaly", line.size() - 16) != line.npos) {

            std::array<std::string, 5> anomalyTypes = {std::string("")};
            int index = 0;

            // Trim start and end of line so only anomaly types remain
            line.erase(0, line.find_first_of(","));
            line.erase(line.find_last_of(","));
            if (line.find("\"",1) != line.npos) {
                line.erase(1,1); 
                line.erase(line.size() - 1); 
            }

            // If an anomaly type is present add it to anomalyTypes
            while (line.find(",",0) != line.npos) {

                line.erase(0,1);
                anomalyTypes[index] = line.substr(0, line.find(","));
                line.erase(0,line.find(","));
                index++;
            }

            objectLabels.insert({"A" + filename, anomalyTypes});

        } else { // If normal
            objectLabels.insert({"N" + filename, std::array<std::string, 5>{std::string("normal"),std::string(""),std::string(""),std::string(""),std::string("")}});
        }
    }
    file.close();
}

