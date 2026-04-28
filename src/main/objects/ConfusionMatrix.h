#pragma once

struct ConfusionMatrix {
    int truePositiveCount  = 0;
    int falsePositiveCount = 0;
    int trueNegativeCount  = 0;
    int falseNegativeCount = 0;

    void update(bool result, bool actualResult) {
        if (result && actualResult) {           // predicted yes, actually yes
            truePositiveCount += 1;
        } else if (result && !actualResult) {   // predicted yes, actually no
            falsePositiveCount += 1;
        } else if (!result && !actualResult) {  // predicted no, actually no
            trueNegativeCount += 1;
        } else if (!result && actualResult) {   // predicted no, actually yes
            falseNegativeCount += 1;
        }
    }

    friend std::ostream& operator<<(std::ostream& os, const ConfusionMatrix& c) {
        os << "\033[32mTP\033[0m: " << c.truePositiveCount
           << " \033[91mFP\033[0m: " << c.falsePositiveCount
           << " \033[32mTN\033[0m: " << c.trueNegativeCount
           << " \033[91mFN\033[0m: " << c.falseNegativeCount;
        return os;
    }
};
