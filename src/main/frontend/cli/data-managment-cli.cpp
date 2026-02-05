/*
 * data-managment-cli
 * Interface for the managment of training, testing and validation data
 */

// CLI11
#include <CLI/CLI.hpp>
// Fault Sense
#include "../../data-preperation/generate-masks.h"
#include "../../data-preperation/synthetic-data-generation.h"

void generate(std::map<const char*, bool> flags);

int main(int argc, char** argv) {

    CLI::App dataManagmentCLI{"A utility for the manangment of datasets and historical data"};
    dataManagmentCLI.require_subcommand();
    argv = dataManagmentCLI.ensure_utf8(argv);

    // Generate synthetic data
    CLI::App* generateSubcommand = dataManagmentCLI.add_subcommand("generate", "Generates synthetic data for testing")->ignore_case()->require_option(1);
    std::map<const char*, bool> generateFlags = {{"all", false}, {"removeNoise", false}};

    generateSubcommand->add_flag("--all, -a", generateFlags["all"], "Generate synthetic testing data for the all test methods");
    generateSubcommand->add_flag("--removeNoise", generateFlags["removeNoise"], "Generate synthetic testing data for the removeNoise method");
    generateSubcommand->final_callback([&generateFlags]() {
        generate(generateFlags);
    });

    CLI11_PARSE(dataManagmentCLI, argc, argv);
    return 0;
}

void generate(std::map<const char*, bool> flags) {

    if (flags["all"] || flags["removeNoise"]) {
        generateRemoveNoiseTestData();
    }
}
