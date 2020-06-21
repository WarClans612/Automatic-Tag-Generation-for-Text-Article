// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <regex>
#include <torch/torch.h>
#include "embeddings.h"
#include "datasets.h"

Datasets::Datasets()
    : train_file("../train.tsv"), dev_file("../dev.tsv"), test_file("../test.tsv")
{
    // Load datasets
    load_train_file(train_file);
    //load_dev_file(dev_file);
    //load_test_file(test_file);
}

Datasets::Datasets(const std::string& fn)
    : train_file(fn)
{
    // Load datasets
    load_train_file(train_file);
}

Datasets::Datasets(const std::string& fn1, const std::string& fn2)
    : train_file(fn1), dev_file(fn2)
{
    // Load datasets
    load_train_file(train_file);
    load_dev_file(dev_file);
}

Datasets::Datasets(const std::string& fn1, const std::string& fn2, const std::string& fn3)
    : train_file(fn1), dev_file(fn2), test_file(fn3)
{
    // Load datasets
    load_train_file(train_file);
    load_dev_file(dev_file);
    load_test_file(test_file);
    update_datasets();
}

void Datasets::load_embedding()
{
    embed = Embeddings("../word2vec/GoogleNews-vectors-negative300.txt");
    return;
}

void Datasets::load_embedding(const std::string& fn)
{
    embed = Embeddings(fn);
    return;
}



void Datasets::load_train_file(const std::string& fn)
{
    load_file(train_it, fn);
    return;
}

void Datasets::load_dev_file(const std::string& fn)
{
    load_file(dev_it, fn);
    return;
}

void Datasets::load_test_file(const std::string& fn)
{
    load_file(test_it, fn);
    return;
}

void Datasets::load_file(std::vector<Example>& target, const std::string& fn)
{
    std::ifstream ifs (fn, std::ifstream::in);

    std::string label;
    std::string text;
    while(ifs >> label)
    {
        Example results;
        // Convert label string into vector and save into results
        for(char& c : label)
        {
            results.label.push_back(double(c-'0'));
        }

        // Clean string
        std::getline(ifs, text);
        std::regex e_clean("[^A-Za-z0-9(),!?\'`]");
        text = std::regex_replace (text, e_clean, " ");
        std::regex e_space("\\s{2,}");
        text = std::regex_replace (text, e_space, " ");
        std::transform(text.begin(), text.end(), text.begin(),
            [](unsigned char c){ return std::tolower(c); });

        // Tokenize string
        std::stringstream ss;
        ss.str(text);
        std::string word;
        while(ss >> word)
        {
            results.text.push_back(word);
        }
    }


    ifs.close();
    return;
}
