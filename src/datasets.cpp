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
}

torch::Tensor& Datasets::get_embeddings()
{
    return embed.get_embeddings();
}

void Datasets::load_embedding(const std::string& fn)
{
    embed = Embeddings(fn);
    return;
}

void Datasets::update_Example(std::vector<Example>& target)
{
    for(auto& ex: target)
    {
        // Reserve size as much as text
        ex.i_text.reserve(ex.text.size());
        // Push converted string into integer
        for(auto& word: ex.text)
        {
            ex.i_text.push_back(embed.stoi(word));
        }
    }
}

void Datasets::update_datasets()
{
    // This function is used to update the text into correct integer for embedding
    // Check if the datasets iterator is not empty
    if (train_it.size() != 0) update_Example(train_it);
    if (dev_it.size() != 0) update_Example(dev_it);
    if (test_it.size() != 0) update_Example(test_it);
    std::cout << "Datasets finished updating" << std::endl;
    return;
}

void Datasets::load_train_file(const std::string& fn)
{
    load_file(train_it, fn);
    std::cout << "Training Datasets finished loading" << std::endl;
    return;
}

void Datasets::load_dev_file(const std::string& fn)
{
    load_file(dev_it, fn);
    std::cout << "Evaluation Datasets finished loading" << std::endl;
    return;
}

void Datasets::load_test_file(const std::string& fn)
{
    load_file(test_it, fn);
    std::cout << "Testing Datasets finished loading" << std::endl;
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
        target.push_back(results);
    }


    ifs.close();
    return;
}

int Datasets::get_train_len()
{
    return train_it.size();
}

void Datasets::init_epoch()
{
    current_batch_idx = 0;
    std::random_shuffle ( train_it.begin(), train_it.end() );
}

torch::Tensor Datasets::get_batch(int batch_size)
{
    // Get Max Length of the batch
    int max_len = 0;
    for(int i=current_batch_idx*batch_size; i < (current_batch_idx+1)*batch_size; ++i)
    {
        if (train_it[i].i_text.size() > max_len)
        {
            max_len = train_it[i].i_text.size();
        }
    }

    // Concatenate sentence into one batch
    auto a = train_it[current_batch_idx*batch_size].i_text;
    a.resize(max_len, 0);
    auto results = torch::tensor(a);
    results = at::reshape(results, {1, -1});
    for(int i=current_batch_idx*batch_size+1; i < (current_batch_idx+1)*batch_size; ++i)
    {
        a = train_it[i].i_text;
        a.resize(max_len, 0);
        results = torch::cat({results, at::reshape(torch::tensor(a), {1, -1})});
    }
    current_batch_idx++;
    return results;
}

torch::Tensor Datasets::get_target(int batch_size)
{
    // Get Max Length of the batch
    int max_len = 0;
    for(int i=current_batch_idx*batch_size; i < (current_batch_idx+1)*batch_size; ++i)
    {
        if (train_it[i].label.size() > max_len)
        {
            max_len = train_it[i].label.size();
        }
    }

    // Concatenate sentence into one batch
    auto a = train_it[current_batch_idx*batch_size].label;
    a.resize(max_len, 0);
    auto results = torch::tensor(a);
    results = at::reshape(results, {1, -1});
    for(int i=current_batch_idx*batch_size+1; i < (current_batch_idx+1)*batch_size; ++i)
    {
        a = train_it[i].label;
        a.resize(max_len, 0);
        results = torch::cat({results, at::reshape(torch::tensor(a), {1, -1})});
    }
    current_batch_idx++;
    return results;
}
