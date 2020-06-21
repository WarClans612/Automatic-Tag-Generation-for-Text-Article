// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>
#include "embeddings.h"

Embeddings::Embeddings()
    : embed_file("../word2vec/GoogleNews-vectors-negative300.txt")
{
    // Loading the entire embeddings file
    load_embed_file(embed_file);
}

Embeddings::Embeddings(const std::string& fn)
    : embed_file(fn)
{
    // Loading the entire embeddings file
    load_embed_file(fn);
}

void Embeddings::load_embed_file(const std::string& fn)
{
    std::ifstream ifs (fn, std::ifstream::in);
    int len, dim;
    ifs >> len >> dim;
    std::cout << "Embeddings Filename " << fn << std::endl;
    std::cout << "Length of corpus is " << len << std::endl;
    std::cout << "Dimension size is " << dim << std::endl;
    embed.reserve(len);
    stoi_dict.reserve(len);
    itos_dict.reserve(len);

    std::string key;
    int j = 0;
    while(ifs >> key)
    {
        std::vector<double> vec(dim);
        for (int i=0; i < dim; i++)
        {
            ifs >> vec[i];
        }
        embed[j] = vec;
        stoi_dict[key] = j;
        itos_dict[j] = key;
        j++;
        if (j == 300) break;
    }
    ifs.close();
}

const std::vector<double> & Embeddings::operator[] (const std::string& word) 
{
    return embed[stoi_dict[word]];
}

int Embeddings::stoi(const std::string& word)
{
    return stoi_dict[word];
}

const std::string Embeddings::itos(const int& value)
{
    return itos_dict[value];
}

