// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <unordered_map>

class Embeddings {

public:

    Embeddings();
    Embeddings(const std::string& fn);
    ~Embeddings() = default;

    void load_embed_file(const std::string& fn);
    const std::vector<double> &operator[] (const std::string& word);

private:

    std::string embed_file;
    std::unordered_map<std::string, std::vector<double>> embed;

};

Embeddings::Embeddings()
    : embed_file("src/embeddings/word2vec/GoogleNews-vectors-negative300.txt")
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

    std::string key;
    int j = 0;
    while(ifs >> key)
    {
        std::vector<double> vec(dim);
        for (size_t i=0; i < dim; i++)
        {
            ifs >> vec[i];
        }
        embed[key] = vec;
        if (j++ > 300) break;
    }
    ifs.close();
}

const std::vector<double> & Embeddings::operator[] (const std::string& word) 
{
    return embed[word];
}
