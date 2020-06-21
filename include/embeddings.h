// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <torch/torch.h>

class Embeddings {

public:

    Embeddings();
    Embeddings(const std::string& fn);
    ~Embeddings() = default;

    torch::Tensor& get_embeddings();
    void load_embed_file(const std::string& fn);
    const std::vector<float> &operator[] (const std::string& word);
    int stoi(const std::string& word);
    const std::string itos(const int& value);

private:

    torch::Tensor embedding;
    std::string embed_file;
    std::unordered_map<int, std::vector<float>> embed;
    std::unordered_map<std::string, int> stoi_dict;
    std::unordered_map<int, std::string> itos_dict;
};