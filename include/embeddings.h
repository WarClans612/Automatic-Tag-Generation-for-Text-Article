// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>

class Embeddings {

public:

    Embeddings();
    Embeddings(const std::string& fn);
    ~Embeddings() = default;

    void load_embed_file(const std::string& fn);
    const std::vector<double> &operator[] (const std::string& word);
    int stoi(const std::string& word);
    const std::string itos(const int& value);

private:

    std::string embed_file;
    std::unordered_map<int, std::vector<double>> embed;
    std::unordered_map<std::string, int> stoi_dict;
    std::unordered_map<int, std::string> itos_dict;
};