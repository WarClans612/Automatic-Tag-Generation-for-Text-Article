// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <torch/torch.h>

class Tagger {

public:

    Tagger();
    Tagger(int output_channel, int epoch, int batch_size, float learning_rate,
        std::string train_file, std::string dev_file, std::string test_file,
        std::string embed_file);
    ~Tagger() = default;

    void train();
    std::vector<float> test(std::string sentence);

private:

    std::shared_ptr<XMLCNN> net;
    Datasets data_it;
    int output_channel;
    int epoch;
    int batch_size;
    float learning_rate;
    std::string train_file;
    std::string dev_file;
    std::string test_file;
    std::string embed_file;

};