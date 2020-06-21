// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <fstream>
#include <torch/torch.h>

class Example {

public:

    std::vector<std::string> text;
    std::vector<int> i_text;
    std::vector<float> label;
};

class Datasets {

public:

    Datasets();
    Datasets(const std::string& fn);
    Datasets(const std::string& fn1, const std::string& fn2);
    Datasets(const std::string& fn1, const std::string& fn2, const std::string& fn3);
    ~Datasets() = default;

    void load_embedding();
    void load_embedding(const std::string& fn);
    void update_Example(std::vector<Example>& target);
    void update_datasets();

    void load_train_file(const std::string& fn);
    void load_dev_file(const std::string& fn);
    void load_test_file(const std::string& fn);
    void load_file(std::vector<Example>& target, const std::string& fn);

    void init_epoch();
    torch::Tensor get_batch(int batch_size);

private:

    int current_batch_idx=0;
    std::string train_file;
    std::string dev_file;
    std::string test_file;

    std::vector<Example> train_it;
    std::vector<Example> dev_it;
    std::vector<Example> test_it;

    Embeddings embed;
};