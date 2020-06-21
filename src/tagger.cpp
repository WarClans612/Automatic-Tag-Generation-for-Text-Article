// Developer: Wilbert (wilbert.phen@gmail.com)

#include <iostream>
#include <vector>
#include <string>
#include <ATen/ATen.h>
#include <torch/torch.h>
#include "model.h"
#include "embeddings.h"
#include "datasets.h"
#include "tagger.h"

Tagger::Tagger()
{
}


Tagger::Tagger(int output_channel, int epoch, int batch_size, float learning_rate,
    std::string train_file, std::string dev_file, std::string test_file,
    std::string embed_file)
{
    this->epoch = epoch;
    this->batch_size = batch_size;
    this->learning_rate = learning_rate;
    this->train_file = train_file;
    this->dev_file = dev_file;
    this->test_file = test_file;
    this->embed_file = embed_file;

    this->data_it = Datasets(this->train_file, this->dev_file, this->test_file);
    this->data_it.load_embedding(this->embed_file);
    this->data_it.update_datasets(); 
    this->net = std::make_shared<XMLCNN>(output_channel, this->data_it.get_embeddings());
}

void Tagger::train()
{
    torch::optim::Adam optimizer(net->parameters(), this->learning_rate);
    for(size_t epoch=1; epoch<=10; ++epoch) {
        std::cout << "Epoch " << epoch << std::endl;
        data_it.init_epoch();
        size_t batch_index=0;

        // Iterate the data loader to yield batches from the dataset.
        for(int i=0; i < (data_it.get_train_len()/batch_size)-1; ++i) {
            auto batch_data = data_it.get_batch(batch_size);
            auto batch_target = data_it.get_target(batch_size);
            // Reset gradients
            optimizer.zero_grad();
            // Execute the model on input data
            torch::Tensor prediction = net->forward(batch_data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = at::binary_cross_entropy_with_logits(prediction, batch_target);
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 20 batches.
            if (++batch_index % 20 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }
    torch::save(net, "net.pt");
}

torch::Tensor Tagger::test(std::string sentence)
{
    auto tokenized = this->data_it.preprocess_string(sentence);
    auto tokenized_int = this->data_it.sentence2int(tokenized);
    auto tensorized = this->data_it.vec2tensor(tokenized_int);

    torch::Tensor prediction = net->forward(tensorized);
    return prediction;
}

