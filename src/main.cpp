#include <ATen/ATen.h>
#include "model.h"

int main() {
    auto net = std::make_shared<XMLCNN>(torch::randn({2, 2}));
    torch::optim::Adam optimizer(net->parameters(), /*lr=*/0.01);

    // Data Loader

    // Training
    for(size_t epoch=1; epoch<=10; ++epoch) {
        size_t batch_index=0;

        // Iterate the data loader to yield batches from the dataset.
        for(auto& batch: *data_loader) {
            // Reset gradients
            optimizer.zero_grad();
            // Execute the model on input data
            torch::Tensor prediction = net->forward(batch.data);
            // Compute a loss value to judge the prediction of our model.
            torch::Tensor loss = at::binary_cross_entropy_with_logits(prediction, batch.target)
            // Compute gradients of the loss w.r.t. the parameters of our model.
            loss.backward();
            // Update the parameters based on the calculated gradients.
            optimizer.step();
            // Output the loss and checkpoint every 100 batches.
            if (++batch_index % 100 == 0) {
                std::cout << "Epoch: " << epoch << " | Batch: " << batch_index
                        << " | Loss: " << loss.item<float>() << std::endl;
                // Serialize your model periodically as a checkpoint.
                torch::save(net, "net.pt");
            }
        }
    }

    // Testing

    return 0;
}