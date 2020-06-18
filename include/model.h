#include <torch/torch.h>
#include <ATen/ATen.h>
#include <iostream>
#include <vector>

struct XMLCNN : torch::nn::Module {
    XMLCNN(){}

    XMLCNN(torch::Tensor TEXT_FIELD) {
        embed = register_module("embed", torch::nn::Embedding::from_pretrained(TEXT_FIELD));
        conv1 = register_module("conv1", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 100, {2, 300}).stride(1).padding({1, 0})));
        conv2 = register_module("conv2", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 100, {4, 300}).stride(1).padding({3, 0})));
        conv3 = register_module("conv3", torch::nn::Conv2d(torch::nn::Conv2dOptions(1, 100, {8, 300}).stride(1).padding({7, 0})));

        dropout = register_module("dropout", torch::nn::Dropout(torch::nn::DropoutOptions(0.5)));
        bottleneck = register_module("bottleneck", torch::nn::Linear(3*100*8, 512)); // ks*output_channel*dynamic_pool_length, num_bottleneck_hidden
        fc1 = register_module("fc1", torch::nn::Linear(512, 90)); // num_bottleneck_hidden, target_class

        pool = register_module("pool", torch::nn::AdaptiveMaxPool1d(8)); // dynamic_pool_length
    }

    torch::Tensor forward(torch::Tensor input) {
        // Static Embedding
        auto x = embed(input);
        x = at::unsqueeze(x, 1);

        // Convolutional Layer
        auto x1 = at::squeeze(torch::relu(conv1(x)), 3);
        auto x2 = at::squeeze(torch::relu(conv2(x)), 3);
        auto x3 = at::squeeze(torch::relu(conv3(x)), 3);

        // Pooling results
        x1 = at::squeeze(pool(x1), 2);
        x2 = at::squeeze(pool(x2), 2);
        x3 = at::squeeze(pool(x3), 2);

        // Concatenate results
        x = at::_cat({x1, x2, x3}, 1);
        x = torch::relu(bottleneck(at::reshape(x, {-1, 3*100*8}))); // ks*output_channel*dynamic_pool_length, num_bottleneck_hidden
        x = dropout(x);
        x = fc1(x);

        return x;
    }

    // Module Layers
    torch::nn::Embedding embed{nullptr};
    torch::nn::Conv2d conv1{nullptr}, conv2{nullptr}, conv3{nullptr};
    torch::nn::Dropout dropout{nullptr}; 
    torch::nn::Linear bottleneck{nullptr}, fc1{nullptr};
    torch::nn::AdaptiveMaxPool1d pool{nullptr};
};
