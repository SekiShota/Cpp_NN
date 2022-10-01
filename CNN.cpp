// 損失関数とソルバーを先に指定
network<mse, adagrad> net;

// レイヤーを上から順に積む
net << convolutional_layer<tan_h>(32, 32, 5, 1, 6) // 32x32in, conv5x5, 1-6 f-maps
    << average_pooling_layer<relu>(28, 28, 6, 2) // 28x28in, 6 f-maps, pool2x2
    << fully_connected_layer<relu>(14 * 14 * 6, 120)
    << fully_connected_layer<softmax>(120, 10);

// 学習データのロード
std::vector<label_t> train_labels;
std::vector<vec_t> train_images;
parse_mnist_labels("train-labels.idx1-ubyte", &train_labels);
parse_mnist_images("train-images.idx3-ubyte", &train_images);

// 学習 (50-epoch, 30-minibatch)
net.train(train_images, train_labels, 30, 50);

// 重みを保存
std::ofstream ofs("weights");
ofs << net;
