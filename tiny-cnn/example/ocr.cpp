#include <iostream>
#include <opencv2/core/core.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/highgui/highgui.hpp>
#include "tiny_cnn.h"

using namespace tiny_cnn;
using namespace tiny_cnn::activation;
using namespace std;

typedef unsigned char byte;

// rescale output to 0-100
template <typename Activation>
double rescale(double x) {
    Activation a;
    return 100.0 * (x - a.scale().first) / (a.scale().second - a.scale().first);
}

// convert tiny_cnn::image to cv::Mat and resize
cv::Mat image2mat(image<byte>& img) {
    cv::Mat ori(img.height(), img.width(), CV_8U, &img.at(0, 0));
    cv::Mat resized;
    cv::resize(ori, resized, cv::Size(), 3, 3, cv::INTER_AREA);
    return resized;
}

void convert_image(const std::string& imagefilename,
    double minv,
    double maxv,
    int w,
    int h,
    vec_t& data)
{
    auto img = cv::imread(imagefilename, cv::IMREAD_GRAYSCALE);
    if (img.data == nullptr) return; // cannot open, or it's not an image

    cv::Mat_<uint8_t> resized;
    cv::resize(img, resized, cv::Size(w, h));

    // mnist dataset is "white on black", so negate required
    std::transform(resized.begin(), resized.end(), std::back_inserter(data),
        [=](uint8_t c) { return (255 - c) * (maxv - minv) / 255.0 + minv; });
}

void recognize(const std::string& dictionary, const std::string& filename) {
    network<mse, gradient_descent_levenberg_marquardt> nn; // specify loss-function and learning strategy

#define O true
#define X false
    static const bool connection [] = {
        O, X, X, X, O, O, O, X, X, O, O, O, O, X, O, O,
        O, O, X, X, X, O, O, O, X, X, O, O, O, O, X, O,
        O, O, O, X, X, X, O, O, O, X, X, O, X, O, O, O,
        X, O, O, O, X, X, O, O, O, O, X, X, O, X, O, O,
        X, X, O, O, O, X, X, O, O, O, O, X, O, O, X, O,
        X, X, X, O, O, O, X, X, O, O, O, O, X, O, O, O
    };
#undef O
#undef X

    nn << convolutional_layer<tan_h>(32, 32, 5, 1, 6)
        << average_pooling_layer<tan_h>(28, 28, 6, 2)
        << convolutional_layer<tan_h>(14, 14, 5, 6, 16, connection_table(connection, 6, 16))
        << average_pooling_layer<tan_h>(10, 10, 16, 2)
        << convolutional_layer<tan_h>(5, 5, 5, 16, 120)
        << fully_connected_layer<tan_h>(120, 10);

    // load nets
    ifstream ifs(dictionary.c_str());
    ifs >> nn;

    // convert imagefile to vec_t
    vec_t data;
    convert_image(filename, -1.0, 1.0, 32, 32, data);

    // recognize
    auto res = nn.predict(data);
    vector<pair<double, int> > scores;

    // sort & print top-3
    for (int i = 0; i < 10; i++)
        scores.emplace_back(rescale<tan_h>(res[i]), i);

    sort(scores.begin(), scores.end(), greater<pair<double, int>>());

    for (int i = 0; i < 3; i++)
        cout << scores[i].second << "," << scores[i].first << endl;

    // visualize outputs of each layer
    for (size_t i = 0; i < nn.depth(); i++) {
        auto out_img = nn[i]->output_to_image();
        cv::imshow("layer:" + std::to_string(i), image2mat(out_img));
    }
    // visualize filter shape of first convolutional layer
    auto weight = nn.at<convolutional_layer<tan_h>>(0).weight_to_image();
    cv::imshow("weights:", image2mat(weight));

    cv::waitKey(0);
}

int main(int argc, char** argv) {
    if (argc != 3) {
        cout << "Usage: ./ocr <weights> <image>" << endl;
        return 0;
    }
    recognize(argv[1], argv[2]);
}
