#include <boost/numeric/ublas/vector.hpp>
#include <boost/numeric/ublas/matrix.hpp>
#include <boost/numeric/ublas/io.hpp>
#include "Dataset.hpp"


using namespace boost::numeric::ublas;

int main (void) {
    //絶対パスを入力してください。
    Mnist mnist;
    mnist.readTrainingFile("/train-images-idx3-ubyte");
    mnist.readLabelFile("train-labels-idx1-ubyte");
}
