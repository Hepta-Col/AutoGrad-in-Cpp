#include "autograd.h"

void test01()
{
    /*Multilayer Perceptron with 3 layers*/
    // input dimension, hidden dimension, output dimension
    MLP_3 mlp(3, 5, 3);

    /*Sum all losses*/
    MatSum msm;

    /*Min Square Error*/
    MSE mse;

    Tensor y(
        {
            {1, 2, 3},
            {4, 5, 6},
        },
        true);

    Tensor z = mlp.forward(y);

    Tensor target(
        {
            {1, 4, 9},
            {16, 25, 36},
        },
        false);

    Tensor loss = mse.apply(z, target);

    Tensor out = msm.apply(loss);

    out.bp();   // back propagation

    y.printInfo("y");   // should has a grad matrix
    z.printInfo("z");

    loss.printInfo("loss");
    out.printInfo("out");
}

void test02()
{
    MatMul mm;

    Tensor x(
        {
            {1, 2, 3},
            {4, 5, 6},
        },
        true);

    Tensor y(
        {
            {0.1, 0.2},
            {0.4, 0.5},
            {0.7, 0.8},
        },
        true);

    Tensor z = mm.apply(x, y);
    z.printInfo("z");

    x.printInfo("x");
    y.printInfo("y");
}

int main()
{
    test01();

    return 0;
}
