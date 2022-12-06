#include "./modules_impl.h"

class Adam : public Optimizer
{
private:
    Module &target_module;
    double learning_rate;
    double beta1;
    double beta2;
    double epsilon;

private:
    // to be continued

public:
    Adam(Module &target_module, double learning_rate, double beta1 = 0.9, double beta2 = 0.999, double epsilon = 1e-8)
        : target_module(target_module), learning_rate(learning_rate), beta1(beta1), beta2(beta2), epsilon(epsilon){}
    ~Adam() = default;

public:
    void step();
};