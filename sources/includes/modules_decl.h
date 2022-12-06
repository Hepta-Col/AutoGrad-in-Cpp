#include "./blocks_impl.h"

class MLP_3 : public Module
{
private:
    size_t in_size;
    size_t hidden_size;
    size_t out_size;
    vector<Tanh> activations;

public:
    MLP_3(size_t in_size, size_t hidden_size, size_t out_size)
        : in_size(in_size), hidden_size(hidden_size), out_size(out_size)
    {
        module_name = "MLP";
        mid_tensor_pts.resize(5);
        block_pts.resize(3);
        activations.resize(3);

        block_pts[0] = new Linear(in_size, hidden_size, true);
        activations[0] = Tanh();
        block_pts[1] = new Linear(hidden_size, hidden_size, true);
        activations[1] = Tanh();
        block_pts[2] = new Linear(hidden_size, out_size, true);
        activations[2] = Tanh();
    }
    ~MLP_3() = default;

public:
    Tensor forward(Tensor &input);
};
