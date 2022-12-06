#include "./funcs_impl.h"

class Linear : public Block
{
private:
    size_t in_size;
    size_t out_size;
    Tensor *weights_pt;
    Tensor *bias_pt;

private:
    MatMul mul_weights;
    MatAddVec add_bias;

public:
    Linear(size_t in_size, size_t out_size, bool need_bias)
        : in_size(in_size), out_size(out_size)
    {
        block_name = "Linear";

        if (need_bias)
        {
            mid_tensor_pts.resize(1);
            param_mat_pts.resize(2);

            weights_pt = new Tensor(in_size, out_size, true);
            param_mat_pts[0] = weights_pt;
            bias_pt = new Tensor(1, out_size, true);
            param_mat_pts[1] = bias_pt;
        }
        else
        {
            mid_tensor_pts.resize(0);
            param_mat_pts.resize(1);

            weights_pt = new Tensor(in_size, out_size, true);
            param_mat_pts[0] = weights_pt;

            bias_pt = nullptr;
        }
    }
    ~Linear() = default;

public:
    Tensor forward(Tensor &input);
};
