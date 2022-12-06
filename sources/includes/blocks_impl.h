#include "./blocks_decl.h"

Tensor Linear::forward(Tensor &input)
{
    Tensor flow = mul_weights.apply(input, *weights_pt);

    if (bias_pt != nullptr)
    {
        mid_tensor_pts[0] = new Tensor(flow);
        flow = add_bias.apply(*(mid_tensor_pts[0]), *bias_pt);
    }
    return flow;
}
