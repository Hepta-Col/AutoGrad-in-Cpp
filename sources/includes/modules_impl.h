#include "./modules_decl.h"

Tensor MLP_3::forward(Tensor &input)
{
    Linear *input_layer_pt = static_cast<Linear *>(block_pts[0]);
    Linear *hidden_layer_pt = static_cast<Linear *>(block_pts[1]);
    Linear *output_layer_pt = static_cast<Linear *>(block_pts[2]);

    Tensor flow;

    flow = input_layer_pt->forward(input);
    mid_tensor_pts[0] = new Tensor(flow);
    flow = activations[0].apply(*(mid_tensor_pts[0]));
    mid_tensor_pts[1] = new Tensor(flow);

    flow = hidden_layer_pt->forward(*(mid_tensor_pts[1]));
    mid_tensor_pts[2] = new Tensor(flow);
    flow = activations[1].apply(*(mid_tensor_pts[2]));
    mid_tensor_pts[3] = new Tensor(flow);

    flow = output_layer_pt->forward(*(mid_tensor_pts[3]));
    mid_tensor_pts[4] = new Tensor(flow);
    flow = activations[2].apply(*(mid_tensor_pts[4]));

    return flow;
}
