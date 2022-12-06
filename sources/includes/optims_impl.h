#include "./optims_decl.h"

void Adam::step()
{
    for (Block *block_pt : target_module.block_pts)
    {
        for (Tensor *param_pt : block_pt->param_mat_pts)
        {
            // to be continued
        }
    }
}
