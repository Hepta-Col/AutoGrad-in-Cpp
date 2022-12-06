#pragma once

#include "./funcs_decl.h"

Tensor MatMul::apply(Tensor &a, Tensor &b)
{
    operand_pts[0] = &a;
    gradient_pts[0] = new Matrix_nn(b.data_pt->transpose());
    operand_pts[1] = &b;
    gradient_pts[1] = new Matrix_nn(a.data_pt->transpose());

    Tensor ret;
    Matrix_nn m_a = *(a.data_pt);
    Matrix_nn m_b = *(b.data_pt);
    ret.data_pt = new Matrix_nn(m_a.matMul(m_b));
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = a.requires_grad || b.requires_grad;
    return ret;
}

void MatMul::differentiate(Matrix_nn grad_chain)
{
    for (size_t op_ind = 0; op_ind < operand_num; op_ind++)
    {
        Tensor *pt_tensor = operand_pts[op_ind];
        assert(pt_tensor != nullptr);
        Matrix_nn gradient = *(gradient_pts[op_ind]);
        Matrix_nn grad_prod = op_ind == 0 ? grad_chain.matMul(gradient) : gradient.matMul(grad_chain);

        if (pt_tensor->grad_func_pt != nullptr)
            pt_tensor->grad_func_pt->differentiate(grad_prod);
        else
            pt_tensor->accumulateGrad(grad_prod);
    }
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MatAdd::apply(Tensor &a, Tensor &b)
{
    operand_pts[0] = &a;
    gradient_pts[0] = new Matrix_nn(1, 1, 1);
    operand_pts[1] = &b;
    gradient_pts[1] = new Matrix_nn(1, 1, 1);

    Tensor ret;
    Matrix_nn m_a = *(a.data_pt);
    Matrix_nn m_b = *(b.data_pt);
    ret.data_pt = new Matrix_nn(m_a.matAdd(m_b));
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = a.requires_grad || b.requires_grad;
    return ret;
}

void MatAdd::differentiate(Matrix_nn grad_chain)
{
    for (size_t op_ind = 0; op_ind < operand_num; op_ind++)
    {
        Tensor *pt_tensor = operand_pts[op_ind];
        assert(pt_tensor != nullptr);
        Matrix_nn gradient = *(gradient_pts[op_ind]);
        Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

        if (pt_tensor->grad_func_pt != nullptr)
            pt_tensor->grad_func_pt->differentiate(grad_prod);
        else
            pt_tensor->accumulateGrad(grad_prod);
    }
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MatAddVec::apply(Tensor &a, Tensor &b)
{
    assert(b.data_pt->getHeight() == 1 && a.data_pt->getWidth() == b.data_pt->getWidth());

    operand_pts[0] = &a;
    gradient_pts[0] = new Matrix_nn(1, 1, 1);
    operand_pts[1] = &b;
    gradient_pts[1] = new Matrix_nn(1, 1, 1);

    Tensor ret;
    ret.data_pt = new Matrix_nn(*(a.data_pt));
    for (size_t row = 0; row < a.data_pt->getHeight(); row++)
        for (size_t ind = 0; ind < a.data_pt->getWidth(); ind++)
            (*(ret.data_pt))(row, ind) += (*(b.data_pt))(0, ind);

    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = a.requires_grad || b.requires_grad;
    return ret;
}

void MatAddVec::differentiate(Matrix_nn grad_chain)
{
    for (size_t op_ind = 0; op_ind < operand_num; op_ind++)
    {
        Tensor *pt_tensor = operand_pts[op_ind];
        assert(pt_tensor != nullptr);
        Matrix_nn gradient = *(gradient_pts[op_ind]);
        Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

        if (op_ind == 1)
            assert(pt_tensor->is_leaf && pt_tensor->grad_func_pt == nullptr);

        if (pt_tensor->grad_func_pt != nullptr)
            pt_tensor->grad_func_pt->differentiate(grad_prod);
        else
            pt_tensor->accumulateGrad(grad_prod);
    }
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MatPointWiseMul::apply(Tensor &a, Tensor &b)
{
    operand_pts[0] = &a;
    gradient_pts[0] = new Matrix_nn(*(b.data_pt));
    operand_pts[1] = &b;
    gradient_pts[1] = new Matrix_nn(*(a.data_pt));

    Tensor ret;
    Matrix_nn m_a = *(a.data_pt);
    Matrix_nn m_b = *(b.data_pt);
    ret.data_pt = new Matrix_nn(m_a.pointWiseMultiply(m_b));
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = a.requires_grad || b.requires_grad;
    return ret;
}

void MatPointWiseMul::differentiate(Matrix_nn grad_chain)
{
    for (size_t op_ind = 0; op_ind < operand_num; op_ind++)
    {
        Tensor *pt_tensor = operand_pts[op_ind];
        assert(pt_tensor != nullptr);
        Matrix_nn gradient = *(gradient_pts[op_ind]);
        Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

        if (pt_tensor->grad_func_pt != nullptr)
            pt_tensor->grad_func_pt->differentiate(grad_prod);
        else
            pt_tensor->accumulateGrad(grad_prod);
    }
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MatSum::apply(Tensor &x)
{
    operand_pts[0] = &x;
    gradient_pts[0] = new Matrix_nn(x.data_pt->getHeight(), x.data_pt->getWidth(), 1);

    Tensor ret(vector<vector<elem_t>>(1, vector<elem_t>(1, x.data_pt->sum())), x.requires_grad);
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = x.requires_grad;
    return ret;
}

void MatSum::differentiate(Matrix_nn grad_chain)
{
    Tensor *pt_tensor = operand_pts[0];
    assert(pt_tensor != nullptr);
    Matrix_nn gradient = *(gradient_pts[0]);

    Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);
    if (pt_tensor->grad_func_pt != nullptr)
        pt_tensor->grad_func_pt->differentiate(grad_prod);
    else
        pt_tensor->accumulateGrad(grad_prod);
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MatScale::apply(Tensor &x, elem_t scale)
{
    operand_pts[0] = &x;
    gradient_pts[0] = new Matrix_nn(1, 1, scale);

    Tensor ret(x.data_pt->getHeight(), x.data_pt->getWidth(), 0, x.requires_grad);
    ret.data_pt = new Matrix_nn(x.data_pt->pointWiseMultiply(scale));
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = x.requires_grad;
    return ret;
}

void MatScale::differentiate(Matrix_nn grad_chain)
{
    Tensor *pt_tensor = operand_pts[0];
    assert(pt_tensor != nullptr);
    Matrix_nn gradient = *(gradient_pts[0]);
    Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

    if (pt_tensor->grad_func_pt != nullptr)
        pt_tensor->grad_func_pt->differentiate(grad_prod);
    else
        pt_tensor->accumulateGrad(grad_prod);
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor Tanh::apply(Tensor &x)
{
    operand_pts[0] = &x;

    Matrix_nn ret_mat(*(x.data_pt));
    for (size_t i = 0; i < ret_mat.getHeight(); i++)
        for (size_t j = 0; j < ret_mat.getWidth(); j++)
            ret_mat(i, j) = tanh(ret_mat(i, j));

    Matrix_nn grad(x.data_pt->getHeight(), x.data_pt->getWidth(), 0);
    for (size_t i = 0; i < grad.getHeight(); i++)
        for (size_t j = 0; j < grad.getWidth(); j++)
            grad(i, j) = 1 - ret_mat(i, j) * ret_mat(i, j);
    gradient_pts[0] = new Matrix_nn(grad);

    Tensor ret;
    ret.data_pt = new Matrix_nn(ret_mat);
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = x.requires_grad;
    return ret;
}

void Tanh::differentiate(Matrix_nn grad_chain)
{
    Tensor *pt_tensor = operand_pts[0];
    assert(pt_tensor != nullptr);
    Matrix_nn gradient = *(gradient_pts[0]);
    Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

    if (pt_tensor->grad_func_pt != nullptr)
        pt_tensor->grad_func_pt->differentiate(grad_prod);
    else
        pt_tensor->accumulateGrad(grad_prod);
}

/*------------------------------------------------------------------------------------------------------------*/

Tensor MSE::apply(Tensor &output, Tensor &target)
{
    operand_pts[0] = &output;
    gradient_pts[0] = new Matrix_nn(*(output.data_pt) - *(target.data_pt));
    operand_pts[1] = &target;
    gradient_pts[1] = new Matrix_nn(*(target.data_pt) - *(output.data_pt));

    Matrix_nn diff = *(output.data_pt) - *(target.data_pt);

    Matrix_nn ret_mat(output.data_pt->getHeight(), output.data_pt->getWidth(), 0);
    for (size_t i = 0; i < output.data_pt->getHeight(); i++)
        for (size_t j = 0; j < output.data_pt->getWidth(); j++)
            ret_mat(i, j) = (diff(i, j) * diff(i, j)) / 2.;

    Tensor ret;
    ret.data_pt = new Matrix_nn(ret_mat);
    ret.grad_pt = nullptr;
    ret.grad_func_pt = this;
    ret.is_leaf = false;
    ret.requires_grad = output.requires_grad || target.requires_grad;
    return ret;
}

void MSE::differentiate(Matrix_nn grad_chain)
{
    for (size_t op_ind = 0; op_ind < operand_num; op_ind++)
    {
        Tensor *pt_tensor = operand_pts[op_ind];
        assert(pt_tensor != nullptr);
        Matrix_nn gradient = *(gradient_pts[op_ind]);
        Matrix_nn grad_prod = grad_chain.pointWiseMultiply(gradient);

        if (pt_tensor->grad_func_pt != nullptr)
            pt_tensor->grad_func_pt->differentiate(grad_prod);
        else
            pt_tensor->accumulateGrad(grad_prod);
    }
}
