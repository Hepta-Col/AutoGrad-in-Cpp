#pragma once

#include "./nn_impl.h"

class MatMul : public Function
{
public:
    MatMul()
    {
        func_name = "MatMul";
        operand_num = 2;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        for (size_t i = 0; i < operand_num; i++)
        {
            operand_pts[i] = nullptr;
            gradient_pts[i] = nullptr;
        }
    }
    ~MatMul() = default;

public:
    Tensor apply(Tensor &a, Tensor &b);
    void differentiate(Matrix_nn grad_chain);
};

class MatAdd : public Function
{
public:
    MatAdd()
    {
        func_name = "MatAdd";
        operand_num = 2;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        for (size_t i = 0; i < operand_num; i++)
        {
            operand_pts[i] = nullptr;
            gradient_pts[i] = nullptr;
        }
    }
    ~MatAdd() = default;

public:
    Tensor apply(Tensor &a, Tensor &b);
    void differentiate(Matrix_nn grad_chain);
};

class MatAddVec : public Function
{
public:
    MatAddVec()
    {
        func_name = "MatAddVec";
        operand_num = 2;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        for (size_t i = 0; i < operand_num; i++)
        {
            operand_pts[i] = nullptr;
            gradient_pts[i] = nullptr;
        }
    }
    ~MatAddVec() = default;

public:
    Tensor apply(Tensor &a, Tensor &b);
    void differentiate(Matrix_nn grad_chain);
};

class MatPointWiseMul : public Function
{
public:
    MatPointWiseMul()
    {
        func_name = "MatPointWiseMul";
        operand_num = 2;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        for (size_t i = 0; i < operand_num; i++)
        {
            operand_pts[i] = nullptr;
            gradient_pts[i] = nullptr;
        }
    }
    ~MatPointWiseMul() = default;

public:
    Tensor apply(Tensor &a, Tensor &b);
    void differentiate(Matrix_nn grad_chain);
};

class MatSum : public Function
{
public:
    MatSum()
    {
        func_name = "MatSum";
        operand_num = 1;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        operand_pts[0] = nullptr;
        gradient_pts[0] = nullptr;
    }
    ~MatSum() = default;

public:
    Tensor apply(Tensor &x);
    void differentiate(Matrix_nn grad_chain);
};

class MatScale : public Function
{
public:
    MatScale()
    {
        func_name = "MatScale";
        operand_num = 1;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        operand_pts[0] = nullptr;
        gradient_pts[0] = nullptr;
    }
    ~MatScale() = default;

public:
    Tensor apply(Tensor &x, elem_t scale);
    void differentiate(Matrix_nn grad_chain);
};

class Tanh : public Function
{
public:
    Tanh()
    {
        func_name = "Activation(tanh)";
        operand_num = 1;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        operand_pts[0] = nullptr;
        gradient_pts[0] = nullptr;
    }
    ~Tanh() = default;

public:
    Tensor apply(Tensor &x);
    void differentiate(Matrix_nn grad_chain);
};

class MSE : public Function
{
public:
    MSE()
    {
        func_name = "Loss(MSE)";
        operand_num = 2;
        operand_pts.resize(operand_num);
        gradient_pts.resize(operand_num);
        for (size_t i = 0; i < operand_num; i++)
        {
            operand_pts[i] = nullptr;
            gradient_pts[i] = nullptr;
        }
    }
    ~MSE() = default;

public:
    Tensor apply(Tensor &output, Tensor &target);
    void differentiate(Matrix_nn grad_chain);
};
