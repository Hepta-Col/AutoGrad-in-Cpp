#pragma once

#include <iostream>
#include <assert.h>
#include <math.h>
#include <vector>
#include <string>
#include <chrono>
#include <random>

#include "./parameters.h"

using namespace std;

/*------------------------------------------------------------------------------------------------------------*/

class Matrix_nn;
class Function;
class Tensor;
class Block;
class Module;
class Optimizer;

/*------------------------------------------------------------------------------------------------------------*/

class Matrix_nn
{
    friend class Tensor;

private:
    vector<vector<elem_t>> m_array;
    size_t m_height;
    size_t m_width;

public:
    Matrix_nn() { m_height = m_width = 0; }
    Matrix_nn(size_t height, size_t width) : m_height(height), m_width(width)
    {
        m_array = vector<vector<elem_t>>(height, vector<elem_t>(width, 0));
        xavier();
    }
    Matrix_nn(size_t height, size_t width, elem_t init_val) : m_height(height), m_width(width)
    {
        m_array = vector<vector<elem_t>>(height, vector<elem_t>(width, init_val));
    }
    Matrix_nn(vector<vector<elem_t>> array)
    {
        size_t height = array.size();
        size_t width = array[0].size();
        for (size_t i = 1; i < height; i++)
        {
            if (array[i].size() != width)
            {
                cout << "MatrixInitializeError: the initial array should be rectangle." << endl;
                exit(EXIT_FAILURE);
            }
        }
        m_array = array;
        m_height = height;
        m_width = width;
    }
    ~Matrix_nn() = default;

public:
    inline size_t getHeight() { return m_height; }
    inline size_t getWidth() { return m_width; }
    inline bool isScalar() { return m_height == 1 && m_width == 1; }
    inline pair<size_t, size_t> shape() { return make_pair(m_height, m_width); }

    void printInfo();

public:
    elem_t &operator()(size_t row, size_t col);
    Matrix_nn &operator=(vector<vector<elem_t>> array);
    Matrix_nn operator-(Matrix_nn &another);

public:
    Matrix_nn transpose();
    Matrix_nn matMul(Matrix_nn &another);
    Matrix_nn addScalar(elem_t addend);
    Matrix_nn pointWiseMultiply(elem_t scale);
    Matrix_nn pointWiseMultiply(Matrix_nn &another);
    Matrix_nn matAdd(Matrix_nn &another);

public:
    vector<Matrix_nn> sliceAndPile(size_t width);
    Matrix_nn concat(Matrix_nn &another);
    elem_t sum();

public:
    Matrix_nn softmax();
    Matrix_nn normalize(double epsilon);
    Matrix_nn positionEncode();
    Matrix_nn dropout(double probability);
    Matrix_nn activate();

private:
    void xavier();
};

class Tensor
{
    friend class MatMul;
    friend class MatAdd;
    friend class MatAddVec;
    friend class MatPointWiseMul;
    friend class MatSum;
    friend class MatScale;
    friend class Tanh;
    friend class MSE;

private:
    Matrix_nn *data_pt;
    Matrix_nn *grad_pt;
    Function *grad_func_pt;
    bool is_leaf;
    bool requires_grad;

public:
    Tensor()
        : data_pt(nullptr), grad_pt(nullptr), grad_func_pt(nullptr), is_leaf(true), requires_grad(false) {}
    Tensor(size_t height, size_t width, bool requires_grad)
        : grad_pt(nullptr), grad_func_pt(nullptr), is_leaf(true), requires_grad(requires_grad)
    {
        data_pt = new Matrix_nn(height, width);
    }
    Tensor(size_t height, size_t width, elem_t init_val, bool requires_grad)
        : grad_pt(nullptr), grad_func_pt(nullptr), is_leaf(true), requires_grad(requires_grad)
    {
        data_pt = new Matrix_nn(height, width, init_val);
    }
    Tensor(vector<vector<elem_t>> array, bool requires_grad)
        : grad_pt(nullptr), grad_func_pt(nullptr), is_leaf(true), requires_grad(requires_grad)
    {
        data_pt = new Matrix_nn(array);
    }
    Tensor(Tensor t, bool requires_grad)
        : grad_pt(nullptr), grad_func_pt(nullptr), is_leaf(true), requires_grad(requires_grad)
    {
        if (t.data_pt == nullptr)
            data_pt = nullptr;
        else
            data_pt = new Matrix_nn(*(t.data_pt));
    }
    ~Tensor() = default;

public:
    void printInfo(string tensor_name = "Default");

public:
    void accumulateGrad(Matrix_nn &grad_prod);
    void bp(Matrix_nn initial = Matrix_nn(1, 1, 1));
};

class Function
{
    friend class Tensor;

protected:
    string func_name;
    size_t operand_num;
    vector<Tensor *> operand_pts;
    vector<Matrix_nn *> gradient_pts;

public:
    Function() = default;
    ~Function() = default;

public:
    virtual void differentiate(Matrix_nn grad_chain) = 0;
};

/*------------------------------------------------------------------------------------------------------------*/

class Block
{
    friend class Adam;

protected:
    string block_name;
    vector<Tensor *> mid_tensor_pts; // should be "new"ed
    vector<Tensor *> param_mat_pts;  // should be "new"ed

public:
    Block() = default;
    ~Block() = default;
};

class Module
{
    friend class Adam;

protected:
    string module_name;
    vector<Tensor *> mid_tensor_pts; // should be "new"ed
    vector<Block *> block_pts;       // should be "new"ed

public:
    Module() = default;
    ~Module() = default;
};

/*------------------------------------------------------------------------------------------------------------*/

class Optimizer
{
protected:
    string optimizer_name;

public:
    Optimizer() = default;
    ~Optimizer() = default;

public:
    virtual void step() = 0;
};
