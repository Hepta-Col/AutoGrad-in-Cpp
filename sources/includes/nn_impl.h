#pragma once

#include "./nn_decl.h"

/*------------------------------------------------------------------------------------------------------------*/

void Matrix_nn::printInfo()
{
    // cout << endl
    //      << "--- Print Matrix ---" << endl
    //      << endl;
    cout << "--Height: " << m_height << endl;
    cout << "--Width: " << m_width << endl;
    cout << "--Matrix: " << endl;
    for (size_t i = 0; i < m_height; i++)
    {
        cout << "  [ ";
        for (size_t j = 0; j < m_width; j++)
        {
            cout << m_array[i][j] << ", ";
        }
        cout << "]" << endl;
    }
    // cout << endl
    //      << "--- End Print ---" << endl
    //      << endl;
}

elem_t &Matrix_nn::operator()(size_t row, size_t col)
{
    assert(row >= 0 && row < m_height && col >= 0 && col < m_width);
    return m_array[row][col];
}

Matrix_nn &Matrix_nn::operator=(vector<vector<elem_t>> array)
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

    return *this;
}

Matrix_nn Matrix_nn::operator-(Matrix_nn &another)
{
    assert(m_height == another.getHeight() && m_width == another.getWidth());
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] -= another.m_array[i][j];
    return ret;
}

Matrix_nn Matrix_nn::transpose()
{
    Matrix_nn ret(m_width, m_height, 0);
    for (size_t i = 0; i < m_width; i++)
        for (size_t j = 0; j < m_height; j++)
            ret.m_array[i][j] = m_array[j][i];
    return ret;
}

Matrix_nn Matrix_nn::matMul(Matrix_nn &another)
{
    assert(m_width == another.m_height);

    size_t target_height = m_height;
    size_t target_width = another.m_width;
    size_t middle = m_width;

    Matrix_nn ret(target_height, target_width, 0);

    for (size_t i = 0; i < target_height; i++)
    {
        for (size_t j = 0; j < target_width; j++)
        {
            elem_t temp = 0;
            for (size_t k = 0; k < middle; k++)
                temp += m_array[i][k] * another.m_array[k][j];
            ret.m_array[i][j] = temp;
        }
    }

    return ret;
}

Matrix_nn Matrix_nn::addScalar(elem_t addend)
{
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] += addend;
    return ret;
}

Matrix_nn Matrix_nn::pointWiseMultiply(elem_t scale)
{
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] *= scale;
    return ret;
}

Matrix_nn Matrix_nn::pointWiseMultiply(Matrix_nn &another)
{
    if (another.isScalar() && !this->isScalar())
        return this->pointWiseMultiply(another.m_array[0][0]);
    else if (!another.isScalar() && this->isScalar())
        return another.pointWiseMultiply(m_array[0][0]);

    assert(m_height == another.m_height && m_width == another.m_width);
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] *= another.m_array[i][j];
    return ret;
}

Matrix_nn Matrix_nn::matAdd(Matrix_nn &another)
{
    if (m_height == 0 && m_width == 0)
        return another;

    if (another.isScalar())
        return this->addScalar(another.m_array[0][0]);

    if (this->isScalar())
        return another.addScalar(m_array[0][0]);

    assert(m_height == another.m_height && m_width == another.m_width);
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] += another.m_array[i][j];
    return ret;
}

vector<Matrix_nn> Matrix_nn::sliceAndPile(size_t width)
{
    assert(width > 0 && m_width % width == 0);

    vector<Matrix_nn> ret;
    ret.resize(m_width / width);
    Matrix_nn temp(m_height, width);

    for (size_t i = 0; i < m_width; i++)
    {
        for (size_t row = 0; row < m_height; row++)
            temp.m_array[row][i % width] = m_array[row][i];

        if ((i + 1) % width == 0)
            ret[i / width] = temp;
    }

    return ret;
}

Matrix_nn Matrix_nn::concat(Matrix_nn &another)
{
    if (m_height == 0 && m_width == 0)
    {
        Matrix_nn ret(another);
        return ret;
    }
    else
    {
        assert(m_height == another.m_height);

        int width1 = m_width;
        int width2 = another.m_width;

        int new_height = m_height;
        int new_width = width1 + width2;

        Matrix_nn ret(new_height, new_width, 0);

        for (size_t i = 0; i < new_height; i++)
        {
            for (size_t j = 0; j < new_width; j++)
            {
                if (j < width1)
                    ret.m_array[i][j] = m_array[i][j];
                else
                    ret.m_array[i][j] = another(i, j - width1);
            }
        }
        return ret;
    }
}

elem_t Matrix_nn::sum()
{
    elem_t sum = 0;
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            sum += m_array[i][j];
    return sum;
}

Matrix_nn Matrix_nn::softmax()
{
    Matrix_nn ret(m_height, m_width, 0);
    vector<double> expo_sum(m_height, 0);

    for (size_t i = 0; i < m_height; i++)
    {
        for (size_t j = 0; j < m_width; j++)
        {
            double temp = exp(m_array[i][j]);
            ret.m_array[i][j] = (elem_t)temp;
            expo_sum[i] += temp;
        }
    }

    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] /= (elem_t)expo_sum[i];

    return ret;
}

Matrix_nn Matrix_nn::normalize(double epsilon)
{
    Matrix_nn ret(m_height, m_width, 0);
    double mean, standard, temp_sum = 0;

    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            temp_sum += m_array[i][j];

    mean = temp_sum / (m_height * m_width);

    temp_sum = 0;

    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            temp_sum += pow(m_array[i][j] - mean, 2);

    standard = sqrt(temp_sum / (m_height * m_width));

    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] = (m_array[i][j] - (elem_t)mean) / ((elem_t)standard + (elem_t)epsilon);

    return ret;
}

Matrix_nn Matrix_nn::positionEncode()
{
    Matrix_nn ret(m_height, m_width, 0);
    if (m_width % 2 == 0)
    {
        for (size_t i = 0; i < m_height; i++)
        {
            for (size_t j = 0; j < m_width; j += 2)
            {
                ret.m_array[i][j] = (elem_t)sin(i / pow(10000, 2 * j / m_width));
                ret.m_array[i][j + 1] = (elem_t)cos(i / pow(10000, (2 * j + 1) / m_width));
            }
        }
    }
    else
    {
        for (size_t i = 0; i < m_height; i++)
        {
            for (size_t j = 0; j < m_width - 1; j += 2)
            {
                ret.m_array[i][j] = (elem_t)sin(i / pow(10000, 2 * j / m_width));
                ret.m_array[i][j + 1] = (elem_t)cos(i / pow(10000, (2 * j + 1) / m_width));
            }
            ret.m_array[i][m_width - 1] = (elem_t)sin(i / pow(10000, 2 * (m_width - 1) / m_width));
        }
    }

    return ret;
}

Matrix_nn Matrix_nn::dropout(double probability)
{
    Matrix_nn ret(*this);
    for (size_t i = 0; i < m_height; i++)
    {
        for (size_t j = 0; j < m_width; j++)
        {
            int seed = chrono::system_clock::now().time_since_epoch().count();
            default_random_engine engine(seed);
            uniform_real_distribution<double> distribution(0, 1);

            int x = distribution(engine);
            if (x < probability)
                ret.m_array[i][j] = 0;
        }
    }
    return ret;
}

Matrix_nn Matrix_nn::activate()
{
    Matrix_nn ret(m_height, m_width);
    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            ret.m_array[i][j] = (elem_t)tanh(m_array[i][j]);

    return ret;
}

void Matrix_nn::xavier()
{
    double bound = sqrt(6. / (m_height + m_width));
    int seed = chrono::system_clock::now().time_since_epoch().count();
    default_random_engine engine(seed);
    uniform_real_distribution<elem_t> distribution(-bound, bound);

    for (size_t i = 0; i < m_height; i++)
        for (size_t j = 0; j < m_width; j++)
            m_array[i][j] = distribution(engine);
}

/*------------------------------------------------------------------------------------------------------------*/

void Tensor::printInfo(string tensor_name)
{
    cout << "(Tensor " << tensor_name << ")" << endl;
    cout << "-------------------" << endl;
    cout << "Data: " << endl;
    data_pt->printInfo();

    cout << endl;

    cout << "Grad: " << endl;
    if (!grad_pt)
        cout << "None" << endl;
    else
        grad_pt->printInfo();

    cout << endl;

    cout << "Grad_func: " << endl;
    if (!grad_func_pt)
        cout << "None" << endl;
    else
        cout << "<" << grad_func_pt->func_name << ">" << endl;

    cout << endl;

    cout << "Is_leaf: " << endl;
    if (is_leaf)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;

    cout << endl;

    cout << "Requires_grad: " << endl;
    if (requires_grad)
        cout << "Yes" << endl;
    else
        cout << "No" << endl;
    cout << "-------------------" << endl;
    cout << endl
         << endl;
}

void Tensor::accumulateGrad(Matrix_nn &grad_prod)
{
    if (grad_pt == nullptr)
        grad_pt = new Matrix_nn(grad_prod);
    else
    {
        Matrix_nn temp = grad_pt->matAdd(grad_prod);
        grad_pt = new Matrix_nn(temp);
    }
}

void Tensor::bp(Matrix_nn initial)
{
    assert(grad_func_pt != nullptr);
    assert(!is_leaf);
    assert(requires_grad);

    assert(initial.shape() == data_pt->shape());

    grad_func_pt->differentiate(initial);
}

/*------------------------------------------------------------------------------------------------------------*/
