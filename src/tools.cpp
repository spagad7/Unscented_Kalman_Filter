#include <iostream>
#include "tools.h"

using Eigen::VectorXd;
using Eigen::MatrixXd;
using std::vector;

Tools::Tools() {}

Tools::~Tools() {}

// Function to calculate Root Mean Square Error
VectorXd Tools::CalculateRMSE(const vector<VectorXd> &estimations,
                              const vector<VectorXd> &ground_truth)
{
    VectorXd rmse(4);
    rmse << 0, 0, 0, 0;
    VectorXd temp(4);
    int n = estimations.size();
    if(n == 0)
        return rmse;

    for(int i=0; i<n; i++)
    {
        temp = estimations[i] - ground_truth[i];
        temp = temp.array() * temp.array();
        rmse += temp;
    }
    rmse = rmse/n;
    rmse = rmse.array().sqrt();

    return rmse;
}
