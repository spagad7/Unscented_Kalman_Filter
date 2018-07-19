#ifndef TOOLS_H_
#define TOOLS_H_
#include <vector>
#include "Dense"

using Eigen::MatrixXd;
using Eigen::VectorXd;
using namespace std;

class Tools {
public:
  /**
  * Constructor.
  */
  Tools();

  /**
  * Destructor.
  */
  virtual ~Tools();

  /**
  * A helper method to calculate RMSE.
  */
  VectorXd CalculateRMSE(const vector<VectorXd> &estimations, const vector<VectorXd> &ground_truth);

  /**
  * Function to calculate Normalized Innovation Squared (NIS) value
  */
  double CalculateNIS(VectorXd z, VectorXd z_pred, MatrixXd S);
};

#endif /* TOOLS_H_ */
