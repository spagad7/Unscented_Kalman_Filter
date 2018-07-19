#include "ukf.h"
#include "Dense"
#include <iostream>
#include "tools.h"
#include <fstream>

using namespace std;
using Eigen::MatrixXd;
using Eigen::VectorXd;
using std::vector;

/**
 * Initializes Unscented Kalman filter
 * This is scaffolding, do not modify
 */
UKF::UKF() {
    // if this is false, laser measurements will be ignored (except during init)
    use_laser_ = true;
    // if this is false, radar measurements will be ignored (except during init)
    use_radar_ = true;
    // initial state vector
    x_ = VectorXd(5);
    // initial covariance matrix
    P_ = MatrixXd(5, 5);
    // Process noise standard deviation longitudinal acceleration in m/s^2
    std_a_ = 1.5;
    // Process noise standard deviation yaw acceleration in rad/s^2
    std_yawdd_ = 0.3;
    // Laser measurement noise standard deviation position1 in m
    std_laspx_ = 0.15;
    // Laser measurement noise standard deviation position2 in m
    std_laspy_ = 0.15;
    // Radar measurement noise standard deviation radius in m
    std_radr_ = 0.3;
    // Radar measurement noise standard deviation angle in rad
    std_radphi_ = 0.03;
    // Radar measurement noise standard deviation radius change in m/s
    std_radrd_ = 0.3;
    // Initialize state vector
    x_ << 0, 0, 0, 0, 0;
    // Initialize state covariance matrix
    P_ << MatrixXd::Identity(5, 5);
    // Dimension of state vector
    n_x_ = 5;
    // Dimension of augmented sigma points
    n_aug_ = 7;
    // Initialize predicted sigma points
    x_sig_pred_ = MatrixXd(5, 15);
    // lambda value
    lambda_ = 3 - n_aug_;
    // init flag
    is_initialized_ = false;
    // set nis flag for radar
    nis_radar_ = true;
    // set nis flag for lidar
    nis_lidar_ = true;
    // time keeper for nis graph
    dt_ = 0.0;

    // Generate weights vector to calculate mean and covariance of predicted sigma points
    weights_ = VectorXd(2*n_aug_+1);
    weights_(0) = lambda_ / (lambda_ + n_aug_);
    for(int i=1; i<2*n_aug_+1; ++i) {
        weights_(i) = 0.5 / (lambda_ + n_aug_);
    }

    // Initialize measurement function for LIDAR update
    H_ = MatrixXd(2, 5);
    H_ <<   1, 0, 0, 0, 0,
            0, 1, 0, 0, 0;

    // Initialize measurement noise matrix
    R_lidar = MatrixXd(2, 2);
    R_lidar <<  std_laspx_*std_laspx_,  0,
                0,                      std_laspy_*std_laspy_;

    // Initialize measurement noise matrix
    R_radar = MatrixXd(3, 3);
    R_radar <<  std_radr_*std_radr_,  0,                        0,
                0,                    std_radphi_*std_radphi_,  0,
                0,                    0,                        std_radrd_*std_radrd_;

    if(nis_radar_) {
        std::string file_name = "../nis_radar.csv";
        fs_rdr_.open(file_name, std::ios::out);
        if(!fs_rdr_.is_open()){
            std::cerr << "Error opening file: " << file_name << std::endl;
            exit(-1);
        }
    }

    if(nis_lidar_) {
        std::string file_name = "../nis_lidar.csv";
        fs_ldr_.open(file_name, std::ios::out);
        if(!fs_ldr_.is_open()){
            std::cerr << "Error opening file: " << file_name << std::endl;
            exit(-1);
        }
    }
}


/**
* Destructor
*/
UKF::~UKF() {
    if(nis_radar_)
        fs_rdr_.close();
    if(nis_lidar_)
        fs_ldr_.close();
}


/**
 * Processes measurements received from LIDAR/RADAR
 * @param {MeasurementPackage} meas_package The latest measurement data of
 * either radar or laser.
 */
void UKF::ProcessMeasurement(MeasurementPackage meas_package) {
    // intialize the state vector if this is the first measurement
    if(!is_initialized_) {
        time_us_ = meas_package.timestamp_;
        if(meas_package.sensor_type_ == MeasurementPackage::RADAR) {
            float rho = meas_package.raw_measurements_(0);
            float theta = meas_package.raw_measurements_(1);
            x_ << rho * cos(theta), rho * sin(theta), 0, 0, 0;
        }
        else if(meas_package.sensor_type_ == MeasurementPackage::LASER) {
            double px = meas_package.raw_measurements_(0);
            double py = meas_package.raw_measurements_(1);
            x_ << px, py, 0, 0, 0;
        }
        is_initialized_ = true;
        return;
    }

    // calculate delta_t and call predict function to predict state vector and the measurements
    double delta_t = double(meas_package.timestamp_ - time_us_) / 1000000.0;
    dt_ += delta_t;
    time_us_ = meas_package.timestamp_;

    if(meas_package.sensor_type_ == MeasurementPackage::LASER && use_laser_) {
        // Prediction step
        Prediction(delta_t);
        // Update function for LIDAR measurement
        UpdateLidar(meas_package);
        } else if (meas_package.sensor_type_ == MeasurementPackage::RADAR && use_radar_){
        // Prediction step
        Prediction(delta_t);
        // Update function for LIDAR measurement
        UpdateRadar(meas_package);
    }
}


/**
 * Predicts sigma points, the state, and the state covariance matrix.
 * @param {double} delta_t the change in time (in seconds) between the last
 * measurement and this one.
 */
void UKF::Prediction(double delta_t) {
    // Matrix to store sigma points
    MatrixXd x_sig_aug = MatrixXd(n_aug_, 2*n_aug_+1);

    // Generation of sigma points
    // augment state vector with longitudinal acceleration and yaw acceleration noise
    VectorXd x_aug = VectorXd(n_aug_);
    x_aug.block<5, 1>(0, 0) = x_;
    x_aug(5) = 0;
    x_aug(6) = 0;

    // calculate Q matrix
    MatrixXd Q = MatrixXd(2, 2);
    Q <<    std_a_*std_a_,  0,
            0,              std_yawdd_*std_yawdd_;

    // Generate augmented P matrix
    MatrixXd P_aug = MatrixXd::Zero(n_aug_, n_aug_);
    P_aug.block<5, 5>(0, 0) = P_;
    P_aug.block<2, 2>(5, 5) = Q;

    // calculate square root of scaled, augmented state covariance matrix
    double scaling_factor = lambda_ + n_aug_;
    MatrixXd P_aug_sq = scaling_factor * P_aug;
    P_aug_sq = P_aug_sq.llt().matrixL();

    // generate sigma points
    x_sig_aug.col(0) = x_aug;
    for(int i=1; i<2*n_aug_+1; ++i) {
        if(i<n_aug_+1)
            x_sig_aug.col(i) = x_aug + P_aug_sq.col(i-1);
        else
            x_sig_aug.col(i) = x_aug - P_aug_sq.col(i-n_aug_-1);
    }

    // Prediction of sigma points
    for(int i=0; i<2*n_aug_+1; ++i) {
        double v = x_sig_aug(2, i);
        double psi = x_sig_aug(3, i);
        double psi_dot = x_sig_aug(4, i);
        double nu_a = x_sig_aug(5, i);
        double nu_psi = x_sig_aug(6, i);

        VectorXd x_tmp = VectorXd(5);
        // if yaw is zero, avoid division by zero
        if(fabs(psi_dot) < 0.001 ) {
            x_tmp <<    v * sin(psi) * delta_t,
                        v * cos(psi) * delta_t,
                        0,
                        0,
                        0;
        } else {
            x_tmp <<    (v/psi_dot) * (sin(psi + psi_dot * delta_t) - sin(psi)),
                        (v/psi_dot) * (cos(psi) - cos(psi + psi_dot * delta_t)),
                        0,
                        psi_dot*delta_t,
                        0;
        }

        // calculate noise vector
        VectorXd x_noise = VectorXd(5);
        x_noise <<  0.5 * delta_t * delta_t * cos(psi) * nu_a,
                    0.5 * delta_t * delta_t * sin(psi) * nu_a,
                    delta_t * nu_a,
                    0.5 * delta_t * delta_t * nu_psi,
                    delta_t * nu_psi;

        // predict sigma points
        x_sig_pred_.col(i) = x_sig_aug.block<5, 1>(0, i) + x_tmp + x_noise;
    }

    // Calculate mean state vector from predicted sigma points
    x_.fill(0.0);
    for(int i=0; i<2*n_aug_+1; ++i) {
        x_ = x_ + weights_(i) * x_sig_pred_.col(i);
    }

    P_.fill(0.0);
    // Calculate state covariance matrix from predicted sigma points
    for(int i=0; i<2*n_aug_+1; ++i) {
        VectorXd diff = x_sig_pred_.col(i) - x_;
        // normalize angle
        while(diff(3) > M_PI)
            diff(3) = diff(3) - 2.*M_PI;
        while(diff(3) < -M_PI)
            diff(3) = diff(3) + 2.*M_PI;
        P_ = P_ + weights_(i) * (diff * diff.transpose());
    }
}


/**
 * Updates the state and the state covariance matrix using a laser measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateLidar(MeasurementPackage meas_package) {
    // Calculate error
    VectorXd z = H_ * x_;
    VectorXd y = meas_package.raw_measurements_ - z;

    // Project uncertanity to measurement space
    MatrixXd S = H_ * P_ * H_.transpose() + R_lidar;

    // Calculate Kalman Gain
    MatrixXd K = P_ * H_.transpose() * S.inverse();

    // Update state vector
    x_ = x_ + K * y;

    // Update state covariance matrix
    P_ = (MatrixXd::Identity(5, 5) - K * H_) * P_;

    if(nis_lidar_) {
        // calculate NIS for radar
        double nis = t_.CalculateNIS(meas_package.raw_measurements_, z, S);
        fs_ldr_ << dt_ << "," << nis << std::endl;
        // /std::cout << "NIS lidar = " << nis << std::endl;
    }
}


/**
 * Updates the state and the state covariance matrix using a radar measurement.
 * @param {MeasurementPackage} meas_package
 */
void UKF::UpdateRadar(MeasurementPackage meas_package) {
    MatrixXd z_pred = MatrixXd(3, 2*n_aug_+1);

    // Predict measurement
    // reuse the sigma points from the sigma points prediction step
    for(int i=0; i<2*n_aug_+1; ++i) {
        double px = x_sig_pred_(0, i);
        double py = x_sig_pred_(1, i);
        double v = x_sig_pred_(2, i);
        double psi = x_sig_pred_(3, i);
        double sq_pxpy = sqrt(px*px + py*py);

        VectorXd tmp = VectorXd(3);
        tmp <<  sq_pxpy,
                atan2(py, px),
                (px*v*cos(psi) + py*v*sin(psi))/sq_pxpy;
        z_pred.col(i) = tmp;
    }

    // Calculate mean and covariance of predicted measurements
    VectorXd z = VectorXd(3);
    MatrixXd S = MatrixXd(3, 3);

    z.fill(0.0);
    // Calculate mean state vector from predicted sigma points
    for(int i=0; i<2*n_aug_+1; ++i){
        z = z + weights_(i) * z_pred.col(i);
    }

    S.fill(0.0);
    // Calculate state covariance matrix from predicted sigma points
    for(int i=0; i<2*n_aug_+1; ++i) {
        VectorXd diff = z_pred.col(i) - z;
        // normalize angle
        while(diff(1) > M_PI)
            diff(1) = diff(1) - 2.*M_PI;
        while(diff(1) < -M_PI)
            diff(1) = diff(1) + 2.*M_PI;
        S = S + weights_(i) * (diff * diff.transpose());
    }
    // add noise
    S = S + R_radar;

    // Calculate cross correlation matrix between sigma points in state space and measurement space
    MatrixXd T = MatrixXd(5, 3);
    T.fill(0.0);
    for(int i=0; i<2*n_aug_+1; ++i) {
        VectorXd x_diff = x_sig_pred_.col(i) - x_;
        // normalize angle
        while(x_diff(3) > M_PI)
            x_diff(3) = x_diff(3) - 2.*M_PI;
        while(x_diff(3) < -M_PI)
            x_diff(3) = x_diff(3) + 2.*M_PI;

        VectorXd z_diff = z_pred.col(i) - z;
        // normalize angle
        while(z_diff(1) > M_PI)
            z_diff(1) = z_diff(1) - 2.*M_PI;
        while(z_diff(1) < -M_PI)
            z_diff(1) = z_diff(1) + 2.*M_PI;
        T = T + weights_(i) * (x_diff * z_diff.transpose());
    }

    // Calculate Kalman Gain matrix
    MatrixXd K = T * S.inverse();

    // update state vector
    VectorXd z_diff = meas_package.raw_measurements_ - z;
    // normalize angle
    while(z_diff(1) > M_PI)
        z_diff(1) = z_diff(1) - 2.*M_PI;
    while(z_diff(1) < -M_PI)
        z_diff(1) = z_diff(1) + 2.*M_PI;

    x_ = x_ + K * z_diff;

    // update state covariance matrix
    P_ = P_ - K * S * K.transpose();

    if(nis_radar_) {
        // calculate NIS for radar
        double nis = t_.CalculateNIS(meas_package.raw_measurements_, z, S);
        fs_rdr_ << dt_ << "," << nis << std::endl;
        // /std::cout << "NIS radar = " << nis << std::endl;
    }
}
