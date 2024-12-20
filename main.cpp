#include <iostream>
#include <vector>

struct Perceptron {
public:
std::vector<double> _w;
double _b;

public:
std::vector<double> _x;
double _z;
double _a;

public:
std::vector<double> _w_grads;
double _b_grad;
double _dLdy;
double _dydz;

public:
void w_init();
double forward(std::vector<double>& x);
void backward(const double& lr, double& gt);
void zero_grad();
};

void Perceptron::w_init() {
  for (size_t i {0}; i < 3; ++i) {
    this->_w.push_back(1.0);
  }

  this->_b = 1.0;
}

double ReLU(double z) {
  return (z <= 0.0 ? 0.0 : z);
}

double Perceptron::forward(std::vector<double>& x) {
  this->_x = x;

  double z {0.0};
  for (size_t i {0}; i < this->_w.size(); ++i) {
    z += x[i] * this->_w[i];
  } z += this->_b;
  
  this->_z = z;
  this->_a = ReLU(z);

  return this->_a;
}

void Perceptron::zero_grad() {
  if (this->_b_grad != 0.0) {
    this->_b_grad = 0.0;
  }
  if (this->_w_grads.size() != 0) {
    for (size_t i {0}; i < this->_w_grads.size(); ++i) {
      this->_w_grads.erase(this->_w_grads.begin());
    }
  }
}

void Perceptron::backward(const double& lr, double& gt) {
  this->_dLdy = -2.0 * (gt - this->_a); // depends on loss function
  this->_dydz = (this->_a < 0.0 ? 0.0 : 1.0); // depends on act function

  for (size_t i {0}; i < this->_w.size(); ++i) {
    this->_w_grads.push_back(this->_x[i] * this->_dydz * this->_dLdy);
  }

  this->_b_grad = 1.0 * this->_dydz * this->_dLdy;

  // updating weights
  for (size_t i {0}; i < this->_w.size(); ++i) {
    this->_w[i] = this->_w[i] - lr * this->_w_grads[i];
  } this->_b = this->_b - lr * this->_b_grad; 
}

double MSE(double& gt, double& pr) {
  return (gt - pr) * (gt - pr);
}

int main() {
  const size_t epochs {100};
  const float lr {0.001};

  std::vector<double> x {3.0, 1.0, 8.0};
  double gt {50.0};

  Perceptron model;
  model.w_init();

  for (size_t epoch {0}; epoch < epochs; ++epoch) {
    // FORWARD PASS
    double pr = model.forward(x);

    // LOSS CALCULATION
    double loss = MSE(gt, pr);

    // BACKWARD PASS
    model.zero_grad();
    model.backward(lr, gt);

    // PRINT CURRENT STATE
    std::cout << "========== WEITGHS ==========\n";
    for (size_t i {0}; i < model._w.size(); ++i) {
      std::cout << model._w[i] << '\n';
    }

    std::cout << '\n';
    std::cout << "==========  BIAS   ==========\n";
    std::cout << model._b << '\n';

    std::cout << "==========  LOSS   ==========\n";
    std::cout << loss << '\n';
    std::cout << '\n';

    std::cout << "========== PREDICT | GT ==========\n";
    std::cout << pr << '\t' << gt << '\n';
  }

  return 0;
}