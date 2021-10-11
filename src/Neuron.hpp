// https://towardsdatascience.com/step-by-step-the-math-behind-neural-networks-490dc1f3cfd9
// https://takinginitiative.wordpress.com/2008/04/03/basic-neural-network-tutorial-theory/



#include <vector>
#include <cmath>

using namespace std;

struct Connection
{
    double weight, d_weight;
};

typedef vector<Neuron> Layer;

class Neuron
{
public:
    Neuron(unsigned numOutputs);

private:
    vector<Connection> m_outputWeights;
    static double randomWeight(void)
    {
        return rand() / double(RAND_MAX);
    }
};


// Weighted Sum - Dot Product
double weighted_sum(vector<double> weights, vector<double> inputs) {
    double w_sum = 0;
    for (int i = 0; i < sizeof(weights); ++i) {
        w_sum += weights[i] * inputs[i];
    }
    return w_sum;
}


// Activation Functions

double sigmoid(double x)
{
    return 1 / (1 + exp(-x));
}

double d_sigmoid(double x)
{
    return sigmoid(x) * (1 - sigmoid(x));
}