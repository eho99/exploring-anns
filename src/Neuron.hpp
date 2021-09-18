#include <vector> 

using namespace std;

struct Connection {
    double weight, d_weight;
};

typedef vector<Neuron> Layer;

class Neuron {
    public:
        Neuron(unsigned numOutputs);

    private:
        double m_outputVal;
        vector<Connection> m_outputWeights;
        static double randomWeight(void) {
            return rand() / double(RAND_MAX);
        }
};


