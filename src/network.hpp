#include <vector>
#include <Neuron.hpp>

using namespace std;

class Network {
    public: 
        Network(const vector<unsigned> &topology);
        void feedForward(const vector<double> &inputVals) {};
        void backProp(const vector<double> &targetVals) {};
        void getResults(vector<double> &resultvals) {};

    private:
        vector<Layer> h_layers; //hidden layer with h_layers[layerNum][neuronNum]
};