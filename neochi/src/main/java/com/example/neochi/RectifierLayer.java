package com.example.neochi;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.Transforms;

public class RectifierLayer extends BaseLayer {
    public RectifierLayer(int n_output, int n_prev_output){
        super(n_output, n_prev_output);
        name = "rectifier";
    }

    protected INDArray func(INDArray x){
        return Transforms.relu(x);
    }

    protected INDArray derivaiveFunc(INDArray x){
        return Transforms.leakyReluDerivative(x, 0.0);
    }
}
