package com.example.neochi;

import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.*;

public class LogisticLayer extends BaseLayer{

    public LogisticLayer(int n_output, int n_prev_output){
        super(n_output, n_prev_output);
        name = "logistic";
    }

    protected INDArray func(INDArray x){
        return Transforms.sigmoid(x);
    }

    protected INDArray derivaiveFunc(INDArray x){
        return Transforms.sigmoidDerivative(x);
    }
}
