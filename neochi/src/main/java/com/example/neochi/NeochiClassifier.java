package com.example.neochi;

import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;

public class NeochiClassifier {
    private Network net;


    public NeochiClassifier(int inputSize){
        //int inputSize = width * height;
        this.net = new Network("neochi", inputSize,0.01);

        //net.addLayer("rectifier",512);
        //net.addLayer("rectifier",256);
        net.addLayer("rectifier",128);
        net.addLayer("rectifier",64);
        net.addLayer("rectifier",32);
        net.addLayer("rectifier",16);
        net.addLayer("logistic",1);
    }

    public void fit(INDArray X, INDArray y, int epochs){

        for(int i=0; i < epochs; i++){
            System.out.printf("epoch:%d \n", i);
            for(int j=0; j < X.size(0); j++){
                net.propagateBackward(X.getRow(j), y.getRow(j));
                net.update(X.getRow(j));
            }
        }
    }

    public  String toJson(){
        JSONObject obj = this.net.toJson();
        String s = obj.toString();
        return s;
    }
}
