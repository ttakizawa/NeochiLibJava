package com.example.neochi;

import org.json.JSONArray;
import org.json.JSONObject;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.ops.transforms.*;

import java.util.ArrayList;

public class Network {
    private String name;
    private int inputSize;
    private double epsilon;
    private ArrayList<BaseLayer> layers;

    public Network(String name, int inputSize, double epsilon){
        this.name = name;
        this.inputSize = inputSize;
        this.epsilon = epsilon;
        this.layers = new ArrayList<BaseLayer>();

    }

    public void addLayer(String type, int n_output){
        int n_prev_output;
        if(this.layers.size()==0){
            n_prev_output = this.inputSize;
        }else{
            n_prev_output = this.layers.get(this.layers.size() - 1).n_output;
        }
        BaseLayer layer = getLayer(type, n_output, n_prev_output);
        this.layers.add(layer);
    }

    public BaseLayer getLayer(String type, int n_output, int n_prev_output){
        if(type.equals("rectifier")){
            return  new RectifierLayer(n_output, n_prev_output);
        }else if(type.equals("logistic")){
            return  new LogisticLayer(n_output, n_prev_output);
        }
        return  null;
    }

    public INDArray errorFunction(INDArray t, INDArray y){
        return t.neg().transpose().mul(Transforms.log(y));
    }

    public INDArray errorDerivative(INDArray t, INDArray y){
        return t.neg().div(y);
    }

    public INDArray propagateForward(INDArray input){
        INDArray output = input;
        for(BaseLayer layer: this.layers){
            output = layer.propagateForward(output);
        }
        return output;
    }

    public void propagateBackward(INDArray X, INDArray y){
        INDArray pred = propagateForward(X);
        System.out.printf("pred_shape row:%d col:%d \n", pred.size(0), pred.size(1));
        INDArray delta = errorDerivative(y, pred);

        BaseLayer nextLayer = null;
        for(int i=this.layers.size() - 1; i>=0;i--){
            if(nextLayer != null){
                this.layers.get(i).propagateBackward(delta, nextLayer.w);
            }else{
                this.layers.get(i).propagateBackward(delta, null);
            }
            nextLayer = this.layers.get(i);
        }
    }

    public void update(INDArray X){
        BaseLayer prevLayer = null;
        for(BaseLayer layer: this.layers){
            if(prevLayer == null){
                layer.update(X, this.epsilon);
            }else{
                layer.update(prevLayer.y, this.epsilon);
            }
        }
    }

    public JSONObject toJson(){
        JSONObject model = new JSONObject();
        JSONObject meta = new JSONObject();
        meta.put("name", this.name);
        meta.put("n_input", this.inputSize);
        meta.put("error_func", "cross_entropy");
        meta.put("epsilon", this.epsilon);
        model.put("meta", meta);
        //JSONArray layersJson = new JSONArray();
        ArrayList<String> layersJson = new ArrayList<String>();
        //JSONArray layersJson = null;
        for(BaseLayer layer: this.layers){
            /*if(layersJson == null){
                layersJson = new JSONArray(layer.toJson());
            }else {
                layersJson.put(layer.toJson());
            }*/
            layersJson.add(layer.toJson().toString());
        }
        JSONArray jsonArray = new JSONArray(layersJson);
        //model.put("layers", layersJson);
        model.put("layers", jsonArray.toString());
        return model;
    }
}
