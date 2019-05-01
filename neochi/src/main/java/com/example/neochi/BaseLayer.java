package com.example.neochi;

import org.json.*;
import org.nd4j.linalg.api.buffer.DataBuffer;
import org.nd4j.linalg.api.ndarray.INDArray;
import org.nd4j.linalg.factory.Nd4j;

public abstract class BaseLayer {
    protected String name = "base";
    public INDArray w;
    public INDArray b;
    public INDArray y;
    public INDArray delta;

    public int n_output;

    public BaseLayer(int n_output, int n_prev_output){
        this.n_output = n_output;
        initW(n_output, n_prev_output);
        initb(n_output);
    }

    protected void initW(int n_output, int n_prev_output){
        this.w = Nd4j.rand(n_output, n_prev_output);
    }

    protected void initb(int n_output){
        //this.b = Nd4j.rand(n_output, 1);
        this.b = Nd4j.rand(1, n_output);
    }

    protected abstract INDArray func(INDArray x);/*{
        return x;
    }*/

    protected abstract INDArray derivaiveFunc(INDArray x);/*{
        return Nd4j.ones(x.rows(), 1);
    }*/

    public INDArray propagateForward(INDArray x){
        //this.y = func(this.w.mmul(x).add(this.b));
        this.y = func(x.mmul(this.w.transpose()).add(this.b.broadcast(x.size(0), this.b.size(1))));
        return this.y;
    }

    public INDArray propagateBackward(INDArray next_delta,INDArray next_w){
        if (next_w != null){
            this.delta = derivaiveFunc(this.y).mmul(next_w.transpose()).mmul(next_delta);
        }else{
            this.delta = derivaiveFunc(this.y).mmul(next_delta);
        }
        return this.delta;
    }

    public void update(INDArray prev_y, double epsilon){
        INDArray delta_w = this.delta.mmul(prev_y.transpose());
        this.w = this.w.sub(delta_w.mul(epsilon));
        this.b = this.b.sub(this.delta.mul(epsilon));
    }

    public JSONObject toJson(){
        JSONObject object = new JSONObject();
        object.put("type", this.name);
        DataBuffer buffer = this.w.data();
        double[] arrayW = buffer.asDouble();
        buffer = this.b.data();
        double[] arrayb = buffer.asDouble();
        JSONArray jsonW = new JSONArray();
        JSONArray jsonb = new JSONArray();
        jsonW.put(arrayW);
        jsonb.put(arrayb);
        object.put("W", jsonW);
        object.put("b", jsonb);
        //object.put("W", this.w);
        //object.put("b", this.b);
        //object.put("W", arrayW);
        //object.put("b", arrayb);
        return object;
    }
}
