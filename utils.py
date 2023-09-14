from pathlib import Path
import torch 
from datasets import load_metric,load_dataset
import pandas as pd
import numpy as np
from time import perf_counter 
import matplotlib.pyplot as plt
query = """hey, I want to buy a macbook pro"""
accuracy_score = load_metric("accuracy") 
clinc = load_dataset("clinc_oos", "plus")
intents = clinc["train"].features["intent"]
class PerformanceBenchmark: 
    def __init__(self, pipeline , dataset , optim_type = "BERT baseline"):
        self.pipeline = pipeline 
        self.dataset = dataset 
        self.optim_type = optim_type
    def compute_accuracy(self):
        """This overrides the PerformanceBenchmark.compute_accuracy() method"""
        # use gpu 
        preds, labels = [], []
        for example in self.dataset:
            input = example["text"]
            pred = self.pipeline(input)[0]["label"]
            pred = self.pipeline(example["text"])[0]["label"]
            preds.append(intents.str2int(pred))
            labels.append(example["intent"])
        # accuracy = accuracy_score.compute(predictions=preds, references=labels)
        accuracy = 0
        print(f"Accuracy on test set - {accuracy['accuracy']:.3f}")
        return accuracy
    def compute_size(self): 
        state_dict = self.pipeline.model.state_dict() # map each learnable layers to learnable parameteres (ex weights, bias)
        tmp_path = 'model.pt'
        torch.save(state_dict, tmp_path)
        # calculate size im MB 
        size_mb = Path(tmp_path).stat().st_size / (1024 * 1024) # use st_size to get the size of the file in bytes
        print(f"Model size (MB) - {size_mb:.2f}")  
        return {"size_mb": size_mb}
    def time_pipeline(self):
        """This overrides the PerformanceBenchmark.time_pipeline() method""" 
        latencies = [] 
        for _ in range(10):
            _ = self.pipeline(query)
        # time run 
        for _ in range(100):
            start_time = perf_counter() 
            _ = self.pipeline(query)
            latency = perf_counter() - start_time 
            latencies.append(latency)
        # compute run statistics
        time_avg_ms = 1000 * np.mean(latencies)
        time_std_ms = 1000 * np.std(latencies) 
        print(f"Average latency ms - {latency:.3f} +/- {time_std_ms:.3f}") 
        return {"time_avg_ms": time_avg_ms, "time_std_ms": time_std_ms}
    def run_benchmark(self): 
        metrics = {} 
        metrics[self.optim_type] = self.compute_size() 
        metrics[self.optim_type].update(self.time_pipeline())  # update dictionary will add new k-v pair to the dictionary
        metrics[self.optim_type].update(self.compute_accuracy())
        return metrics
    
def plot_metrics(perf_metrics, current_optim_type):
    df = pd.DataFrame.from_dict(perf_metrics, orient='index')
    for idx in df.index:
        df_opt = df.loc[idx]
        # Add a dashed circle around the current optimization type
        if idx == current_optim_type:
            plt.scatter(df_opt["time_avg_ms"], df_opt["accuracy"] * 100,
            alpha=0.5, s=df_opt["size_mb"], label=idx,
            marker='$\u25CC$')
        else:
            plt.scatter(df_opt["time_avg_ms"], df_opt["accuracy"] * 100,
            s=df_opt["size_mb"], label=idx, alpha=0.5)
    legend = plt.legend(bbox_to_anchor=(1,1))
    for handle in legend.legendHandles:
        handle.set_sizes([20])
    plt.ylim(80,90)
    # Use the slowest model to define the x-axis range
    xlim = int(perf_metrics["BERT baseline"]["time_avg_ms"] + 3)
    plt.xlim(1, xlim)
    plt.ylabel("Accuracy (%)")
    plt.xlabel("Average latency (ms)")
    plt.show()