# Uncertainty-aware Causal Graph Learning

## introduction
GNNs ideally utilize subgraphs to predict labels when the graphs are unbiased, indicating that only causal subgraphs are related to the graph labels. However, graph datasets inevitably contain biases due to uncontrollable data collection methods. Consider Figure 1 where most of the digit backgrounds are the same, such as the red background for “zero”. In this case, GNNs do not need to learn the correct function to achieve high accuracy in predicting the digit labels. Instead, it is much easier for them to learn statistical shortcuts that link the background color with the most frequently occurring digit in each case. Unfortunately, such methods generalize poorly when encountering out-of-distribution (OOD) data, where the background color changes in the testing dataset. Consequently, these non-causal subgraphs, which are only marginally related to the causal part, provide limited exposure to the causal subgraphs for predicting the labels.


<p align="center"><img src="image/CMIST.png" width=50% height=50%></p>
<p align="center"><em>Figure 1.</em> CMNIST-75sp Dataset.</p>
