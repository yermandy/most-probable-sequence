

Problem formulation (inference task):

$$ \hat{y_1}, \dots, \hat{y_n}  \in \text{arg max}_{y_1, \ \dots \ ,y_n} \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) $$

$$ y_i \in \{0,1,\dots,Y\} $$

$$ \forall i \in \{1,\dots,n \} $$


Scores are represented by
$$
f_{i}(y_i,y_{i+1}) = \braket{w(y_i + y_{i+1}), \phi(x_{i,i+i})} + b(y_i + y_{i+1})
$$

where $w$ and $b$ are weights and biases of the last layer of the network


