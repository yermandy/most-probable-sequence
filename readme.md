
#### Problem formulation (inference)

$$ \hat{y_1}, \dots, \hat{y_n}  \in \argmax_{y_1, \ \dots \ ,y_n} {\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})} $$

$$ y_i \in \{0,1,\dots,Y\} $$

$$ \forall i \in \{1,\dots,n \} $$


Scores are represented by
$$
f_{i}(y_i,y_{i+1}) = \braket{w(y_i + y_{i+1}), \phi(x_{i,i+i})} + b(y_i + y_{i+1})
$$

where $w$ and $b$ are weights and biases of the last layer of the network


<!-- TODO: dimentsions of all variables -->

#### Margin rescaling loss

$$
\Delta (\theta, \bar{x}, \bar{y}) = \max_{y_1, \ \dots \ ,y_n} \Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}} + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) | \Big] - \sum_{i=1}^{n-1} f_{i}(\bar{y}_i, \bar{y}_{i+1})
$$