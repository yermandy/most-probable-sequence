
#### Problem formulation (inference)

$$ 
\begin{align}

\hat{y_1}, \dots, \hat{y_n}  &\in \mathop{\text{arg max}}\limits_{y_1,...,y_n} {\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})} \\

y_i &\in \{0,1,\dots,Y\} \\

\forall i &\in \{1,\dots,n \}

\end{align}
$$


Scores are represented by

$$
f_{i}(y_i,y_{i+1}) = \braket{w(y_i + y_{i+1}), \phi(x_{i,i+i})} + b(y_i + y_{i+1})
$$

where $w$ and $b$ are weights and biases of the last layer of the network

<!-- TODO: dimentsions of all variables -->

#### Margin rescaling loss

$$
\begin{align}
    
    \Delta (\theta, \bar x, \bar y) &= \max_{y_1, ... ,y_n} \Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}}| + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    
    &= \max_{c \in \{0, ... , n \cdot Y\}} \quad  \max_{y_1 + ... + y_n = c} \Big[ \frac{1}{\bar{c}} |\sum_{i=1}^n {y_i - \bar{c}}| + \sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    
    &= \max_{c \in \{0, ... , n \cdot Y\}} \Big[ \frac{1}{\bar{c}} |c - \bar{c}| + \max_{y_1 + ... + y_n = c}\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1})  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1}) \\
    
    &= \max_{c \in \{0, ... , n \cdot Y\}} \Big[ \frac{1}{\bar{c}} |c - \bar{c}| + F(c)  \Big] - \sum_{i=1}^{n-1} f_{i}(\bar y_i, \bar y_{i+1})
    
\end{align}
$$
