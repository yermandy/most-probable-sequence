
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

$$\bar c = \sum_{i=1}^{n} \bar y_i$$

where $\bar y_i$ denote the ground truth labeling for $i$-th window.

#### Solving $F(c)$ efficiently with Dynamic Programming

Let $F_k(c, y_k)$ has the following form:

$$
\begin{align}
    F_k(c, y_k) &= \max_{y_1+...+y_n=c} \Big[ \sum_{i=1}^{k-2} f_{i}(y_i,y_{i+1}) + f_{k-1}(y_{k-1},y_{k}) \Big] \\
    &= \max_{y_{k-1} \in \{ l(c, k),...,u(c)\} } \Big[ F_{k-1}(c-y_{k-1}, y_{k-1}) + f_{k-1}(y_{k-1},y_{k}) \Big] \\
    \forall c &\in \{ 0,\dots, (k - 1) \cdot (Y - 1) + 1 \} \\
    \forall y_k &\in \{ 0,\dots, Y\}
\end{align}
$$

where 
$$
\begin{align}
l(c, k) &= \max(0, c - (Y-1)\cdot(k - 2)) \\
u(c) &= \min(c, Y)
\end{align}
$$

Then we can define $F(c)$ as

$$
\begin{align}
F(c) &= \max_{y_1 + ... + y_n = c}\sum_{i=1}^{n-1} f_{i}(y_i,y_{i+1}) \\
     &= \max_{y_n \in \{ l(c, n),...,u(c)\}} F_n(c - y_n, y_n)
\end{align}
$$



