Auto Diff
---------
* Suppose you had the following function and you need its gradient
  ```js
  f(x, y) = yx^2 + y + 2

  grad(f) = [
    2xy,
    X^2 + 1
  ]
  ```

* To compute its gradient using code, we have a few options:

Finite Difference Approximation
-------------------------------
* We can use the derivative definition to get an approximation of a derivative:
  ```js
  h^(x0) = lim x->x0 (h(x) - h(x0)) / (x - x0)

  // using epsilon to represent the infetesimal change
  h^(x) = lim eps->0 (h(x + eps) - h(x)) / eps

  ```
  - To get the rate of change of the slope at point x0, we get the rate of change for an infetisimal
    range.
  - See notebook for code example.

* This approach is imprecise and does not scale well to bigger functions.
  - See the floating point in the results in the notebook.

* It can be used to verify the results of a different approach since it's so simple.

Forward-Mode Autodiff
----------------------
* Using this approach, we build a graph structure for the function and a graph for each of its partial derivatives.

* Consider this function:
  ```js
  f(x,y) = 5 + x*y
  
  grad = [
    y,
    x
  ]
  ```

* The function graph breaks each component recursively to a node (similar to how a compiler would do it):
  - leaf node: x
  - leaf node: y
  - mult node: x*y
  - leaf node: 5
  - add node: 5 + x*y

* Now we can create a similar graph for the partial derivative. For example, the graph for the partial derivative 
  with respect to x:
  - leaf node: y
  - leaf node der: x --> 1
  - mult node: y*1
  - leaf node: x
  - leaf node der: y --> 0
  - mult node: x*0
  - add node
    * Since the derivative of a product is addition 
  ...

* The idea is to break the function and derivation operation to nodes and solve them iteratively.
  - This way each node can have its own derivation and we can put them together as blocks to represent more
    complicated functions.

* This form of differentiation is precise but still cannot scale - if we have 1000 parameters, we need to run thru
  the derivative graph 1000 times to get the gradient.

Reverse-Mode Autodiff
---------------------
* Reverse mode comes in where forward-mode left off, it can compute the partial derivative of any number of parameters 
  with just two passes of the graph.

* It first does a forward pass, storing the results of each node. Then it goes in reverse, finding the derivative
  of each node. 
  - At the end, we're left with the partial derivatives with respect to each parameter at their nodes.

  ```js
  f(x,y) = x^2 * y + y + 2
  ```

* At each node we take the partial derivative of the function with respect to the node while taking into account the 
  chain rule. For example, for f above:
  ```js
  f(3,4) = 3^2 * 4 + 4 + 2 = 9 * 4 + 6 = 36 + 6 = 42

  n7 = n5 + n6 = 42
  n6 = n2 + n3 = 6
  n5 = n4 * n2 = 36
  n4 = n1^2= 9
  n3 = 2
  n2 = y
  n1 = x

  f(x,y) = n1^2 * n2 + n2 + n3 = n4 * n2 + n6 = n5 + n6 = n7
  
  ∂f/∂n7 = 1

  ∂f/∂n6 = ∂f/∂n7 * ∂n7/∂n6 = 1 * ∂(n5 + n6)/∂n6 = 1
  ∂f/∂n5 = ∂f/∂n7 * ∂n7/∂n5 = 1 * ∂(n5 + n6)/∂n5 = 1

  ∂f/∂n4 = ∂f/∂n5 * ∂n5/∂n4 = 1 * ∂(n4 * n2)/∂n4 = n2 = y = 4
  ∂f/∂n3 = ∂f/∂n6 * ∂n6/∂n3 = 1 * ∂(n2 + n3)/∂n3 = 1
  ∂f/∂n2 = ∂f/∂n5 * ∂n5/∂n2 = 1 * ∂(n4 * n2)/∂n2 = n4 = 9

  ∂f/∂y = ∂f/∂n2 + ∂f/∂n3 = 9 + 1 = 10 // (which is equal to x^2 + 1, the partial der of y)
  ∂f/∂x = n1 * ∂f/∂n4 + n1 * ∂f/∂n4 = 3 * 4 + 3 * 4 = 24 // (which is equal to 2xy, the partial der of x)
  ```

* Reverse diff is scalable since it only needs to do a forward and a reverse pass to any function, regardless the
  number of parameters.

