#+begin_src python
import numpy as np
x = np.random.rand(4, 5)

y = np.sum(x, axis=1)
print(y.shape)
#+end_src