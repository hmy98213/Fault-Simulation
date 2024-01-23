import tensornetwork as tn
import numpy as np
A = tn.Node(np.array([[0.9, -0.3], [0.3, 0.9]], dtype = complex))
B = tn.Node(np.array([0, 1], dtype = complex))
tn.connect(A[0], B[0])
result = tn.contractors.auto([A, B]).tensor
print (result)