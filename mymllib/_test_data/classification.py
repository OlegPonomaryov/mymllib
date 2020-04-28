"""Test data for classification models."""

# Classes:
#   0 (A) - both x1 and x2 are high
#   1 (B) - both x1 and x2 are low
#   2 (C) - x1 is high, x2 is low
X = [[24, 32],
     [3, 0],
     [19, 1],

     [17, 28],
     [0, 5],
     [27, 5],

     [20, 30],
     [2, 3],
     [22, 3]]

y = [0, 1, 2,  0, 1, 2,  0, 1, 2]

y_text = ["A", "B", "C", "A", "B", "C", "A", "B", "C"]

# A one-hot version of the y
y_one_hot = [[1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],

             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1],

             [1, 0, 0],
             [0, 1, 0],
             [0, 0, 1]]

# A binary version of the y with respect to the class 2
y_bin = [0, 0, 1,  0, 0, 1,  0, 0, 1]

# An index from which the test part of the dataset starts
test_set_start = 6