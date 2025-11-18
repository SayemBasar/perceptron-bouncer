import numpy as np

def get_training_data():
    features = [
    "wears_i_love_javascript_tshirt",
    "is_carrying_menacing_rubber_duck",
    "tries_to_pay_with_pennies",
    "jumps_the_queue",
    ]

# Each line represents a person, and whether or not they have the features above
    dataset = [
        [1, 1, 1, 1],  
        [1, 1, 1, 0],          
        [1, 1, 0, 0],  
            [0, 1, 1, 0],  
            [0, 0, 0, 1],  
        [0, 1, 0, 0],  
            [0, 0, 0, 0],  
    ]

# This list represents whether the people above are (0 = not allowed in) or (1 = allowed in)
    targets = [0, 0, 0, 1, 1, 1, 1]  

    return {
        "features_name": features,
        "X":np.array(dataset),
        "y": np.array(targets)
    }

# data=get_training_data()
# print(data)