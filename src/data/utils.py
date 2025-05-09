"""
Utility functions for data processing.
"""
from typing import List

def list_identity_columns() -> List[str]:
    """
    Return the list of standard identity columns used in the dataset.
    
    This is the canonical source for identity columns - other modules should import from here.
    """
    return [
        'male', 'female', 'homosexual_gay_or_lesbian', 'christian', 'jewish',
        'muslim', 'black', 'white', 'psychiatric_or_mental_illness',
        'asian', 'hindu', 'buddhist', 'atheist', 'bisexual', 'transgender'
    ] 