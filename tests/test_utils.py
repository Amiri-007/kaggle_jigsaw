import unittest
from src.data.utils import list_identity_columns as utils_list_identity_columns
from src.data import list_identity_columns as data_list_identity_columns
from scripts.write_metrics import list_identity_columns as metrics_list_identity_columns

class TestIdentityColumns(unittest.TestCase):
    """Test that identity columns are consistent across the codebase."""
    
    def test_identity_columns_consistency(self):
        """Verify that all imports of list_identity_columns return the same list."""
        # Get identity columns from each source
        utils_cols = utils_list_identity_columns()
        data_cols = data_list_identity_columns()
        metrics_cols = metrics_list_identity_columns()
        
        # Convert to sets for comparison
        utils_set = set(utils_cols)
        data_set = set(data_cols)
        metrics_set = set(metrics_cols)
        
        # Verify all sets are equal
        self.assertEqual(utils_set, data_set, 
                        "Identity columns in utils.py and __init__.py don't match")
        self.assertEqual(utils_set, metrics_set, 
                        "Identity columns in utils.py and write_metrics.py don't match")
        
        # Verify the counts are the same (no duplicates)
        self.assertEqual(len(utils_cols), len(data_cols),
                        "Different number of identity columns in utils.py and __init__.py")
        self.assertEqual(len(utils_cols), len(metrics_cols),
                        "Different number of identity columns in utils.py and write_metrics.py")

if __name__ == "__main__":
    unittest.main() 