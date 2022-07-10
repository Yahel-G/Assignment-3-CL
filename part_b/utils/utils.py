# =============================================================================
# Imports
# =============================================================================
import json
from enum import Enum

import numpy as np


# =============================================================================
# JSON encoders
# =============================================================================
class NumpyEncoder(json.JSONEncoder):
    """ 
    Enables to put NumPy arrays into a JSON file.
    """
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()

        if isinstance(obj, Enum):
            return obj.name

        return json.JSONEncoder.default(self, obj)


# =============================================================================
# Write dict to JSON file with NumpyEncoder
# =============================================================================
def save_numpy_to_json(file_name, data):
    """
    Saves dictionary (dict object) to a JSON file with NumpyEncoder which allow
    writing NumPy arrays into JSON.

    Parameters
    ----------
    file_name: [String]
        Full path and file name to save to.
    data: [Dict]
        Dictionary object to save into JSON

    Returns
    -------
    Saves JSON file in path "file_name"

    """

    # Encode as JSON object
    json_results = json.dumps(data,
                              cls=NumpyEncoder,
                              sort_keys=False,
                              indent=5,
                              separators=(',', ': '))

    json_results = json_results.replace('NaN', 'null')

    # Write to file
    with open(file_name, 'w') as file:
        file.write(json_results)
