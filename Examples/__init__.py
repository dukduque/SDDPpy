def import_SDDP():
    import os
    import sys
    examples_path = os.path.dirname(os.path.realpath(__file__))
    parent_path = os.path.abspath(os.path.join(examples_path, os.pardir))
    print(parent_path)
    sys.path.append(parent_path)


import_SDDP()