# Testing
We decided to use the __getitems__ as the function to test, as it could be unintuitive.

The __getitem__ method in the EEG_Sample class is responsible for returning a new EEG_Sample object based on the current access pattern and the provided index or slice. This function could produce unexpected results if:
- The access pattern is not handled correctly (e.g., mixing up Epoch and Minute).
- The returned object does not match the expected type or shape.
- Edge cases (index passed as slice is of wrong type)