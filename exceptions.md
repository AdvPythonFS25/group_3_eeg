1. first try/ except block: lines 54-57: This extracts the age attribute from the metadata of each mne.object. If there is none, the except block sets the age to None. This allows for the program to still work even if one or more of the files metadata has been corrupted.

2. second try/ except block: 106-114: This code needs to get an input from the user to select a single queried object if more than one fits the parameter criteria. In this case, if the user selects an integer but it is out of the range of the queried objects, the else statement in the try prompts the user to pick a valdi integer. If the user does not select an integer, the except block prompts the user to input an integer.

3. third try. except block 240-247: ensures the user inputs positive integer or None.


4. third try/ except block: 263-275: This is probably the most impactful try/ except block in our code. It intakes a time range as a tuple (as a string) from the user, and converts it to the tuple type, while disabling the posibility of malicious injection using {"__builtins__": {}}. It then checks the input is a tuple, the tuple has only the start and end values, and that they are within the 0 and the max time (arbitrarily large value). If ay of these conditions fail, the except statement prints an error message and sets the value of time_range to None.