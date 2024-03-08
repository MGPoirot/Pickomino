`states_old.pkl` is a dictionary with as 

* `keys` all possible collections with 8 dice, from game start with an empty collection of (0, 0, 0, 0, 0, 0), all the way to game end, an collection of for example (8, 0, 0, 0, 0, 0).
* `value `each possible dice roll that can happen after (1287 for (0, 0, 0, 0, 0, 0) and 0 for (8, 0, 0, 0, 0, 0))

`states_all.pkl` improves on this dict in that its values in combination with the key always sum to 8. Also rolls of (0,0,0,0,0,0) were replaced with `None` .

`states_valid.pkl` is the same as `states_all.pkl` but 

* `values` are only <u>valid</u> pick ups. This is a pruned version of all roll. For example when the collection is (0, 0, 4, 4, 0, 0), an invalid pickup would be (0, 0, 2, 0, 0, 0).



# Dice rolling in Pickomino

**Rules:** Each turn players roll their dice and set aside all those matching any single value. The remaining dice are rolled and any value is set aside again until the player stops and takes a tile or busts and puts their last tile back. When a player busts and fails to take a tile they must also turn the highest tile face-down.
Players

**Implementation:** `dice.py` calculates the probabilities of rolling dice in the game Pickomino.

# Code Definitions



# pickomino-rl

Reinforcement learning pickomino bot



# Previously


T. Ten Cate found that 
https://frozenfractal.com/blog/2015/5/3/how-to-win-at-pickomino/