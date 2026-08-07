# y = 0

# def myFunc():
#   global y
#   y = 10
#   yield "Hello"
#   y = 20
#   yield 51
#   y = 30
#   yield "Good Bye"

# #The function is called:
# x = myFunc()

# #But y is still 0:
# print("At this point, y is still:", y)

# #Run the first iteration:
# next(x)

# #And y becomes 10:
# print("Now, y is:", y)

# #Run another iteration:
# next(x)

# #And y becomes 20:
# print("Now, y is:", y)

# #Run another iteration:
# next(x)

# #And y becomes 30:
# print("Now, y is:", y)

import time
def generator():
    for num in range(0, 5):
       yield(num)
       time.sleep(1)


for result in generator():
    print(result, end=" ") 