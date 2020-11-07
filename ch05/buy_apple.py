import numpy as np 
from layer_naive import MulLayer

apple = 100
apple_num = 2
tax = 1.1

#layer
mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()

# forward
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)


# backward
dprice = 1
dapple_price, dtax = mul_tax_layer.backward(dprice)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)


print("price: {:0.0f}".format(price))
print("dapple: {}".format(dapple))
print("dapple_num: {:0.0f}".format(dapple_num))