from mpb import MPBNeuron

a1 = MPBNeuron(weights=[1,1], bias=-1.5)
a2 = MPBNeuron(weights=[1,1], bias=-0.5)

b1 = MPBNeuron(weights=[-1], bias=0.5)
b2 = MPBNeuron(weights=[1], bias=-0.5)

y = MPBNeuron(weights=[1,1], bias=-1.5)

def feedforward(x, v):

    out_a1 = a1.evaluate([x, v])
    out_a2 = a2.evaluate([x, v])

    out_b1 = b1.evaluate([out_a1])
    out_b2 = b2.evaluate([out_a2])

    out_y = y.evaluate([out_b1, out_b2])


    return out_y

print(feedforward(0,0))
print(feedforward(0,1))
print(feedforward(1,0))
print(feedforward(1,1))
