from mp_standard import MPStandardNeuron

c1_xhigh = MPStandardNeuron(weights=[-1], bias=8)
c1_xlow = MPStandardNeuron(weights=[1], bias=-1)
c1_yhigh = MPStandardNeuron(weights=[-1], bias=1)
c1_ylow = MPStandardNeuron(weights=[1], bias=0)

c2_xhigh = MPStandardNeuron(weights=[-1, -1.5], bias=7.5)
c2_xlow = MPStandardNeuron(weights=[1], bias=0)
c2_yhigh = MPStandardNeuron(weights=[-1], bias=5)
c2_ylow = MPStandardNeuron(weights=[1], bias=-3)

c3_xhigh = MPStandardNeuron(weights=[-1], bias=0)
c3_xlow = MPStandardNeuron(weights=[1], bias=4)
c3_yhigh = MPStandardNeuron(weights=[-1], bias=2)
c3_ylow = MPStandardNeuron(weights=[1], bias=0)

c1_final = MPStandardNeuron(weights=[1,1,1,1], bias=-3.5)
c2_final = MPStandardNeuron(weights=[1,1,1,1], bias=-3.5)
c3_final = MPStandardNeuron(weights=[1,1,1,1], bias=-3.5)


def feedforward(x, y):

    #Evaluate c_1 intermediates
    out_c1_1 = c1_xhigh.evaluate([x])
    out_c1_2 = c1_xlow.evaluate([x])
    out_c1_3 = c1_yhigh.evaluate([y])
    out_c1_4 = c1_ylow.evaluate([y])

    #Evaluate c_2 intermediates
    out_c2_1 = c2_xhigh.evaluate([x, y])
    out_c2_2 = c2_xlow.evaluate([x])
    out_c2_3 = c2_yhigh.evaluate([y])
    out_c2_4 = c2_ylow.evaluate([y])

    #Evaluate c_3 intermediates
    out_c3_1 = c3_xhigh.evaluate([x])
    out_c3_2 = c3_xlow.evaluate([x])
    out_c3_3 = c3_yhigh.evaluate([y])
    out_c3_4 = c3_ylow.evaluate([y])

    c1 = c1_final.evaluate([out_c1_1, out_c1_2, out_c1_3, out_c1_4])
    c2 = c2_final.evaluate([out_c2_1, out_c2_2, out_c2_3, out_c2_4])
    c3 = c3_final.evaluate([out_c3_1, out_c3_2, out_c3_3, out_c3_4])

    return (c1, c2, c3)

print(str(feedforward(1.5,0.5)) + " should be class 1")
print(str(feedforward(-3,1)) + " should be class 3")
print(str(feedforward(0.1,3.5)) + " should be class 2")
print(str(feedforward(4,2)) + " should be class 4")
print(str(feedforward(0.5,1.5)) + " should be class 4")
print(str(feedforward(1,1)) + " Should be class 1")
print(str(feedforward(3,3)) + " Should be class 2")

