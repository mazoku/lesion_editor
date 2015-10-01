import tools
import numpy as np
import io3d

fname = '/home/tomas/Data/liver_segmentation/tryba/data_other/org-exp_183_46324212_venous_5.0_B30f-.pklz'
dr = io3d.DataReader()
datap = dr.Get3DData(fname, dataplus_format=True)

a = datap['data3d']
a = a[15:18, 150:160, 150:160]

b = tools.resize3D(a, scale = 0.5)
c = tools.resize3D(a, scale = 0.5)

# print 'ORIG:'
# print a
# print '\n'

print 'CV2:'
print b
print '\n'

print 'SKIMAGE:'
print c