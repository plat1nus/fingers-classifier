# [[1.0731894e-07 2.7606636e-06 5.0129290e-03 6.4103856e-06 9.9402493e-01
#   9.5281383e-04]]

def decode_predict(pred):
    return pred[0].tolist().index(max(pred[0]))