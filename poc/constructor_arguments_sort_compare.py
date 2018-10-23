# https://qiita.com/giwa/items/afd1476c5211e28361ba
import inspect
import tensorflow.keras.layers as layers

def getArgument(c):
    signature = inspect.signature(c.__init__)
    args = []
    for k in signature.parameters:
        if k == 'self':
            continue
        if k == 'kwargs':
            continue
        args.append(k)
    return args



classes = [layers.Activation,layers.ActivityRegularization,layers.AlphaDropout,layers.BatchNormalization,layers.Cropping1D,layers.Cropping2D,layers.Cropping3D,layers.Dense,layers.Dropout,      layers.ELU,layers.Embedding,layers.Flatten,layers.GRUCell,layers.GaussianDropout,layers.GaussianNoise,layers.InputLayer,layers.LSTMCell,layers.Lambda,layers.LeakyReLU,layers.LocallyConnected1D,layers.LocallyConnected2D,layers.Masking,layers.PReLU,layers.Permute,layers.RNN,layers.ReLU,layers.RepeatVector,layers.Reshape,layers.SimpleRNNCell,layers.Softmax,layers.StackedRNNCells,layers.ThresholdedReLU,layers.UpSampling1D,layers.UpSampling2D,layers.UpSampling3D,layers.Wrapper,layers.ZeroPadding1D,layers.ZeroPadding2D,layers.ZeroPadding3D]
current = layers.Cropping2D
currentArguments = getArgument(current)

def argument_cmp_key(x):
    args = getArgument(x)
    count = 0
    for a in currentArguments:
        if a in args:
            count = count + 1
    clen = len(args)
    return count * -1, clen * -1, current


classes.sort(key=argument_cmp_key)

for c in classes:
    print(c)

