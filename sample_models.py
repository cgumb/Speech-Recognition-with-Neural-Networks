from keras import backend as K
from keras.models import Model
from keras.layers import (BatchNormalization, Conv1D, MaxPooling1D, Dense, Input, GaussianNoise, Dropout,
    TimeDistributed, Activation, Bidirectional, SimpleRNN, GRU, LSTM, CuDNNGRU, CuDNNLSTM)

def simple_rnn_model(input_dim, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = GRU(output_dim, return_sequences=True, 
                 implementation=2, name='rnn')(input_data)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(simp_rnn)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def rnn_model(input_dim, units, activation, output_dim=29):
    """ Build a recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add recurrent layer
    simp_rnn = CuDNNGRU(units, return_sequences=True, name='rnn')(input_data)
    # TODO: Add batch normalization 
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model


def cnn_rnn_model(input_dim, filters, kernel_size, conv_stride,
    conv_border_mode, units, output_dim=29):
    """ Build a recurrent + convolutional network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # Add convolutional layer
    conv_1d = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     activation='relu',
                     name='conv1d')(input_data)
    # Add batch normalization
    bn_cnn = BatchNormalization(name='bn_conv_1d')(conv_1d)
    # Add a recurrent layer
    simp_rnn = SimpleRNN(units, activation='relu',
        return_sequences=True, implementation=2, name='rnn')(bn_cnn)
    # TODO: Add batch normalization
    bn_rnn = BatchNormalization()(simp_rnn)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: cnn_output_length(
        x, kernel_size, conv_border_mode, conv_stride)
    print(model.summary())
    return model

def cnn_output_length(input_length, filter_size, border_mode, stride,
                       dilation=1, num_cnns=1, pool_size=0):
    """ Compute the length of the output sequence after 1D convolution along
        time. Note that this function is in line with the function used in
        Convolution1D class from Keras.
    Params:
        input_length (int): Length of the input sequence.
        filter_size (int): Width of the convolution kernel.
        border_mode (str): Only support `same` or `valid`.
        stride (int): Stride size used in 1D convolution.
        dilation (int)
    """
    if input_length is None:
        return None
    assert border_mode in {'same', 'valid'}
    dilated_filter_size = filter_size + (filter_size - 1) * (dilation - 1)
    if border_mode == 'same':
        output_length = input_length
    elif border_mode == 'valid':
        # I had to tweak this a bit to get it to play
        # nice with my final model's deep CNN
        output_length = input_length - num_cnns*(dilated_filter_size + 1) - (pool_size + 1)
    return (output_length + stride - 1) // stride

def deep_rnn_model(input_dim, units, recur_layers=2, output_dim=29):
    """ Build a deep recurrent network for speech 
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add recurrent layers, each with batch normalization
    link = input_data
    for i in range(recur_layers):
        rnn = CuDNNGRU(units, return_sequences=True, name='rnn{}'.format(i+1))(link)
        bn_rnn = BatchNormalization(name='bn_rnn{}'.format(i+1), )(rnn)
        link = bn_rnn
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(link)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def bidirectional_rnn_model(input_dim, units, output_dim=29):
    """ Build a bidirectional recurrent network for speech
    """
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    # TODO: Add bidirectional recurrent layer
    bidir_rnn =  Bidirectional(CuDNNGRU(units, return_sequences=True))(input_data)
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bidir_rnn)
    # Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    model.output_length = lambda x: x
    print(model.summary())
    return model

def no_noise(input_dim, units, output_dim=29):
    """ Build a deep network for speech 
    """
    filters = 256
    kernel_size = 11
    conv_stride = 1
    conv_border_mode = 'valid'
    conv_dilation_rate = 1
    num_cnns = 2
    pool_size = 2
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layers
    cnn1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn1')(input_data)
    cnn2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn2')(cnn1)
    # Add batch normalization, maxpooling and dropout
    bn_cnn = BatchNormalization(name='bn_cnn1')(cnn2)
    mp_cnn = MaxPooling1D(pool_size=2)(bn_cnn)
    do_cnn = Dropout(0.2)(mp_cnn)
    
    # Bidirectional GRU layers
    rnn1 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.3, recurrent_dropout=0), name='bidir_rnn_1')(do_cnn)
    bn_rnn1 = BatchNormalization(name='bidir_bn_1')(rnn1)
    rnn2 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.3, recurrent_dropout=0), name='bidir_rnn_2')(bn_rnn1)
    bn_rnn2 = BatchNormalization(name='bidir_bn_2')(rnn2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)#(do_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode,conv_stride, dilation=conv_dilation_rate,
                                                      num_cnns=num_cnns, pool_size=pool_size)//2
    print(model.summary())
    return model

def no_dropout(input_dim, units, output_dim=29):
    """ Build a deep network for speech 
    """
    filters = 256
    kernel_size = 11
    conv_stride = 1
    conv_border_mode = 'valid'
    conv_dilation_rate = 1
    num_cnns = 2
    pool_size = 2
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    noise = GaussianNoise(0.5)(input_data)
    
    # Add convolutional layers
    cnn1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn1')(noise)
    cnn2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn2')(cnn1)
    # Add batch normalization, maxpooling and dropout
    bn_cnn = BatchNormalization(name='bn_cnn1')(cnn2)
    mp_cnn = MaxPooling1D(pool_size=2)(bn_cnn)
    
    # Bidirectional GRU layers
    rnn1 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, recurrent_dropout=0), name='bidir_rnn_1')(mp_cnn)
    bn_rnn1 = BatchNormalization(name='bidir_bn_1')(rnn1)
    rnn2 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, recurrent_dropout=0), name='bidir_rnn_2')(bn_rnn1)
    bn_rnn2 = BatchNormalization(name='bidir_bn_2')(rnn2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)#(do_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode,conv_stride, dilation=conv_dilation_rate,
                                                      num_cnns=num_cnns, pool_size=pool_size)//2
    print(model.summary())
    return model

def no_noise_no_dropout(input_dim, units, output_dim=29):
    """ Build a deep network for speech 
    """
    filters = 256
    kernel_size = 11
    conv_stride = 1
    conv_border_mode = 'valid'
    conv_dilation_rate = 1
    num_cnns = 2
    pool_size = 2
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    
    # Add convolutional layers
    cnn1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn1')(input_data)
    cnn2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn2')(cnn1)
    # Add batch normalization, maxpooling and dropout
    bn_cnn = BatchNormalization(name='bn_cnn1')(cnn2)
    mp_cnn = MaxPooling1D(pool_size=2)(bn_cnn)
    
    # Bidirectional GRU layers
    rnn1 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2), name='bidir_rnn_1')(mp_cnn)
    bn_rnn1 = BatchNormalization(name='bidir_bn_1')(rnn1)
    rnn2 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2), name='bidir_rnn_2')(bn_rnn1)
    bn_rnn2 = BatchNormalization(name='bidir_bn_2')(rnn2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)#(do_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode,conv_stride, dilation=conv_dilation_rate,
                                                      num_cnns=num_cnns, pool_size=pool_size)//2
    print(model.summary())
    return model

def final_model(input_dim, units, output_dim=29):
    """ Build a deep network for speech 
    """
    filters = 256
    kernel_size = 11
    conv_stride = 1
    conv_border_mode = 'valid'
    conv_dilation_rate = 1
    num_cnns = 2
    pool_size = 2
    # Main acoustic input
    input_data = Input(name='the_input', shape=(None, input_dim))
    noise = GaussianNoise(0.5)(input_data)
    
    # Add convolutional layers
    cnn1 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn1')(noise)
    cnn2 = Conv1D(filters, kernel_size, 
                     strides=conv_stride, 
                     padding=conv_border_mode,
                     dilation_rate = conv_dilation_rate,
                     activation='relu',
                     name='cnn2')(cnn1)
    # Add batch normalization, maxpooling and dropout
    bn_cnn = BatchNormalization(name='bn_cnn1')(cnn2)
    mp_cnn = MaxPooling1D(pool_size=2)(bn_cnn)
    do_cnn = Dropout(0.2)(mp_cnn)
    
    # Bidirectional GRU layers
    rnn1 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.3, recurrent_dropout=0), name='bidir_rnn_1')(do_cnn)
    bn_rnn1 = BatchNormalization(name='bidir_bn_1')(rnn1)
    rnn2 = Bidirectional(GRU(units, return_sequences=True, activation='relu',
        implementation=2, dropout=0.3, recurrent_dropout=0), name='bidir_rnn_2')(bn_rnn1)
    bn_rnn2 = BatchNormalization(name='bidir_bn_2')(rnn2)
    
    # TODO: Add a TimeDistributed(Dense(output_dim)) layer
    time_dense = TimeDistributed(Dense(output_dim))(bn_rnn2)#(do_rnn)
    # TODO: Add softmax activation layer
    y_pred = Activation('softmax', name='softmax')(time_dense)
    # Specify the model
    model = Model(inputs=input_data, outputs=y_pred)
    # TODO: Specify model.output_length
    
    model.output_length = lambda x: cnn_output_length(x, kernel_size, 
                                                      conv_border_mode,conv_stride, dilation=conv_dilation_rate,
                                                      num_cnns=num_cnns, pool_size=pool_size)//2
    print(model.summary())
    return model