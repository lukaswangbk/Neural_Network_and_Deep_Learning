import tensorflow as tf

def LSTM_step(cell_inputs, cell_states, kernel, recurrent_kernel, bias):
    """
    Run one time step of the cell. That is, given the current inputs and the cell states from the last time step, calculate the current state and cell output.
    You will notice that TensorFlow LSTMCell has a lot of other features. But we will not try them. Focus on the very basic LSTM functionality.
    Hint: In LSTM there exist both matrix multiplication and element-wise multiplication. Try not to mix them.
        
        
    :param cell_inputs: The input at the current time step. The last dimension of it should be 1.
    :param cell_states:  The state value of the cell from the last time step, containing previous hidden state h_tml and cell state c_tml.
    :param kernel: The kernel matrix for the multiplication with cell_inputs
    :param recurrent_kernel: The kernel matrix for the multiplication with hidden state h_tml
    :param bias: Common bias value
    
    
    :return: current hidden state, and a list of hidden state and cell state. For details check TensorFlow LSTMCell class.
    """
    
    
    ###################################################
    # TODO:      INSERT YOUR CODE BELOW               #
    # params                                          #
    ###################################################
    
    # Get idea from https://github.com/keras-team/keras/blob/v2.7.0/keras/layers/recurrent_v2.py#L944-L1276
    h_tm1 = cell_states[0]
    c_tm1 = cell_states[1]

    z = tf.tensordot(cell_inputs, kernel, axes=1)
    z += tf.tensordot(h_tm1, recurrent_kernel, axes=1)
    z += bias

    z0, z1, z2, z3 = tf.split(z, 4, axis=1)

    i = tf.sigmoid(z0)
    f = tf.sigmoid(z1)
    c = f * c_tm1 + i * tf.tanh(z2)
    o = tf.sigmoid(z3)

    h = o * tf.tanh(c)
    return h, [h, c]
    
    ###################################################
    # END TODO                                        #
    ###################################################
