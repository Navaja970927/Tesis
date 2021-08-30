from clases.ImbalancedPerformance import ImbalancedPerformanceClass
from clases.AlgorithmsDLplus.CNN import *
from clases.AlgorithmsDLplus.BPNN import *
from clases.AlgorithmsDLplus.AE import *
from clases.AlgorithmsDLplus.DAE import *
from clases.AlgorithmsDLplus.LSTM import *
from clases.AlgorithmsDLplus.RNN import *

ip = ImbalancedPerformanceClass()
ip.solve_imbalanced("creditcard.csv")
ip.epochs = 1

# Variantes con CNN detrás
# ip.performanceCNN_AE(CNN_AE(ip))
# ip.performanceCNN_BPNN(CNN_BPNN(ip))
# ip.performanceCNN_DAE(CNN_DAE(ip))
# ip.performanceCNN_LSTM(CNN_LSTM(ip))
# ip.performanceCNN_RNN(CNN_RNN(ip))

# Varaintes con BPNN detrás
# ip.performanceBPNN_CNN(BPNN_CNN(ip))
# ip.performanceBPNN_AE(BPNN_AE(ip))
# ip.performanceBPNN_DAE(BPNN_DAE(ip))

# Variantes con AE detrás
# ip.performanceAE_DAE(AE_DAE(ip))
# ip.performanceAE_BPNN(AE_BPNN(ip))

# Variantes con DAE detrás
# ip.performanceDAE_BPNN(DAE_BPNN(ip))
# ip.performanceDAE_AE(DAE_AE(ip))

# Variantes con RNN detrás
# ip.performanceRNN_AE(RNN_AE(ip))
# ip.performanceRNN_DAE(RNN_DAE(ip))
# ip.performanceRNN_BPNN(RNN_BPNN(ip))

# Variantes con LSTM detrás
# ip.performanceLSTM_AE(LSTM_AE(ip))
# ip.performanceLSTM_DAE(LSTM_DAE(ip))
# ip.performanceLSTM_BPNN(LSTM_BPNN(ip))

ip.show_comparison()
