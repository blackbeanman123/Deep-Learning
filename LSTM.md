# LSTM

explanation:http://colah.github.io/posts/2015-08-Understanding-LSTMs/

tips:

1. f\_t, i\_t, o_t are matrixs have values between 0~1; Cell state -> -1~1
2. the selective part:New_Cell_State =  sigmoid([h_t-1, x_t]) (·) Old_Cell_State

usage: 

1. 忘记阶段
2. 选择记忆阶段
3. 输出阶段