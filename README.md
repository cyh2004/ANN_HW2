# ANN_HW2
+ mlp
  + 取消BatchNorm层
    `python main.py --noBatchNorm`
  + 取消Dropout层
    `python main.py --noDropout`
+ cnn
  + 取消BatchNorm层
    `python main.py --noBatchNorm`
  + 取消Dropout层
    `python main.py --noDropout`
  + 将BatchNorm层移到ReLU层后执行
    `python main.py --switch 1`
  + 将Dropout层移到Maxpool层后执行
    `python main.py --switch 2`
  + 调整学习率与drop_rate
    `python main.py --learning_rate <num> --drop_rate <num>`
