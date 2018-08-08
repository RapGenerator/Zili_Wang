##8.6
### BiLstm
* 对encoder采用BiLstm进行编码，将hidden_state concat后进行dense，效果：可以对encoder的信息增加

### 主题生成
* 采用了LDA的方法对主题进行抽取，之后将生成的keywords喂给encoder


### Skip-thought vector
* 采用两个decoder，因此，对于loss等的计算需要进行学习

### revese
* 对数据集采用revese方法进行encoder，押韵效果提高！

##8.7 
###BiLstm
* 完成了BilSTM——encoder
* 需要keyword 个人认为不能采用百度那篇文章
* skip-thought Vector中的两个decoder存在如果inference的问题

##8.8
###SeqGan
开坑这个，今天demo跑起来了，生成押韵是一个很重要的问题



