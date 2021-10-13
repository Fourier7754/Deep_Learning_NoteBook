import random
import torch as pt
import numpy as np


class linear_regression():
    def __init__(self, batch_size=None, learning_rate=None,
                 init_alpha: pt.tensor = None, init_beta: pt.tensor = None, epoch=None):

        self.batch_size = batch_size
        self.lr = learning_rate

        self.alpha = init_alpha
        self.alpha.requires_grad_(requires_grad=True)

        self.beta = init_beta
        self.beta.requires_grad_(requires_grad=True)

        self.epoch = epoch

    def generate_test_data(self, x_dim):
        alpha = pt.tensor((np.arange(x_dim) + 2), dtype=pt.float32).unsqueeze(1)
        beta = pt.tensor(2, dtype=pt.float32)
        x = pt.randn(4000, len(alpha), dtype=pt.float32)
        y = pt.mm(x, alpha) + beta
        y = y + pt.tensor(np.random.normal(0, 1, size=y.size()), dtype=pt.float32)
        print('vector_alpha=', alpha, '\n')
        print('scalar_beta=', beta, '\n')
        return x, y

    def generate_train_data(self, x, y):
        num_examples = len(x)
        indices = list(range(num_examples))
        random.shuffle(indices)  # 样本的读取顺序是随机的
        for i in range(0, num_examples, self.batch_size):
            j = pt.LongTensor(indices[i: min(i + self.batch_size, num_examples)])  # 最后一次可能不足一个batch
            yield x.index_select(0, j), y.index_select(0, j)

    def linear_model(self, x):
        value = pt.mm(self.alpha.unsqueeze(1).T, x.T) + self.beta
        return value

    """我们计算loss是对某个batch计算loss
        在定义输入数据时，我们就要对数据的格式进行要求
        行数对应变量序号，列对应各个变量     
                                    """

    def loss_function(self, y_hat, y):
        loss = (y_hat - y.view(y_hat.size())) ** 2 / 2
        return loss

    """此处使用的是mini-batch sdg
    在训练时对batch内的样本逐一计算损失，全部输出后将损失累加
    对累加的损失计算一个全局的梯度，随后在优化时，除batch-size得到batch平均梯度
    这部分如何编写仍存在疑义"""

    def sgd(self):
        # alpha是向量，反向传播的梯度与alpha形状一致，也是向量
        self.alpha.data = self.alpha.data - self.alpha.grad * self.lr / self.batch_size
        self.beta.data = self.beta.data - self.beta.grad * self.lr / self.batch_size

    def train_linear_model(self, x, y):
        for i in tqdm(range(self.epoch)):
            for train_data, train_value in self.generate_train_data(x, y):
                loss = self.loss_function(self.linear_model(train_data), train_value).sum()
                loss.backward()
                self.sgd()
                self.alpha.grad.data.zero_()
                self.beta.grad.data.zero_()
            loss_all_data = self.loss_function(self.linear_model(x), y).mean().item()

        print('loss=', loss_all_data)

if __name__ == "__main__":
    model_1=linear_regression(batch_size=10,epoch=200,learning_rate=0.0001,
                              init_alpha=pt.tensor([1,1,1],dtype=pt.float32),
                              init_beta=pt.tensor([1],dtype=pt.float32))
    x,y=model_1.generate_test_data(x_dim=3)
    model_1.train_linear_model(x,y)
    print(model_1.alpha)
    print(model_1.beta)




