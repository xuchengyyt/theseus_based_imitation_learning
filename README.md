本仓主要实现了一个基于theseus库的模仿学习
主要用来验证theseus在调参以及梯度反传方面的可行性
代码首先基于casadi实现一个简单的避障路径规划
以casadi生成的轨迹作为基准轨迹
通过theseus库来对MPC的调参，实现对casadi生成的轨迹来模仿
