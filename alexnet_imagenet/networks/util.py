import torch.nn as nn
import torch
import numpy
from sklearn.linear_model import LinearRegression
import platform


class BinOp():
    def __init__(self, model):
        self.base_number = model.base_number
        # count the number of Conv2d and Linear
        count_targets = 0
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                count_targets = count_targets + 1

        start_range = 1
        end_range = count_targets - 2
        self.bin_range = numpy.linspace(start_range,
                                        end_range, end_range - start_range + 1) \
            .astype('int').tolist()
        self.num_of_params = len(self.bin_range)
        self.saved_params = []
        self.target_params = []
        self.target_modules = []
        self.alphas = []
        index = -1
        for m in model.modules():
            if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
                index = index + 1
                if index in self.bin_range:
                    tmp = m.weight.data.clone()
                    self.saved_params.append(tmp)
                    self.target_modules.append(m.weight)

        for index_conv in range(int(self.num_of_params / self.base_number)):
            self.alphas.append(torch.zeros(self.base_number))

    def binarization(self):
        # self.meancenterConvParams()
        self.clampConvParams()
        self.save_params()
        self.binarizeConvParams()

    def clampConvParams(self):
        for index in range(int(self.num_of_params / self.base_number)):
            self.target_modules[index * self.base_number].data.clamp(-1.0, 1.0,
                                                                     out=self.target_modules[
                                                                         index * self.base_number].data)

    def save_params(self):
        for index in range(int(self.num_of_params / self.base_number)):
            self.saved_params[index * self.base_number].copy_(self.target_modules[index * self.base_number].data)

    def ABC_binarizeConvParams(self):
        for index_conv in range(int(self.num_of_params / self.base_number)):
            n_vec = self.target_modules[index_conv * self.base_number].data.nelement()
            k_size = self.target_modules[index_conv * self.base_number].data.size()

            W = self.target_modules[index_conv * self.base_number].data.view(n_vec)

            W_neg_mean = W.mean(dim=0, keepdim=True).neg().expand(n_vec)
            W_std = W.std(dim=0, keepdim=True).expand(n_vec)
            if self.base_number == 1:
                B = W.add(W_neg_mean).sign().view(1, n_vec)
            if self.base_number == 3:
                t1 = W.add(W_neg_mean).add(W_std.mul(-1)).sign().view(1, n_vec)
                t2 = W.add(W_neg_mean).sign().view(1, n_vec)
                t3 = W.add(W_neg_mean).add(W_std).sign().view(1, n_vec)
                B = torch.cat((t1, t2, t3))
            # for base in range(self.base_number):
            #     u_i=-1 + base * 2 / (self.base_number-1)
            #     t=W.add(W_neg_mean).add(W_std.mul(u_i)).sign()
            #     if base==0:
            #         B=t.view(1,n_vec)
            #     else:
            #         B=torch.cat((B,t.view(1,n_vec)))
            LRM = LinearRegression()
            LRM.fit(B.t(), W)
            # alpha = torch.from_numpy(LRM.coef_)
            if platform.system() == "Windows":
                alpha = torch.Tensor(LRM.coef_)
            else:
                alpha = torch.Tensor(LRM.coef_).cuda()

            self.alphas[index_conv].copy_(alpha)
            for base in range(self.base_number):
                self.target_modules[index_conv * self.base_number + base].data.copy_(
                    B[base].mul(alpha[base]).view(k_size))

    def ABC_updateBinaryGradWeight(self):
        # original version:
        for index_conv in range(int(self.num_of_params / self.base_number)):
            if self.base_number == 1:
                pass
            if self.base_number == 3:
                # explanation of dW=dB*alpha^2:
                # dB=d(L)/d(alpha*B)=1/alpha*d(L)/d(B)
                alpha_dB1 = self.target_modules[index_conv * self.base_number].grad.data. \
                    mul(self.alphas[index_conv][0] * self.alphas[index_conv][0])
                alpha_dB2 = self.target_modules[index_conv * self.base_number + 1].grad.data. \
                    mul(self.alphas[index_conv][1] * self.alphas[index_conv][1])
                alpha_dB3 = self.target_modules[index_conv * self.base_number + 2].grad.data. \
                    mul(self.alphas[index_conv][2] * self.alphas[index_conv][2])

                dW = alpha_dB1.add(alpha_dB2).add(alpha_dB3)
                # attach STE to single base OR the sum of them?
                W = self.target_modules[index_conv * self.base_number].data
                dW[W.lt(-1)] = 0
                dW[W.gt(1)] = 0
                dW.mul(1e+9)
                self.target_modules[index_conv * self.base_number].grad.data.copy_(dW)

    binarizeConvParams = ABC_binarizeConvParams
    updateBinaryGradWeight = ABC_updateBinaryGradWeight

    def restore(self):
        for index in range(int(self.num_of_params / self.base_number)):
            self.target_modules[index * self.base_number].data.copy_(self.saved_params[index * self.base_number])

    # def binarizeConvParams(self):
    #     for index in range(self.num_of_params):
    #
    #         n = self.target_modules[index].data[0].nelement()
    #         s = self.target_modules[index].data.size()
    #         if len(s) == 4:
    #             m = self.target_modules[index].data.norm(1, 3, keepdim=True) \
    #                 .sum(2, keepdim=True).sum(1, keepdim=True).div(n)
    #         elif len(s) == 2:
    #             m = self.target_modules[index].data.norm(1, 1, keepdim=True).div(n)
    #         self.target_modules[index].data.sign() \
    #             .mul(m.expand(s), out=self.target_modules[index].data)

    # def updateBinaryGradWeight(self):
    #     for index in range(self.num_of_params):
    #         weight = self.target_modules[index].data
    #         n = weight[0].nelement()
    #         s = weight.size()
    #         if len(s) == 4:
    #             m = weight.norm(1, 3, keepdim=True) \
    #                 .sum(2, keepdim=True).sum(1, keepdim=True).div(n).(s)
    #         elif len(s) == 2:
    #             m = weight.norm(1, 1, keepdim=True).div(n).expand(s)
    #         # m=alpha
    #
    #         m[weight.lt(-1.0)] = 0
    #         m[weight.gt(1.0)] = 0
    #         # m=alpha*r_1
    #
    #         m = m.mul(self.target_modules[index].grad.data)
    #         # m=alpha*r_1*gi
    #
    #         m_add = weight.sign().mul(self.target_modules[index].grad.data)
    #         # m_add=gi*sign(w_i)
    #         if len(s) == 4:
    #             m_add = m_add.sum(3, keepdim=True).sum(2, keepdim=True).sum(1, keepdim=True).div(n).expand(s)
    #         elif len(s) == 2:
    #             m_add = m_add.sum(1, keepdim=True).div(n).expand(s)
    #         # in the notes, W_i stands for the i^th element of  W(C*k*k)
    #
    #         m_add = m_add.mul(weight.sign())
    #         self.target_modules[index].grad.data = m.add(m_add).mul(1.0 - 1.0 / s[1]).mul(n)
    #         self.target_modules[index].grad.data = self.target_modules[index].grad.data.mul(1e+9)
    #
    # def meancenterConvParams(self):
    #     for index in range(int(self.num_of_params/self.base_number)):
    #         s = self.target_modules[index*self.base_number].data.size()
    #         negMean = self.target_modules[index*self.base_number].data.mean(1, keepdim=True). \
    #             mul(-1).expand_as(self.target_modules[index*self.base_number].data)
    #         self.target_modules[index*self.base_number].data = self.target_modules[index*self.base_number].data.add(negMean)
    #
