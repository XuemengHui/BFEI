import torch

import model.network


class Model(object):
    def __init__(self, **params):
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')
        self.net = model.network.Network(
            classes=params.get('classes', 10),
            channels=params.get('channels', 1),
            dropout_rate=params.get('dropout_rate', 0.5)
        )
        self.net.to(self.device)

        self.lr = params.get('lr', 1e-3)
        self.lr_step = params.get('lr_step', [50])
        self.lr_decay = params.get('lr_decay', 0.1)

        self.lr_scheduler = None

        self.momentum = params.get('momentum', 0.9)
        self.weight_decay = params.get('weight_decay', 4e-3)
        self.lam1 = params.get('lam1', 1.0)
        self.lam2 = params.get('lam2', 1.0)
        self.lam3 = params.get('lam3', 1.0)
        self.criterion = torch.nn.CrossEntropyLoss()
        self.MSE = torch.nn.MSELoss()
        self.L1 = torch.nn.L1Loss()
        self.optimizer = torch.optim.SGD(
            self.net.parameters(),
            lr=self.lr,
            momentum=self.momentum,
            weight_decay=self.weight_decay
        )
        self.cls_num = params.get('classes', 10)
        if self.lr_decay:
            self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(
                optimizer=self.optimizer,
                milestones=self.lr_step,
                gamma=self.lr_decay
            )

    def optimize(self, x, asc, y):
        label_mat_list = torch.zeros([len(y), self.cls_num, self.cls_num])
        for i in range(len(y)):
            eyemat = torch.eye(self.cls_num, requires_grad=False).cuda()
            label_mat_list[i, :, :] = eyemat
        label_mat_list = torch.FloatTensor(
            label_mat_list).cuda()
        logits_fuse_img, logits_fuse_asc, logits_img, logits_asc, confusion_mat, confusion_mat2 = self.net(
            x.to(self.device), asc.to(self.device))
        loss_cls = self.criterion(logits_img, y.to(
            self.device)) + self.criterion(logits_fuse_img, y.to(self.device))
        loss_cls += self.lam1*(self.criterion(logits_fuse_asc, y.to(self.device)
                                              ) + self.criterion(logits_asc, y.to(self.device)))

        loss_aln = self.lam2*self.MSE(confusion_mat, label_mat_list)
        loss_co = self.lam3 * self.MSE(logits_asc, logits_img)
        loss = loss_cls + loss_aln + loss_co

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss_cls.item(), loss_aln.item(), loss_co.item()

    @torch.no_grad()
    def inference(self, x, asc):
        return self.net(x.to(self.device), asc.to(self.device))

    def save(self, path):
        torch.save(self.net.state_dict(), path)

    def load(self, path):
        self.net.load_state_dict(torch.load(path))
        self.net.eval()
