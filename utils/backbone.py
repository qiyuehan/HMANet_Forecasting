import torch
import torch.nn as nn
from utils.Patch_embed import PatchEmbedding
from utils.multi_deformable_conv import Multi_Def_Conv


class SeqToPatches(nn.Module):
    def __init__(self, patch_size):
        super().__init__()
        self.patch_size = patch_size
        self.unfold = nn.Unfold(kernel_size=patch_size, stride=patch_size)
    def forward(self, x_enc):
        assert len(x_enc.size()) == 4
        s2p = self.unfold(x_enc)
        s2p = s2p.permute(0, 2, 1)
        return s2p

class Model(nn.Module):

    def __init__(self, configs, chunk_size=24):
        """
        chunk_size: int, reshape T into [num_chunks, chunk_size]
        """
        super(Model, self).__init__()
        self.task_name = configs.task_name
        self.pre_train = configs.pre_train
        self.enc_in = configs.enc_in

        self.mask_ratio = configs.mask_rate
        self.kernel = configs.kernel_size
        self.patch_len = configs.patch_len
        self.num_group = (max(configs.pre_train, self.patch_len) - self.patch_len) // self.patch_len + 1
        self.mul_deformable_conv = Multi_Def_Conv(in_channels=self.num_group, out_channels=self.num_group,
                                                  dilation_rates=configs.dilation_rate)
        self.ff = nn.Sequential(nn.Conv2d(self.num_group, self.num_group, kernel_size=3, padding=1, stride=1),
                                nn.Dropout(configs.dropout),
                                )
        self.bn = nn.BatchNorm2d(self.num_group)
        self.relu = nn.LeakyReLU()
        self.dropout = nn.Dropout(configs.dropout)
        self.proj = nn.Linear(self.pre_train, self.pre_train)
        self.patch_embedding = PatchEmbedding(
                     configs.d_model, patch_len=self.kernel[0],stride=1, padding=0, dropout=configs.dropout)
        self.share_p = nn.Conv2d(self.num_group, self.num_group, 1)
        self.block_linear = nn.Linear(self.num_group, self.num_group)


    def encoder(self, x_patch_masked,epoch, i):

        b, num_p, block_len, n = x_patch_masked.shape
        x_block_mask = x_patch_masked.permute(0,2,3,1)
        group_inner = self.block_linear(x_block_mask).permute(0,3,1,2)
        mul_deformabel = self.mul_deformable_conv(x_patch_masked)  # [32, 88, 8, 12]
        repres = group_inner + group_inner * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]
        repres = self.ff(repres)
        all_fea = repres.reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)
        output2 = self.proj(output.permute(0,2,1)).permute(0, 2, 1)
        return output

    def encoder_ori(self, x_patch_masked,epoch, i):
        b, num_p,n, patch_len = x_patch_masked.shape
        group_inner = self.share_p(x_patch_masked)

        # Deformable conv
        mul_deformabel = self.mul_deformable_conv(x_patch_masked)  # [32, 88, 8, 12]
        repres = group_inner + x_patch_masked * torch.softmax(mul_deformabel, dim=1) #[32, 88, 8, 12]
        repres = self.ff(repres)
        all_fea = repres.permute(0, 1, 3, 2).reshape(b, -1, n)
        output = self.proj(all_fea.permute(0,2,1)).permute(0, 2, 1)
        return output

    def forecast(self, x_enc, epoch, i):
        return self.encoder(x_enc, epoch, i)

    def forward(self, x_enc, epoch, i):
        if self.task_name == 'long_term_forecast' or self.task_name == 'short_term_forecast':
            dec_out = self.forecast(x_enc, epoch,i)
            return dec_out
        return None
