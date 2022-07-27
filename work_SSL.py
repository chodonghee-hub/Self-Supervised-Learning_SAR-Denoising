from util import show, plot_images, plot_tensors
from skimage.morphology import disk
from skimage.filters import gaussian, median
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize
from mask import Masker
from models.dncnn import DnCNN
from torch.nn import MSELoss
from torch.optim import Adam
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os 
import openpyxl
import shutil
import time 

class SSupervised(object) : 
        
    def __init__(self, params) : 
        plt.rc('figure', figsize = (5,5))

        self.dir_title_by_date = ''
        self.record_train_time = ''
        self.excel_file_path = ''
        self.target_excel = ''
        self.first_cell_info = 10
        self.sheet_info_list = ['100%', '50%', '25%', '10%', '1%']
        self.cell_info_dict = dict()
        self.cell_update_dict = dict()
        self.csv_first_cell = 'C'
        self.cell_EPOCH = self.csv_first_cell
        self.cell_PSNR = chr(ord(self.csv_first_cell)+1)
        self.cell_LOSS = chr(ord(self.csv_first_cell)+2)
        self.cell_VALID_LOSS = chr(ord(self.csv_first_cell)+3)

        self.losses = []
        self.val_losses = []
        self.best_images = []
        self.best_val_loss = 1
        self.learning_rate = params.lr
        self.epoch = params.epoch

        for sheet in self.sheet_info_list : 
            self.cell_info_dict[sheet] = self.first_cell_info - 1
            self.cell_update_dict[sheet] = 0

        # =============================================
        #   이미지 읽기, 그레이스케일 변환, 사이즈 조절 
        # =============================================
        self.img_gray = cv2.imread(os.path.join(params.img_path, params.img_file_name)) 
        self.img_cv_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_BGR2GRAY)
        self.img_resize = resize(self.img_cv_gray, (params.resize,params.resize))
        
        # =============================================
        #   노이즈 이미지 생성
        # =============================================
        self.noisy_image = random_noise(self.img_resize, mode = params.noise_mode, var=3)        
        # self.noisy_image = random_noise(self.img_resize, params = 'gaussian', var=3)        
        self.noisy = torch.Tensor(self.noisy_image[np.newaxis, np.newaxis])


        # =============================================
        #   Masking 
        # =============================================
        self.masker = Masker(width = 4, mode=params.mask_mode)


        # =============================================
        #   Model 
        # =============================================
        self.model = DnCNN(1, num_of_layers = params.cnn_layer)
        sum(p.numel() for p in self.model.parameters() if p.requires_grad)

        # --------- GPU ---------
        # model = model.to(device)
        # noisy = noisy.to(device)


    # =============================================
    #   Training
    # =============================================
    def __train__(self) : 
        self.losses = []
        self.val_losses = []
        self.best_images = []
        self.best_val_loss = 1
        best_psnr = 0 
        loss_function = MSELoss()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        now = time.localtime() 
        self.record_train_time = "%02d-%02d" % (now.tm_hour, now.tm_min)
        
        for i in range(self.epoch):
            self.model.train()
            
            net_input, mask = self.masker.mask(self.noisy, i % (self.masker.n_masks - 1))
            net_output = self.model(net_input)
            
            loss = loss_function(net_output*mask, self.noisy*mask)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            if i % 10 == 0 :
                self.losses.append(loss.item())
                self.model.eval()
                
                net_input, mask = self.masker.mask(self.noisy, self.masker.n_masks - 1)
                net_output = self.model(net_input)
            
                val_loss = loss_function(net_output*mask, self.noisy*mask)
                
                self.val_losses.append(val_loss.item())
                
                idx_loss = round(loss.item(), 5)
                idx_val_loss = round(val_loss.item(), 5)
                

                if val_loss < self.best_val_loss:
                    self.best_val_loss = val_loss
                    denoised = np.clip(self.model(self.noisy).detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
                    best_psnr = psnr(denoised, self.noisy_image)
                    self.best_images.append(denoised)
                    np.round(best_psnr, 2)
                    print(" ( ! ) Update Model PSNR : ", np.round(best_psnr, 2))

            # 100% 50% 25% 10% 1%
            if 0 in [i%int(self.epoch*1), i%int(self.epoch*0.5), i%int(self.epoch*0.25), i%int(self.epoch*0.1), i%int(self.epoch*0.01)] : 
                for per in [1, 0.5, 0.25, 0.1, 0.01] : 
                    if i%(self.epoch*per) == 0 : 
                        self.__save_csv__(f'{int(per*100)}%', f"{i}/{self.epoch}", np.round(best_psnr, 2), idx_loss, idx_val_loss)
                        self.cell_update_dict[f'{int(per*100)}%'] += 1

            # update information & check end of epoch 
            if i % 100 == 0 or i+1 == self.epoch : 
                print(f" ( {i}/{self.epoch} )", end = '')
                print(f"{'LOSS'.rjust(15, ' ')}{str(idx_loss).rjust(10, ' ')}{'VAL LOSS'.rjust(15, ' ')}{str(idx_val_loss).rjust(10, ' ')}")
                print('='.ljust(65, '='))
                self.__update_info__()


    # =============================================
    #   Save 
    # =============================================
    def __save__(self) : 
        if 'images' not in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}') : 
            os.mkdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}/images')
        
        plot_images(self.best_images)
        plt.savefig(f'./results/{self.dir_title_by_date}/{self.record_train_time}/images/plot_images.png')


    # =============================================
    #   Make defualt directorys 
    # =============================================
    def __set_default_dirs__(self, _path = '.') : 
        now = time.localtime() 
        self.dir_title_by_date = "%04d-%02d-%02d" % (now.tm_year, now.tm_mon, now.tm_mday)

        if 'results' not in os.listdir(f'{_path}/') : 
            os.mkdir(f'{_path}/results')

        if self.dir_title_by_date not in os.listdir(f'{_path}/results') : 
            os.mkdir(f'{_path}/results/{self.dir_title_by_date}')
        

    def __update_info__(self) : 
        print(f" {'○ Sheet Title'.ljust(23,' ')}", end = '')
        for key, val in self.cell_update_dict.items() : 
            print(f"{key} ({val})".rjust(8,' '), end = '')
            self.cell_update_dict[key] = 0 
        print(f"\n ○ {'Save record at'.ljust(20, ' ')}{str(f'{self.excel_file_path}/{self.target_excel}').rjust(40, ' ')}\n\n")
    

    # =============================================
    #   Save CSV 
    # =============================================
    def __save_csv__(self, _sheet_flag, _epoch, _psnr, _loss, _val_loss) : 
        assert 'result_format.xlsx' in os.listdir('./'), "\n\n** CSV result format not exists - check file name : result_format.xlsx"
        if self.excel_file_path == '' : 
            self.excel_file_path = f'./results/{self.dir_title_by_date}/{self.record_train_time}'

        if self.record_train_time not in os.listdir(f'./results/{self.dir_title_by_date}/') : 
            os.mkdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}')
            shutil.copy('./result_format.xlsx', f'{self.excel_file_path}/train_log.xlsx')

        if self.target_excel == '' : 
            self.target_excel = os.listdir(self.excel_file_path)[0]
        wb = openpyxl.load_workbook(f'{self.excel_file_path}/{self.target_excel}')

        self.cell_info_dict[_sheet_flag] += 1
        cell_point = self.cell_info_dict[_sheet_flag]
        
        ws = wb[_sheet_flag]
        ws[f'{self.cell_EPOCH}{cell_point}'] = _epoch
        ws[f'{self.cell_PSNR}{cell_point}'] = _psnr
        ws[f'{self.cell_LOSS}{cell_point}'] = _loss
        ws[f'{self.cell_VALID_LOSS}{cell_point}'] = _val_loss

        wb.save(f'{self.excel_file_path}/{self.target_excel}')
    

    # =============================================
    #   Main
    # =============================================
    def __start__(self) :
        self.__set_default_dirs__()
        self.__train__()
        self.__save__()
