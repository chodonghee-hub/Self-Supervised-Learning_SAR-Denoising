from util import select_email_provider, send_email_to, show, plot_images, plot_tensors, create_montage
from skimage.morphology import disk
from skimage.filters import gaussian, median
from skimage.util import random_noise
from skimage.metrics import peak_signal_noise_ratio as psnr
from skimage.transform import resize
from mask import Masker
from models.dncnn import DnCNN
from torch.nn import MSELoss
from torch.optim import Adam
from tqdm import tqdm
from PIL import Image

import numpy as np
import matplotlib.pyplot as plt
import cv2
import torch
import os 
import openpyxl
import shutil
import time 
import smtplib


class SSupervised(object) : 
        
    def __init__(self, params) : 
        plt.rc('figure', figsize = (5,5))
        self.p = params
        self.select_mode = params.selector
        self.SMTP = ''

        if params.selector == 'train' : 
            self.__set_train__()
        elif params.selector == 'test' : 
            self.__set_test__()
      
        # =============================================
        #   이미지 읽기, 그레이스케일 변환, 사이즈 조절 
        # =============================================
        self.img_gray = cv2.imread(os.path.join(self.p.img_path, self.p.img_file_name)) 
        self.img_cv_gray = cv2.cvtColor(self.img_gray, cv2.COLOR_BGR2GRAY)
        self.img_resize = resize(self.img_cv_gray, (self.p.resize,self.p.resize))
        
        # =============================================
        #   노이즈 이미지 생성
        # =============================================
        self.noisy_image = random_noise(self.img_resize, mode = self.p.noise_mode, var=1)        
        # self.noisy_image = random_noise(self.img_resize, params = 'gaussian', var=3)        
        # self.noisy = torch.Tensor(self.noisy_image[np.newaxis, np.newaxis])         # ...     0728 test - without adding noise
        self.noisy = torch.Tensor(self.img_resize[np.newaxis, np.newaxis])

        # =============================================
        #   Masking 
        # =============================================
        self.masker = Masker(width = 10, mode=self.p.mask_mode)
        
        self.my_address, self.my_password, self.recv_address = select_email_provider()

        if self.my_address != '' : 
            self.SMTP = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            login_status, _ = self.SMTP.login(self.my_address, self.my_password)
            if login_status == 235 : 
                print(f'( V ) Authentication Success : {self.my_address}\n')
        
        # --------- GPU ---------
        device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f'\n( DEVICE ) : {device}')
        self.model = self.model.to(device)
        self.noisy = self.noisy.to(device)

    def __del__(self) : 
        if self.SMTP != '' : 
            self.SMTP.quit()
        print('\n\n** SMTP quit ! \n\n')

    def __set_train__(self) : 

        self.dir_title_by_date = ''
        self.record_train_time = "%02d-%02d" % (time.localtime().tm_hour, time.localtime().tm_min)
        self.excel_file_path = ''
        self.ckpt_file_path = ''
        self.target_excel = ''
        self.plt_image_title = ''
        self.first_cell_info = 10
        self.sheet_info_list = ['10', '5', '1']
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
        self.save_img_param = 0                                 # .. for save best image parameter
        self.learning_rate = self.p.lr
        self.epoch = self.p.epoch


        for sheet in self.sheet_info_list : 
            self.cell_info_dict[sheet] = self.first_cell_info - 1
            self.cell_update_dict[sheet] = 0
        self.model = DnCNN(1, num_of_layers = self.p.cnn_layer)
        sum(p.numel() for p in self.model.parameters() if p.requires_grad)


    def __set_test__(self) : 
        self.model = DnCNN(1, num_of_layers = self.p.cnn_layer)
        sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        pass
    
    # =============================================
    #   Training
    # =============================================
    def __train__(self) : 
        
        self.best_val_loss = 1
        best_psnr, idx_loss, idx_val_loss = 0, 0, 0 
        self.ckpt_cnt_by_epoch = 0
        div_point = 100 
        self.psnr_ls = dict() 
        loss_function = MSELoss()
        optimizer = Adam(self.model.parameters(), 
                        lr=self.learning_rate
                        )

         # CUDA support
        self.use_cuda = torch.cuda.is_available()
        if self.use_cuda:
            self.model = self.model.cuda()
            if self.trainable:
                loss_function = loss_function.cuda()

        
        i = 0
        for _ in range(self.epoch//div_point):
            target_dic = {'1': [], '5': [], '10': []}

            for _ in tqdm(range(div_point), desc="Train Process") :
                
                self.model.train()
            
                # net_input, mask = self.masker.mask(self.noisy, i % (self.masker.n_masks - 1))
                net_input, mask = self.masker.mask(self.noisy, i % (self.masker.n_masks))
                net_output = self.model(net_input)
                
                loss = loss_function(net_output*mask, self.noisy*mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
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
                    # best_psnr = psnr(denoised, self.noisy_image)        # ...     0728 test - without adding noise
                    best_psnr = psnr(denoised, self.img_resize)                    
                    self.best_images.append(denoised)
                    np.round(best_psnr, 2)
                    if np.round(best_psnr, 2) not in self.psnr_ls.keys() : 
                        self.psnr_ls[np.round(best_psnr, 2)] = i

                for div in target_dic.keys() : 
                    if i%int(div) == 0 : 
                        target_dic[div].append([f"{i}/{self.epoch}", np.round(best_psnr, 2), idx_loss, idx_val_loss])
                        self.cell_update_dict[div] += 1
                    else : 
                        break

                i += 1
                self.work_save_model(self.model, i)       # ... save ckpt - test 
                time.sleep(0.05)

            # update information & check end of epoch 
            if i % div_point == 0 and i > 0 : 
                self.__save_csv__(target_dic)
                print(f"\n\n ● [ {i}/{self.epoch} ]", end = '')
                print(f"{'LOSS'.rjust(15, ' ')}{str(idx_loss).rjust(10, ' ')}{'VAL LOSS'.rjust(15, ' ')}{str(idx_val_loss).rjust(10, ' ')}")
                print('='.ljust(65, '='))
                self.__save__(i)
                self.__update_info__()

           

    # =============================================
    #   Save Image - Sequence
    # =============================================
    def __save__(self, _epoch) : 
        assert 'images' in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}'), f'\n\n** No such directory : " images "'
        if len(self.best_images[self.save_img_param:len(self.best_images)]) > 1 : 
            plot_images(self.best_images[self.save_img_param:len(self.best_images)])
            self.save_img_param = len(self.best_images)
            savePath = f'./results/{self.dir_title_by_date}/{self.record_train_time}/images/sequence_{self.plt_image_title}-{_epoch}.png'
            plt.savefig(savePath)

            idx_subject = f"[ Training {self.plt_image_title}-{_epoch} ] → sequence_{self.plt_image_title}-{_epoch}.png"
            idx_data = f"[ EPOCH : {self.plt_image_title}-{_epoch} ]"
            if self.my_address != '' : 
                send_email_to(
                    _smtp = self.SMTP,
                    _my_address = self.my_address,
                    _subject = idx_subject,
                    _data = idx_data,
                    _img_path = savePath,
                    _recv_address = self.recv_address
                    )
            
            self.plt_image_title = _epoch
        
        # self.work_save_model(_epoch)       # ... save ckpt - test 


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

        assert 'result_format.xlsx' in os.listdir('./'), "\n\n** CSV result format not exists - check file name : result_format.xlsx"
        self.excel_file_path = f'./results/{self.dir_title_by_date}/{self.record_train_time}'

        if self.record_train_time not in os.listdir(f'./results/{self.dir_title_by_date}/') : 
            os.mkdir(f"{self.excel_file_path}")
            shutil.copy('./result_format.xlsx', f'{self.excel_file_path}/train_log.xlsx')
        
        if 'images' not in os.listdir(f'{self.excel_file_path}') : 
            os.mkdir(f'{self.excel_file_path}/images')

        if 'ckpt' not in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}') :
            os.mkdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}/ckpt')

    # =============================================
    #   CSV Update Information  
    # =============================================
    def __update_info__(self) : 
        print(f"{' ( Besat PSNR )'.ljust(20, ' ')}{'PSNR : '.rjust(20, ' ')}{str(max(self.psnr_ls.keys())).rjust(5, ' ')}{'EPOCH : '.rjust(15, ' ')}{str(self.psnr_ls.get(max(self.psnr_ls.keys()))).rjust(5, ' ')}")
        print(f" ○ {'Saved Check point : '.ljust(31, ' ')}{str(self.ckpt_cnt_by_epoch).rjust(30, ' ')}")        
        self.ckpt_cnt_by_epoch = 0 
        print(f" {'○ Sheet Title'.ljust(36,' ')}", end = '')
        for key, val in self.cell_update_dict.items() : 
            print(f"{key} ({val})".rjust(9,' '), end = '')
            self.cell_update_dict[key] = 0 
        print(f"\n ○ {'Save record at'.ljust(20, ' ')}{str(f'{self.target_excel}').rjust(40, ' ')}\n\n")
    

    # =============================================
    #   Save CSV
    # =============================================
    def __save_csv__(self, _save_target) : 
        if self.target_excel == '' : 
            self.target_excel = f'{self.excel_file_path}/train_log.xlsx'

        for _sheet_flag in _save_target.keys() : 
            wb = openpyxl.load_workbook(f'{self.target_excel}')
            for [_epoch, _psnr, _loss, _val_loss] in _save_target[_sheet_flag]: 

                self.cell_info_dict[_sheet_flag] += 1
                cell_point = self.cell_info_dict[_sheet_flag]
                
                ws = wb[_sheet_flag]
                ws[f'{self.cell_EPOCH}{cell_point}'] = _epoch
                ws[f'{self.cell_PSNR}{cell_point}'] = _psnr
                ws[f'{self.cell_LOSS}{cell_point}'] = _loss
                ws[f'{self.cell_VALID_LOSS}{cell_point}'] = _val_loss

            wb.save(f'{self.target_excel}')


    # =============================================
    #   Save Model - Test 
    # =============================================
    def work_save_model(self, _model, _epoch):
        CKPT_PATH = f'./results/{self.dir_title_by_date}/{self.record_train_time}/ckpt'

        fname = '{}/ssl-epoch{}.pt'.format(CKPT_PATH, _epoch)
        self.ckpt_cnt_by_epoch += 1
        # torch.save(self.model.state_dict(), fname)
        torch.save(_model.state_dict(), fname)


    # =============================================
    #   Load Model - Test 
    # =============================================
    def work_load_model(self, ckpt_fname):
        print('Loading checkpoint from: {}'.format(ckpt_fname))
        r'''
        if torch.cuda.is_available():
            self.model.load_state_dict(torch.load(ckpt_fname))
        else:
            self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'))
        '''
        self.model.load_state_dict(torch.load(ckpt_fname, map_location='cpu'), strict=False)
        # self.model = torch.load(ckpt_fname)
        self.model.eval()

        self.work_test()

    def work_test(self):
        """Evaluates denoiser on test set."""

        # self.model.train(False)

        denoised = np.clip(self.model(self.noisy).detach().cpu().numpy()[0, 0], 0, 1).astype(np.float64)
        self.best_images.append(denoised)
        self.__save__(1)

        

    # =============================================
    #   Main
    # =============================================
    def __start__(self, save_file = None) :
        self.__set_default_dirs__()
        if self.select_mode == 'train' : 
            self.__train__()
            # self.__save__()
        elif self.select_mode == 'test' : 
            self.work_test(save_file)
