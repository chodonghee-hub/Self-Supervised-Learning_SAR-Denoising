from util import select_email_provider, send_email_to, show, plot_images, plot_tensors
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
# import smtplib


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

        self.my_address, self.my_password, self.recv_address = select_email_provider()
        r'''
        if self.my_address != '' : 
            self.SMTP = smtplib.SMTP_SSL("smtp.gmail.com", 465)
            login_status, _ = self.SMTP.login(self.my_address, self.my_password)
            if login_status == 235 : 
                print(f'( V ) Authentication Success : {self.my_address}\n')
        '''
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
        # self.noisy = torch.Tensor(self.noisy_image[np.newaxis, np.newaxis])         # ...     0728 test - without adding noise
        self.noisy = torch.Tensor(self.img_resize[np.newaxis, np.newaxis])


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
        
        self.best_val_loss = 1
        best_psnr, idx_loss, idx_val_loss = 0, 0, 0 
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

            if i % 10 == 0 and i > 0:
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
                    print(f" ( ! ) Update Model PSNR : {np.round(best_psnr, 2)}\n")
                    self.__save_single_img__(i, [denoised])

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
            
            time.sleep(1)


    # =============================================
    #   Training - Test
    # =============================================
    def work__train__(self) : 
        
        self.best_val_loss = 1
        best_psnr, idx_loss, idx_val_loss = 0, 0, 0 
        loss_function = MSELoss()
        optimizer = Adam(self.model.parameters(), lr=self.learning_rate)

        now = time.localtime() 
        self.record_train_time = "%02d-%02d" % (now.tm_hour, now.tm_min)
        
        div_point = 100 
        psnr_ls = dict() 

        i = 0
        for _ in range(self.epoch//div_point):
            
            for _ in tqdm(range(div_point), desc="Train Process") :

                self.model.train()
            
                net_input, mask = self.masker.mask(self.noisy, i % (self.masker.n_masks - 1))
                net_output = self.model(net_input)
                
                loss = loss_function(net_output*mask, self.noisy*mask)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                if i % 10 == 0 and i > 0:
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
                        # print(f" ( ! ) Update Model PSNR : {np.round(best_psnr, 2)}\n")           # ...       0729 test - skip print for tqdm 
                        if np.round(best_psnr, 2) not in psnr_ls.keys() : 
                            psnr_ls[np.round(best_psnr, 2)] = i
                        
                        self.__save_single_img__(i, [self.best_images[::1]], best_psnr, idx_val_loss)    # .. save error, for send email 
                        # self.__save__()

                # 100% 50% 25% 10% 1%
                if 0 in [i%int(self.epoch*1), i%int(self.epoch*0.5), i%int(self.epoch*0.25), i%int(self.epoch*0.1), i%int(self.epoch*0.01)] : 
                    for per in [1, 0.5, 0.25, 0.1, 0.01] : 
                        if i%(self.epoch*per) == 0 : 
                            self.__save_csv__(f'{int(per*100)}%', f"{i}/{self.epoch}", np.round(best_psnr, 2), idx_loss, idx_val_loss)
                            self.cell_update_dict[f'{int(per*100)}%'] += 1

                # update information & check end of epoch 
                if i % div_point == 0 and i > 0 : 
                    print(f" ( {i}/{self.epoch} )", end = '')
                    print(f"{'LOSS'.rjust(15, ' ')}{str(idx_loss).rjust(10, ' ')}{'VAL LOSS'.rjust(15, ' ')}{str(idx_val_loss).rjust(10, ' ')}")
                    print(f"{'○ Besat PSNR'.ljust(20, ' ')}{'PSNR : '.ljust(5, ' ')}{str(max(psnr_ls.keys())).ljust(15, ' ')}{'EPOCH : '.ljust(5, ' ')}{psnr_ls.get(max(psnr_ls.keys()))}")
                    print('='.ljust(65, '='))
                    self.__update_info__()
                    

                i += 1
                time.sleep(0.05)

    # =============================================
    #   Save Image - Sequence
    # =============================================
    def __save__(self) : 
        
        if 'images' not in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}') : 
            os.mkdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}/images')

        assert 'images' in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}'), f'\n\n** No such directory : " images "'
        plot_images(self.best_images)
        savePath = f'./results/{self.dir_title_by_date}/{self.record_train_time}/images/sequence_images.png'
        plt.savefig(savePath)
        # plt.savefig(savePath)

        idx_subject = f"[ Finish Training ] → sequence_images.png"
        idx_data = f"[ Finish Training ]"
        if self.my_address != '' : 
            send_email_to(
                # _smtp = self.SMTP,
                _my_address = self.my_address,
                _my_password = self.my_password,
                # _subject = f"[ Finish Training ] {time.strftime('%Y-%m-%d %H:%M:%S')} → sequence_images.png",
                _subject = idx_subject,
                _data = idx_data,
                _img_path = savePath,
                _recv_address = self.recv_address
                )


    # =============================================
    #   Save Image - Single
    # =============================================
    def __save_single_img__(self, _epoch, _img_target,  _psnr, _v_loss) : 
        if 'images' not in os.listdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}') : 
            os.mkdir(f'./results/{self.dir_title_by_date}/{self.record_train_time}/images')
        r'''
        plot_images(_img_target)
        # plt.imshow(_img_target, cmap=plt.cm.gray)
        savePath = f'./results/{self.dir_title_by_date}/{self.record_train_time}/images/EPOCH-{_epoch}.png'
        plt.savefig(savePath)
        # Image.fromarray(_img_target).save(savePath)
        '''

        if self.my_address != '' : 
            # idx_subject = str(f"[ Update Information ] → EPOCH-{_epoch}.png").encode('utf-8')
            idx_subject = f"[ Update Information ] → EPOCH-{_epoch}.png"
            idx_data = f"[ Update best cut information ] \n* PSNR : {_psnr} \n* VAL_LOSS : {_v_loss}"
            send_email_to(
                # _smtp = self.SMTP,
                _my_address = self.my_address,
                _my_password = self.my_password,
                # _subject = f"[ Update Information ] {time.strftime('%Y-%m-%d %H:%M:%S')} → EPOCH-{_epoch}.png",
                _subject = idx_subject,
                _data = idx_data,
                # _img_path = savePath,
                _recv_address = self.recv_address
                )


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
        

    # =============================================
    #   CSV Update Information  
    # =============================================
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
        r'''
        try : 
            self.__set_default_dirs__()
            self.__train__()
            self.__save__()
            self.SMTP.quit()

        except Exception as e : 
            print(f"** Error message : {e}")
            self.SMTP.quit()
        '''
        self.__set_default_dirs__()
        self.work__train__()
        self.__save__()
